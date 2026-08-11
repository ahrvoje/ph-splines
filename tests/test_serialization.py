"""Copy and pickle protocol tests for the immutable cubic family and NURBS
handles.

Immutability is a public-API contract; ordinary Python object handling -
``pickle``, ``copy.copy``, ``copy.deepcopy``, process transport - must work.
Restoration re-verifies invariants, so corrupted payloads fail typed.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from ph_spline import (
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    OffsetConstructionError,
    PHBSplineOpen,
)

OPEN_POINTS = [[0.0, 0.0], [1.0, 0.4], [2.0, 1.3], [2.6, 2.4]]
CLOSED_POINTS = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
S_POINTS = [
    [0.0, 0.0], [1.0, 0.8], [2.0, 1.0], [3.0, 0.2], [4.0, 0.0], [5.0, 0.8],
]


def _assert_same_geometry(a, b):
    assert type(a) is type(b)
    assert a.num_points == b.num_points
    assert a.length == b.length
    for u in np.linspace(0.0, 1.0, 41):
        u = float(u)
        assert np.array_equal(a.point(u), b.point(u))
        assert np.array_equal(a.tangent(u), b.tangent(u))
        assert a.signed_curvature(u) == b.signed_curvature(u)
    s = 0.37 * a.length
    assert a.parameter_at_length(s) == b.parameter_at_length(s)


@pytest.fixture(
    params=["open", "closed", "inflectional"],
    ids=["open", "closed", "inflectional"],
)
def curve(request):
    if request.param == "open":
        return CubicPHSplineOpen(OPEN_POINTS)
    if request.param == "closed":
        return CubicPHSplineClosed(CLOSED_POINTS)
    return CubicPHSplineOpen(S_POINTS)


def test_pickle_round_trip(curve):
    restored = pickle.loads(pickle.dumps(curve))
    _assert_same_geometry(curve, restored)


def test_deepcopy(curve):
    _assert_same_geometry(curve, copy.deepcopy(curve))


def test_shallow_copy(curve):
    _assert_same_geometry(curve, copy.copy(curve))


def test_restored_spline_is_still_immutable(curve):
    restored = pickle.loads(pickle.dumps(curve))
    with pytest.raises(AttributeError):
        restored._scale = 2.0
    with pytest.raises(AttributeError):
        del restored._segments


def test_restored_offset_matches(curve):
    restored = pickle.loads(pickle.dumps(curve))
    first = curve.offset(0.25)
    second = restored.offset(0.25)
    assert np.array_equal(first.control_points, second.control_points)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.knots, second.knots)


def test_corrupted_spline_payload_fails_typed(curve):
    state = curve.__getstate__()
    segments = list(state["_segments"])
    bad = pickle.loads(pickle.dumps(segments[0]))
    object.__setattr__(bad, "w0", 0.0j)
    object.__setattr__(bad, "w1", 0.0j)
    object.__setattr__(bad, "a", 0.0j)
    object.__setattr__(bad, "b", 0.0j)
    object.__setattr__(bad, "A", 0.0)
    object.__setattr__(bad, "B", 0.0)
    object.__setattr__(bad, "C", 0.0)
    segments[0] = bad
    state["_segments"] = tuple(segments)
    blank = object.__new__(type(curve))
    with pytest.raises(Exception) as info:
        blank.__setstate__(state)
    assert "PHSpline" in " ".join(
        base.__name__ for base in type(info.value).__mro__
    )


# ---------------------------------------------------------------------------
# NURBS handle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def handle():
    return CubicPHSplineOpen(OPEN_POINTS).offset(0.35)


def test_handle_pickle_round_trip(handle):
    restored = pickle.loads(pickle.dumps(handle))
    assert restored.degree == handle.degree
    assert restored.closed == handle.closed
    assert np.array_equal(restored.knots, handle.knots)
    assert np.array_equal(restored.control_points, handle.control_points)
    assert np.array_equal(restored.weights, handle.weights)
    for u in np.linspace(0.0, 1.0, 23):
        assert np.array_equal(restored.point(float(u)), handle.point(float(u)))


def test_handle_deepcopy_and_copy(handle):
    for clone in (copy.deepcopy(handle), copy.copy(handle)):
        assert np.array_equal(clone.point(0.4), handle.point(0.4))
        assert not clone.weights.flags.writeable


def test_restored_handle_is_still_immutable_and_verified(handle):
    restored = pickle.loads(pickle.dumps(handle))
    with pytest.raises(AttributeError):
        restored._degree = 7
    assert not restored.knots.flags.writeable


def test_corrupted_handle_payload_fails_typed(handle):
    state = handle.__getstate__()
    weights = np.array(state["_weights"], copy=True)
    weights[3] = -1.0
    state["_weights"] = weights
    blank = object.__new__(type(handle))
    with pytest.raises(OffsetConstructionError):
        blank.__setstate__(state)


def test_phb_family_unaffected():
    spline = PHBSplineOpen([[0, 0], [1, 0.4], [2, -0.7], [3, 1.1]])
    restored = pickle.loads(pickle.dumps(spline))
    assert np.array_equal(restored.point(0.4), spline.point(0.4))
    snapshot = spline.snapshot()
    again = pickle.loads(pickle.dumps(snapshot))
    assert np.array_equal(again.point(0.4), snapshot.point(0.4))
