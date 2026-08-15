"""Frame and curvature queries on the exact offset NURBS (spec 11.7.7 / 15.6.9).

Every query is verified against the closed-form parallel-curve identities
through the public source geometry API:

    T_d = sign(1 - d kappa) T,        N_d = sign(1 - d kappa) N,
    kappa_d = kappa / |1 - d kappa|,  K_d = kappa / (1 - d kappa) N_L,

against the shared-curvature-center identity of parallel curves, and
against an independent central finite-difference oracle applied to the
published handle itself.  Cusp behavior, validation, immutability,
serialization, and metric-less handles are covered separately.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from ph_spline import (
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    NumericalPrecisionError,
    OffsetConstructionError,
    ParameterOutOfRangeError,
    PHBSplineClosed,
    PHBSplineOpen,
    UndefinedPrincipalNormalError,
    UndefinedTangentError,
)

S_CURVE = [
    [0.0, 0.0], [1.0, 0.8], [2.0, 1.0], [3.0, 0.2], [4.0, 0.0], [5.0, 0.8],
]
WAVY_RING = [
    [
        (2.0 + 0.3 * math.cos(5.0 * t)) * math.cos(t),
        (2.0 + 0.3 * math.cos(5.0 * t)) * math.sin(t),
    ]
    for t in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
]
BSPLINE_OPEN = [[0.0, 0.0], [1.0, 0.4], [2.0, -0.7], [3.0, 1.1], [2.2, 2.0]]
PENTAGON = [
    [1.0, 0.0], [0.31, 0.95], [-0.81, 0.59], [-0.81, -0.59], [0.31, -0.95],
]


def _build(name: str):
    if name == "cubic_open":
        return CubicPHSplineOpen(S_CURVE)
    if name == "cubic_closed":
        return CubicPHSplineClosed(WAVY_RING)
    if name == "bspline_open_g2":
        return PHBSplineOpen(BSPLINE_OPEN)
    if name == "bspline_open_g4":
        return PHBSplineOpen(BSPLINE_OPEN, g_order=4)
    if name == "bspline_closed":
        return PHBSplineClosed(PENTAGON)
    raise AssertionError(name)


CASE_NAMES = [
    "cubic_open",
    "cubic_closed",
    "bspline_open_g2",
    "bspline_open_g4",
    "bspline_closed",
]

#: Signed cusp-free distances per curve: well inside ``(-rho_R, rho_L)``.
DISTANCE_KINDS = ("left", "right", "zero")


@pytest.fixture(params=CASE_NAMES, ids=CASE_NAMES, scope="module")
def curve(request):
    return _build(request.param)


def _distance(curve, kind: str) -> float:
    rho_left, rho_right = curve.min_curvature_radii
    if kind == "left":
        return 0.4 * min(rho_left, 2.0)
    if kind == "right":
        return -0.4 * min(rho_right, 2.0)
    return 0.0


def _parameters(handle, seed: int = 20260815, count: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        ([0.0, 1.0], np.unique(handle.knots), rng.uniform(0.0, 1.0, count))
    )


def _scale(curve) -> float:
    points = curve.points if hasattr(curve, "points") else curve._points
    return max(1.0, float(np.max(np.abs(points))))


# ---------------------------------------------------------------------------
# Closed-form parallel-curve identities against the source geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", DISTANCE_KINDS)
def test_tangent_matches_reversal_identity(curve, kind):
    d = _distance(curve, kind)
    handle = curve.offset(d)
    tol = 1e-10 * _scale(curve)
    for u in _parameters(handle):
        u = float(u)
        flip = 1.0 if 1.0 - d * curve.signed_curvature(u) >= 0.0 else -1.0
        produced = handle.tangent(u)
        expected = flip * curve.tangent(u)
        assert float(np.max(np.abs(produced - expected))) <= tol
        assert abs(float(np.hypot(*produced)) - 1.0) <= 1e-14


@pytest.mark.parametrize("kind", DISTANCE_KINDS)
def test_normal_sides_match_reversal_identity(curve, kind):
    d = _distance(curve, kind)
    handle = curve.offset(d)
    tol = 1e-10 * _scale(curve)
    for u in _parameters(handle):
        u = float(u)
        flip = 1.0 if 1.0 - d * curve.signed_curvature(u) >= 0.0 else -1.0
        left = handle.normal(u)
        right = handle.normal(u, "right")
        tangent = handle.tangent(u)
        # Exact algebraic relations between the three unit vectors.
        assert np.array_equal(right, -left)
        assert np.array_equal(left, np.array([-tangent[1], tangent[0]]))
        assert float(np.max(np.abs(left - flip * curve.normal(u)))) <= tol
        assert (
            float(np.max(np.abs(right - flip * curve.normal(u, "right"))))
            <= tol
        )


@pytest.mark.parametrize("kind", DISTANCE_KINDS)
def test_signed_curvature_matches_identity(curve, kind):
    d = _distance(curve, kind)
    handle = curve.offset(d)
    for u in _parameters(handle):
        u = float(u)
        kappa = curve.signed_curvature(u)
        expected = kappa / abs(1.0 - d * kappa)
        produced = handle.signed_curvature(u)
        assert abs(produced - expected) <= 1e-9 * max(1.0, abs(expected))


@pytest.mark.parametrize("kind", DISTANCE_KINDS)
def test_curvature_vector_matches_identity(curve, kind):
    d = _distance(curve, kind)
    handle = curve.offset(d)
    for u in _parameters(handle):
        u = float(u)
        kappa = curve.signed_curvature(u)
        # No absolute value: tangent and normal reversals cancel exactly.
        expected = (kappa / (1.0 - d * kappa)) * curve.normal(u)
        produced = handle.curvature_vector(u)
        magnitude = max(1.0, float(np.max(np.abs(expected))))
        assert float(np.max(np.abs(produced - expected))) <= 1e-8 * magnitude


@pytest.mark.parametrize("kind", ("left", "right"))
def test_principal_normal_sign_convention_and_shared_center(curve, kind):
    d = _distance(curve, kind)
    handle = curve.offset(d)
    tol = 1e-6 * _scale(curve)
    checked = 0
    for u in _parameters(handle):
        u = float(u)
        kappa = curve.signed_curvature(u)
        if abs(kappa) <= 0.05:
            continue
        produced = handle.principal_normal(u)
        kappa_d = handle.signed_curvature(u)
        sign = 1.0 if kappa_d > 0.0 else -1.0
        assert np.array_equal(produced, sign * handle.normal(u))
        # A parallel curve shares every curvature center with its source.
        center_source = curve.point(u) + curve.principal_normal(u) / abs(kappa)
        center_offset = handle.point(u) + produced / abs(kappa_d)
        assert float(np.max(np.abs(center_offset - center_source))) <= tol
        checked += 1
    assert checked > 10


def test_zero_distance_reproduces_source_frames(curve):
    handle = curve.offset(0.0)
    tol = 1e-10 * _scale(curve)
    for u in _parameters(handle, count=20):
        u = float(u)
        assert float(np.max(np.abs(handle.tangent(u) - curve.tangent(u)))) <= tol
        assert float(np.max(np.abs(handle.normal(u) - curve.normal(u)))) <= tol
        kappa = curve.signed_curvature(u)
        assert abs(handle.signed_curvature(u) - kappa) <= 1e-9 * max(
            1.0, abs(kappa)
        )
        expected = kappa * curve.normal(u)
        assert float(
            np.max(np.abs(handle.curvature_vector(u) - expected))
        ) <= 1e-8 * max(1.0, float(np.max(np.abs(expected))))


# ---------------------------------------------------------------------------
# Independent finite-difference oracle on the published handle itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, kind", [("cubic_open", "left"), ("bspline_closed", "right")]
)
def test_finite_difference_oracle(name, kind):
    curve = _build(name)
    d = _distance(curve, kind)
    handle = curve.offset(d)
    # Step balances central-difference truncation against the binary64
    # roundoff floor of the second difference (~eps / step**2).  Samples
    # keep clear of the knots: the parametric second derivative may jump
    # there, which invalidates a straddling central difference.
    step = 1e-4
    knots = np.unique(handle.knots)
    for u in np.linspace(0.05, 0.95, 19):
        u = float(u)
        if float(np.min(np.abs(knots - u))) <= 100.0 * step:
            continue
        ahead = handle.point(u + step)
        behind = handle.point(u - step)
        first = (ahead - behind) / (2.0 * step)
        second = (ahead - 2.0 * handle.point(u) + behind) / step**2
        speed = float(np.hypot(*first))
        tangent_fd = first / speed
        kappa_fd = float(first[0] * second[1] - first[1] * second[0]) / speed**3
        assert float(np.max(np.abs(handle.tangent(u) - tangent_fd))) <= 1e-6
        kappa = handle.signed_curvature(u)
        assert abs(kappa - kappa_fd) <= 1e-4 * max(1.0, abs(kappa))


# ---------------------------------------------------------------------------
# Straight sources: exact translation, zero curvature
# ---------------------------------------------------------------------------


def test_straight_offset_frames():
    line = CubicPHSplineOpen([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    handle = line.offset(2.5)
    direction = np.array([0.6, 0.8])
    for u in (0.0, 0.25, 0.5, 0.9, 1.0):
        assert float(np.max(np.abs(handle.tangent(u) - direction))) <= 1e-14
        assert float(
            np.max(np.abs(handle.normal(u) - np.array([-0.8, 0.6])))
        ) <= 1e-14
        assert abs(handle.signed_curvature(u)) <= 1e-12
        assert float(np.max(np.abs(handle.curvature_vector(u)))) <= 1e-12
        with pytest.raises(UndefinedPrincipalNormalError) as info:
            handle.principal_normal(u)
        assert info.value.quantity == "curvature numerator"


# ---------------------------------------------------------------------------
# Cusps: guard, direction reversal, curvature divergence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cusped():
    curve = CubicPHSplineOpen(S_CURVE)
    rho_left, _ = curve.min_curvature_radii
    distance = 1.4 * rho_left
    return curve, distance, curve.offset(distance)


def test_cusped_offset_has_certified_cusps(cusped):
    _, _, handle = cusped
    assert len(handle.cusps) >= 1


def test_all_frame_queries_raise_at_certified_cusps(cusped):
    _, _, handle = cusped
    for cusp in handle.cusps:
        u = cusp.parameter
        for query in (
            handle.tangent,
            handle.normal,
            handle.principal_normal,
            handle.signed_curvature,
            handle.curvature_vector,
        ):
            with pytest.raises(UndefinedTangentError) as info:
                query(u)
            assert info.value.quantity == "offset speed"


def test_tangent_reverses_across_odd_cusp(cusped):
    curve, distance, handle = cusped
    cusp = handle.cusps[0]
    assert cusp.multiplicity % 2 == 1
    delta = 1e-5
    before = handle.tangent(cusp.parameter - delta)
    after = handle.tangent(cusp.parameter + delta)
    assert float(before @ after) < -0.99
    # The reversed side opposes the source tangent, the other side agrees.
    for u, expected_sign in (
        (cusp.parameter - delta, 1.0),
        (cusp.parameter + delta, -1.0),
    ):
        flip = 1.0 if 1.0 - distance * curve.signed_curvature(u) >= 0.0 else -1.0
        agreement = float(handle.tangent(u) @ curve.tangent(u))
        assert flip == expected_sign
        assert agreement * expected_sign > 0.99


def test_curvature_diverges_with_source_sign_near_cusp(cusped):
    curve, _, handle = cusped
    cusp = handle.cusps[0]
    for u in (cusp.parameter - 1e-5, cusp.parameter + 1e-5):
        kappa_source = curve.signed_curvature(u)
        kappa_offset = handle.signed_curvature(u)
        assert kappa_offset * kappa_source > 0.0
        assert abs(kappa_offset) > 100.0 * abs(kappa_source)


def test_fully_reversed_offset_opposes_source_everywhere():
    # A near-circular arc offset beyond its center: 1 - d * kappa < 0 on
    # the whole domain, so the traversal reverses without any cusp.
    arc = CubicPHSplineOpen(
        [[math.cos(a), math.sin(a)] for a in np.linspace(0.1, 1.3, 9)]
    )
    handle = arc.offset(1.3)
    assert handle.cusps == ()
    for u in np.linspace(0.0, 1.0, 21):
        u = float(u)
        assert float(handle.tangent(u) @ arc.tangent(u)) < -0.999999
        assert handle.signed_curvature(u) * arc.signed_curvature(u) > 0.0


# ---------------------------------------------------------------------------
# Validation: the scalar rules of point(u) apply to every frame query
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame_handle():
    return CubicPHSplineOpen(S_CURVE).offset(0.2)


def _queries(handle):
    return (
        handle.tangent,
        handle.normal,
        handle.principal_normal,
        handle.signed_curvature,
        handle.curvature_vector,
    )


def test_frame_queries_reject_non_scalar_arguments(frame_handle):
    for query in _queries(frame_handle):
        for bad in (True, np.True_, "0.5", [0.5], np.array([0.5, 0.6]), None):
            with pytest.raises(TypeError):
                query(bad)


def test_frame_queries_reject_nan_and_out_of_domain(frame_handle):
    for query in _queries(frame_handle):
        with pytest.raises(ParameterOutOfRangeError):
            query(float("nan"))
        with pytest.raises(ParameterOutOfRangeError):
            query(-0.001)
        with pytest.raises(ParameterOutOfRangeError):
            query(1.001)


def test_frame_queries_clamp_few_ulp_violations(frame_handle):
    below = -1.0e-16
    above = 1.0 + 2.0e-16
    assert np.array_equal(
        frame_handle.tangent(below), frame_handle.tangent(0.0)
    )
    assert np.array_equal(
        frame_handle.tangent(above), frame_handle.tangent(1.0)
    )
    assert frame_handle.signed_curvature(below) == (
        frame_handle.signed_curvature(0.0)
    )


def test_frame_queries_accept_numpy_scalars(frame_handle):
    assert np.array_equal(
        frame_handle.tangent(np.float64(0.25)), frame_handle.tangent(0.25)
    )
    assert frame_handle.signed_curvature(
        np.float64(0.25)
    ) == frame_handle.signed_curvature(0.25)


def test_normal_side_argument_is_validated(frame_handle):
    for bad in ("Left", "RIGHT", "", "l", None, 1):
        with pytest.raises(ValueError):
            frame_handle.normal(0.5, bad)


def test_frame_results_are_fresh_arrays(frame_handle):
    first = frame_handle.tangent(0.4)
    reference = np.array(first, copy=True)
    first += 1000.0
    assert np.array_equal(frame_handle.tangent(0.4), reference)
    vector = frame_handle.curvature_vector(0.4)
    vector_reference = np.array(vector, copy=True)
    vector *= -3.0
    assert np.array_equal(frame_handle.curvature_vector(0.4), vector_reference)


def test_signed_curvature_returns_python_float(frame_handle):
    assert type(frame_handle.signed_curvature(0.3)) is float


# ---------------------------------------------------------------------------
# Determinism, serialization, and closed-seam consistency
# ---------------------------------------------------------------------------


def test_repeated_queries_are_deterministic(frame_handle):
    for u in (0.0, 0.31, 0.77, 1.0):
        assert np.array_equal(
            frame_handle.tangent(u), frame_handle.tangent(u)
        )
        assert frame_handle.signed_curvature(u) == (
            frame_handle.signed_curvature(u)
        )


def test_serialization_preserves_frame_values(curve):
    handle = curve.offset(_distance(curve, "left"))
    clone = pickle.loads(pickle.dumps(handle))
    for u in (0.0, 0.2, 0.5, 0.83, 1.0):
        assert np.array_equal(clone.tangent(u), handle.tangent(u))
        assert np.array_equal(clone.normal(u), handle.normal(u))
        assert clone.signed_curvature(u) == handle.signed_curvature(u)
        assert np.array_equal(
            clone.curvature_vector(u), handle.curvature_vector(u)
        )


@pytest.mark.parametrize("name", ("cubic_closed", "bspline_closed"))
@pytest.mark.parametrize("kind", ("left", "right"))
def test_closed_seam_frames_agree(name, kind):
    curve = _build(name)
    handle = curve.offset(_distance(curve, kind))
    assert handle.closed is True
    tol = 1e-9 * _scale(curve)
    assert float(
        np.max(np.abs(handle.tangent(0.0) - handle.tangent(1.0)))
    ) <= tol
    kappa0 = handle.signed_curvature(0.0)
    kappa1 = handle.signed_curvature(1.0)
    assert abs(kappa0 - kappa1) <= 1e-8 * max(1.0, abs(kappa0))


# ---------------------------------------------------------------------------
# Frame queries are geometry-only: no metric certificate required
# ---------------------------------------------------------------------------


def test_metricless_handle_supports_frames_but_not_distance():
    from test_offset_handle import (
        _synthetic_oracle,
        _synthetic_rational_source,
    )
    from ph_spline.nurbs import build_offset_handle

    controls, speed, hodograph = _synthetic_rational_source()
    distance = 0.4
    oracle = _synthetic_oracle(controls, speed, distance)
    handle = build_offset_handle(
        span_controls=[controls],
        span_speeds=[speed],
        span_hodographs=[hodograph],
        hodograph_tolerance=1e-8,
        breakpoints=np.array([0.0, 1.0]),
        distance=distance,
        distance_normalized=distance,
        origin=(0.0, 0.0),
        scale=1.0,
        closed=False,
        join_tolerance=1e-12,
        oracle=lambda u: (oracle(u)[0], (0.0, 0.0)),
    )
    with pytest.raises(OffsetConstructionError):
        handle.length
    for u in (0.1, 0.5, 0.9):
        tangent = handle.tangent(u)
        assert abs(float(np.hypot(*tangent)) - 1.0) <= 1e-14
        assert math.isfinite(handle.signed_curvature(u))


# ---------------------------------------------------------------------------
# Nonfinite protection
# ---------------------------------------------------------------------------


def test_nonfinite_curvature_raises_typed_error(frame_handle):
    # Guard coverage: the public API never returns a nonfinite curvature.
    for u in np.linspace(0.0, 1.0, 33):
        value = frame_handle.signed_curvature(float(u))
        assert math.isfinite(value)
        assert isinstance(value, float)
    assert issubclass(NumericalPrecisionError, RuntimeError)
