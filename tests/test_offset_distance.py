"""Offset NURBS distance queries: routine and demanding-but-ordinary cases.

First half of the acceptance suite for
``OffsetNURBS_Distance_Specification.md``: the API/compatibility matrix
over all four source families (spec 14.2), scalar validation (3.5),
endpoint identities (3.2/3.3), correct rounding against the independent
high-precision oracle (11.1 / 14.1), the worked cubic example (4.10),
forward/inverse consistency, the metamorphic turning identity (14.4),
serialization and source-edit isolation (6.1), and d == 0 / straight-span
metric equivalence (14.3).

The companion file ``test_offset_distance_extreme.py`` holds the
ill-conditioned and adversarial half of the suite.
"""

from __future__ import annotations

import math
import pickle

import mpmath as mp
import numpy as np
import pytest

from ph_spline import (
    ArcLengthOutOfRangeError,
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    OffsetConstructionError,
    ParameterOutOfRangeError,
    PHBSplineClosed,
    PHBSplineOpen,
)
from ph_spline.offset_metric import build_offset_metric

from offset_oracle import OffsetOracle, assert_correctly_rounded

OPEN_PTS = [[0.0, 0.0], [1.0, 0.4], [2.0, 1.3], [2.6, 2.4]]
CLOSED_PTS = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
B_OPEN_PTS = [[0.0, 0.0], [1.0, 0.8], [2.2, 0.9], [3.0, 0.2], [4.0, 0.0]]
B_CLOSED_PTS = [[1.0, 0.0], [0.3, 1.1], [-1.0, 0.4], [-0.6, -1.0], [0.5, -0.9]]

FAMILIES = [
    ("cubic-open", lambda: CubicPHSplineOpen(OPEN_PTS), 0.35),
    ("cubic-closed", lambda: CubicPHSplineClosed(CLOSED_PTS), 0.25),
    ("bspline-open", lambda: PHBSplineOpen(B_OPEN_PTS), -0.3),
    ("bspline-closed", lambda: PHBSplineClosed(B_CLOSED_PTS), 0.2),
]


@pytest.fixture(scope="module", params=FAMILIES, ids=[f[0] for f in FAMILIES])
def family_handle(request):
    name, make, d = request.param
    curve = make()
    return name, curve, d, curve.offset(d)


@pytest.fixture(scope="module")
def cubic_handle():
    curve = CubicPHSplineOpen(OPEN_PTS)
    return curve, curve.offset(0.35)


# ---------------------------------------------------------------------------
# API matrix and scalar validation (spec 14.2 / 3.5)
# ---------------------------------------------------------------------------


def test_all_families_expose_the_distance_api(family_handle):
    _, _, _, h = family_handle
    assert isinstance(h.length, float)
    assert isinstance(h.arc_length(0.5), float)
    assert isinstance(h.parameter_at_length(0.5 * h.length), float)
    p = h.point_at_length(0.5 * h.length)
    assert p.shape == (2,) and p.dtype == np.float64


def test_length_is_finite_positive_and_o1(family_handle):
    _, _, _, h = family_handle
    L = h.length
    assert math.isfinite(L) and L > 0.0
    assert h.length == L  # deterministic repeat


def test_scalar_type_rejection(family_handle):
    _, _, _, h = family_handle
    for bad in (True, np.True_, "0.5", [0.5], np.array([0.5]), 1 + 2j, None):
        with pytest.raises(TypeError):
            h.arc_length(bad)
        with pytest.raises(TypeError):
            h.parameter_at_length(bad)
        with pytest.raises(TypeError):
            h.point_at_length(bad)


def test_numpy_scalars_accepted(family_handle):
    _, _, _, h = family_handle
    assert h.arc_length(np.float64(0.25)) == h.arc_length(0.25)
    s = np.float32(0.5 * h.length)
    assert h.parameter_at_length(s) == h.parameter_at_length(float(s))


def test_u_domain_validation(family_handle):
    _, _, _, h = family_handle
    with pytest.raises(ParameterOutOfRangeError):
        h.arc_length(float("nan"))
    with pytest.raises(ParameterOutOfRangeError):
        h.arc_length(-0.001)
    with pytest.raises(ParameterOutOfRangeError):
        h.arc_length(1.001)
    # four-eps endpoint slack clamps
    assert h.arc_length(-3.0e-16) == 0.0
    assert h.arc_length(1.0 + 3.0e-16) == h.length


def test_s_domain_validation(family_handle):
    _, _, _, h = family_handle
    L = h.length
    with pytest.raises(ArcLengthOutOfRangeError):
        h.parameter_at_length(float("nan"))
    with pytest.raises(ArcLengthOutOfRangeError):
        h.parameter_at_length(float("inf"))
    with pytest.raises(ArcLengthOutOfRangeError):
        h.parameter_at_length(-float("inf"))
    with pytest.raises(ArcLengthOutOfRangeError):
        h.parameter_at_length(-10.0 * math.ulp(L))
    with pytest.raises(ArcLengthOutOfRangeError):
        h.parameter_at_length(L + 10.0 * math.ulp(L))
    # four-ulp-of-L clamp windows (spec 3.5)
    assert h.parameter_at_length(-3.0 * math.ulp(L)) == 0.0
    assert h.parameter_at_length(L + 3.0 * math.ulp(L)) == 1.0


def test_endpoint_identities_exact(family_handle):
    _, _, _, h = family_handle
    assert h.arc_length(0.0) == 0.0
    assert h.arc_length(1.0) == h.length
    assert h.parameter_at_length(0.0) == 0.0
    assert h.parameter_at_length(h.length) == 1.0
    assert np.array_equal(h.point_at_length(0.0), h.point(0.0))
    assert np.array_equal(h.point_at_length(h.length), h.point(1.0))


def test_closed_topology_does_not_wrap(family_handle):
    name, _, _, h = family_handle
    if not h.closed:
        pytest.skip("open handle")
    assert h.parameter_at_length(0.0) == 0.0
    assert h.parameter_at_length(h.length) == 1.0
    seam = np.hypot(*(h.point(0.0) - h.point(1.0)))
    assert seam <= 1e-9 * max(1.0, h.length)


# ---------------------------------------------------------------------------
# Correct rounding against the independent oracle (spec 11.1 / 14.1)
# ---------------------------------------------------------------------------


def test_length_and_arc_length_are_correctly_rounded(family_handle):
    name, _, d, h = family_handle
    oracle = OffsetOracle(h, prec=300)
    assert_correctly_rounded(h.length, oracle.length(), f"{name} length")
    rng = np.random.default_rng(20260813)
    for u in list(rng.uniform(0.0, 1.0, 25)) + [0.125, 0.5, 0.875]:
        u = float(u)
        assert_correctly_rounded(
            h.arc_length(u), oracle.arc_length(u), f"{name} arc({u!r})"
        )


def test_arc_length_public_nondecrease(family_handle):
    _, _, _, h = family_handle
    us = np.linspace(0.0, 1.0, 801)
    values = [h.arc_length(float(u)) for u in us]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert values[0] == 0.0 and values[-1] == h.length
    assert all(0.0 <= v <= h.length for v in values)


def test_inverse_round_trip_regular_points(family_handle):
    _, _, _, h = family_handle
    L = h.length
    for frac in (2.0**-52, 1e-12, 1e-9, 0.01, 0.25, 0.5, 0.75, 0.99,
                 1 - 1e-9, 1 - 2.0**-52):
        s = frac * L
        u = h.parameter_at_length(s)
        assert 0.0 <= u <= 1.0
        a = h.arc_length(u)
        # residual gate: certified evaluation error plus target quantization
        assert abs(a - s) <= 1e-12 * L + 8.0 * math.ulp(L)


def test_point_at_length_equals_two_call_expression(family_handle):
    _, _, _, h = family_handle
    L = h.length
    for frac in (0.1, 0.37, 0.5, 0.9):
        s = frac * L
        u = h.parameter_at_length(s)
        assert np.array_equal(h.point_at_length(s), h.point(u))


def test_prefix_boundaries_return_stored_values(family_handle):
    _, _, _, h = family_handle
    metric = h._metric
    for j, b in enumerate(metric._bounds):
        assert h.arc_length(b) == metric._prefix_float[j]


def test_inverse_at_prefix_neighborhoods(family_handle):
    _, _, _, h = family_handle
    metric = h._metric
    L = h.length
    for pf in metric._prefix_float[1:-1]:
        for s in (pf, math.nextafter(pf, 0.0), math.nextafter(pf, math.inf)):
            if 0.0 <= s <= L:
                u = h.parameter_at_length(s)
                assert 0.0 <= u <= 1.0
                assert abs(h.arc_length(u) - s) <= 1e-12 * L + 8 * math.ulp(L)


# ---------------------------------------------------------------------------
# Worked cubic example: deterministic unit oracle (spec 4.10)
# ---------------------------------------------------------------------------


def _worked_example_metric():
    return build_offset_metric(
        span_preimages=[[1 + 0j, 1 + 1j]],
        span_widths=[1.0],
        breakpoints=[0.0, 1.0],
        distance=1.0,
        scale=1.0,
        closed=False,
    )


def test_worked_example_cusp_and_cell_lengths():
    m = _worked_example_metric()
    c = 0.6435942529055827  # sqrt(sqrt(2) - 1)
    assert len(m._cells) == 2
    assert m._bounds[1] == c
    assert m._cells[0].eta == -1 and m._cells[1].eta == 1
    # Correctly rounded reference values: the exact spec quantities are
    # l0 = 2 atan(c) - c - c^3/3 = 0.41126166475721384972...
    # L1 = 0.58506033605286441355...  (the spec prints truncated decimals).
    assert m._prefix_float[1] == 0.4112616647572139
    assert m.length == 0.5850603360528644


def test_worked_example_distance_formula():
    m = _worked_example_metric()
    c = 0.6435942529055827
    for t in (0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 0.999):
        got = m.arc_length(t)
        with mp.workprec(250):
            cx = mp.sqrt(mp.sqrt(2) - 1)
            def D(a, t_):
                return (t_ - a) + (t_**3 - a**3) / 3 - 2 * mp.atan2(
                    t_ - a, 1 + a * t_
                )
            if t <= float(cx):
                exact = -D(mp.mpf(0), mp.mpf(t))
            else:
                exact = -D(mp.mpf(0), cx) + D(cx, mp.mpf(t))
        assert got == float(exact), (t, got, float(exact))


def test_worked_example_public_cusp_record():
    from ph_spline import OffsetCusp

    m = _worked_example_metric()
    assert m.cusps == (OffsetCusp(0.6435942529055827, 1),)


def test_cusp_free_offsets_report_no_cusps(family_handle):
    _, curve, d, h = family_handle
    assert h.cusps == ()  # the fixture offsets are all cusp-free
    assert curve.offset(0.0).cusps == ()


def test_cusps_are_zero_speed_points_and_prefix_anchors():
    loop = CubicPHSplineClosed(CLOSED_PTS)
    h = loop.offset(1.0)
    cusps = h.cusps
    assert len(cusps) == 8 and all(c.multiplicity == 1 for c in cusps)
    assert [c.parameter for c in cusps] == sorted(c.parameter for c in cusps)
    du = 1e-7
    for c in cusps:
        u = c.parameter
        # arc_length is stationary at the cusp: the local finite-difference
        # speed collapses relative to the mean speed of the handle.
        v_local = (h.arc_length(u + du) - h.arc_length(u - du)) / (2 * du)
        assert v_local <= 1e-4 * h.length
        # every interior cusp is a stored metric prefix anchor
        assert h.arc_length(u) in h._metric._prefix_float
    # the record survives serialization (rebuilt and re-verified)
    clone = pickle.loads(pickle.dumps(h))
    assert clone.cusps == cusps


def test_worked_example_inverse_both_sides_of_cusp():
    m = _worked_example_metric()
    l0 = m._prefix_float[1]
    for s in (0.5 * l0, l0 - 1e-9, l0, l0 + 1e-9, 0.9 * m.length):
        u = m.parameter_at_length(s)
        assert abs(m.arc_length(u) - s) <= 1e-13


# ---------------------------------------------------------------------------
# Zero-offset and straight-span metric equivalence (spec 14.3 / 4.6.6)
# ---------------------------------------------------------------------------


def test_zero_offset_metric_matches_source(family_handle):
    _, curve, _, _ = family_handle
    h = curve.offset(0.0)
    L_src = curve.arc_length(1.0)
    assert abs(h.length - L_src) <= 4.0 * math.ulp(L_src)
    for u in (0.2, 0.5, 0.8):
        assert abs(h.arc_length(u) - curve.arc_length(u)) <= 8.0 * math.ulp(L_src)


def test_straight_spans_offset_metric_equals_source_metric():
    line = CubicPHSplineOpen([[0.0, 0.0], [1.0, 0.0], [2.5, 0.0], [4.0, 0.0]])
    L_src = line.arc_length(1.0)
    for d in (0.7, -3.0, 1e6):
        h = line.offset(d)
        assert abs(h.length - L_src) <= 4.0 * math.ulp(max(L_src, abs(d)))
        for u in (0.1, 0.5, 0.9):
            assert abs(h.arc_length(u) - line.arc_length(u)) <= 8.0 * math.ulp(
                max(L_src, abs(d))
            )
        u = h.parameter_at_length(0.5 * h.length)
        assert abs(h.arc_length(u) - 0.5 * h.length) <= 4.0 * math.ulp(L_src)


def test_positive_and_negative_cusp_free_offsets(family_handle):
    name, curve, d, _ = family_handle
    oracle_checked = 0
    for dd in (abs(d) / 2, -abs(d) / 2):
        h = curve.offset(dd)
        o = OffsetOracle(h, prec=300)
        assert_correctly_rounded(h.length, o.length(), f"{name} d={dd}")
        oracle_checked += 1
    assert oracle_checked == 2


# ---------------------------------------------------------------------------
# Metamorphic identities (spec 14.4)
# ---------------------------------------------------------------------------


def test_closed_turning_identity_l_d():
    """Cusp-free closed offset: L_d = L_0 -+ 2 pi d (turning number one)."""
    loop = CubicPHSplineClosed(CLOSED_PTS)
    flat = loop.offset(0.0)
    L0 = flat.length
    for d in (0.05, 0.15, -0.05, -0.2):
        Ld = loop.offset(d).length
        expected = L0 - 2.0 * math.pi * d
        alt = L0 + 2.0 * math.pi * d
        best = min(abs(Ld - expected), abs(Ld - alt))
        assert best <= 1e-12 * max(L0, 1.0), (d, Ld, expected, alt)


def test_sign_reversing_offset_lobe_identity():
    """Reversed lobes add twice their magnitude over the signed primitive."""
    loop = CubicPHSplineClosed(CLOSED_PTS)
    L0 = loop.offset(0.0).length
    d = 1.0  # inside the curvature range: reversal lobes exist
    h = loop.offset(d)
    metric = h._metric
    signed = sum(
        c.eta * c.length_float for c in metric._cells
    )
    unsigned = sum(c.length_float for c in metric._cells)
    neg = sum(c.length_float for c in metric._cells if c.eta < 0)
    assert abs(unsigned - h.length) <= 1e-12 * max(1.0, h.length)
    assert abs((unsigned - signed) - 2.0 * neg) <= 1e-12 * max(1.0, h.length)
    # signed primitive equals L0 - 2 pi d for the turning-number-one loop
    assert abs(signed - (L0 - 2.0 * math.pi * d)) <= 1e-10 * max(1.0, L0)


def test_derivative_identity_finite_difference(cubic_handle):
    """A'(u) = |1 - d kappa| * |z'(u)| away from joins and cusps."""
    curve, h = cubic_handle
    d = 0.35
    du = 1e-7
    for u in (0.15, 0.45, 0.85):
        a_prime = (h.arc_length(u + du) - h.arc_length(u - du)) / (2 * du)
        kappa = curve.signed_curvature(u)
        # source speed |z'(u)| via finite difference of source arc length
        v_src = (curve.arc_length(u + du) - curve.arc_length(u - du)) / (2 * du)
        expected = abs(1.0 - d * kappa) * v_src
        assert abs(a_prime - expected) <= 1e-5 * max(1.0, expected)


# ---------------------------------------------------------------------------
# Immutability, serialization, and source-edit isolation (spec 6.1)
# ---------------------------------------------------------------------------


def test_pickle_round_trip_rebuilds_and_verifies(family_handle):
    _, _, _, h = family_handle
    clone = pickle.loads(pickle.dumps(h))
    assert clone.length == h.length
    for u in (0.2, 0.5, 0.9):
        assert clone.arc_length(u) == h.arc_length(u)
    s = 0.4 * h.length
    assert clone.parameter_at_length(s) == h.parameter_at_length(s)


def test_copy_shares_verified_state(family_handle):
    import copy

    _, _, _, h = family_handle
    dup = copy.deepcopy(h)
    assert dup.length == h.length
    assert dup.arc_length(0.3) == h.arc_length(0.3)


def test_source_edit_does_not_change_distance_results():
    curve = PHBSplineOpen(B_OPEN_PTS)
    h = curve.offset(-0.3)
    before = (h.length, h.arc_length(0.42), h.parameter_at_length(0.3 * h.length))
    handle_pt = curve.point_handle(2)
    curve.move_point(handle_pt, [2.2, 1.4])
    after = (h.length, h.arc_length(0.42), h.parameter_at_length(0.3 * h.length))
    assert before == after


def test_existing_geometry_unchanged_by_metric(family_handle):
    """The metric addition must not perturb the published NURBS arrays."""
    _, curve, d, h = family_handle
    # point() must still satisfy the exact offset identity against the source.
    for u in (0.15, 0.5, 0.85):
        base = curve.point(u)
        normal = curve.normal(u, side="left")
        expected = base + d * normal
        assert np.hypot(*(h.point(u) - expected)) <= 1e-9 * max(
            1.0, float(np.max(np.abs(expected)))
        )
    assert h.num_control_points == h.num_spans * h.degree + 1


def test_handle_without_certificate_rejects_distance_queries():
    """Internal handles built without PH metric data expose no distances."""
    from ph_spline.nurbs import build_offset_handle
    from fractions import Fraction

    controls = np.array(
        [[0.0, 0.0], [1.0, 0.5], [2.0, 0.5], [3.0, 0.0]], dtype=np.float64
    )
    speed = np.array([1.0, 0.9, 1.0])
    hodograph = 3.0 * np.diff(controls, axis=0)

    def oracle(u):
        t = Fraction(u)

        def bern(vals):
            work = [Fraction(float(v)) for v in vals]
            s = 1 - t
            while len(work) > 1:
                work = [s * a + t * b for a, b in zip(work[:-1], work[1:])]
            return work[0]

        x = bern(controls[:, 0])
        y = bern(controls[:, 1])
        vx = bern(3.0 * np.diff(controls[:, 0]))
        vy = bern(3.0 * np.diff(controls[:, 1]))
        sg = bern(speed)
        dd = Fraction(0.25)
        return (
            (float(x + dd * (-vy) / sg), float(y + dd * vx / sg)),
            (0.0, 0.0),
        )

    h = build_offset_handle(
        span_controls=[controls],
        span_speeds=[speed],
        span_hodographs=[hodograph],
        hodograph_tolerance=1e-8,
        breakpoints=np.array([0.0, 1.0]),
        distance=0.25,
        distance_normalized=0.25,
        origin=(0.0, 0.0),
        scale=1.0,
        closed=False,
        join_tolerance=1e-12,
        oracle=oracle,
    )
    with pytest.raises(OffsetConstructionError):
        _ = h.length
    with pytest.raises(OffsetConstructionError):
        h.arc_length(0.5)
    with pytest.raises(OffsetConstructionError):
        h.parameter_at_length(0.1)


def test_construction_and_queries_are_deterministic(family_handle):
    _, curve, d, h = family_handle
    h2 = curve.offset(d)
    assert h2.length == h.length
    assert h2._metric._prefix_float == h._metric._prefix_float
    for u in (0.11, 0.53, 0.97):
        assert h2.arc_length(u) == h.arc_length(u)
        assert h.arc_length(u) == h.arc_length(u)
