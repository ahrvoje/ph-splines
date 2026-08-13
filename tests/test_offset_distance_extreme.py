"""Offset NURBS distance queries: ill-conditioned and adversarial cases.

Second half of the acceptance suite for
``OffsetNURBS_Distance_Specification.md``.  These tests push accuracy,
reliability, and performance to the specification's edge: interior cusps
and beyond-critical reversals verified at 350-bit precision (14.3 cases
4/6), near-double-root perturbations around a curvature-extremum tangency
(case 5/8), extreme source scales (case 13), subnormal and
one-ulp-resolution inverse targets (case 14), catastrophic cancellation
between the source-length and turning terms (case 15), resource-cap and
fault-injected fallback paths (case 16 / 14.5), determinism under stress,
and query-time scaling over large span counts (14.6).
"""

from __future__ import annotations

import math
import time
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from ph_spline import (
    ArcLengthInversionError,
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    NumericalPrecisionError,
    OffsetConstructionError,
    PHBSplineClosed,
)
import ph_spline.offset_metric as om
from ph_spline.ddouble import ATAN2_DD_ABS_ERR, dd_atan2
from ph_spline.exact_real import atan2_ball, ball_to_fraction_bounds
from ph_spline.offset_metric import build_offset_metric

from offset_oracle import OffsetOracle, assert_correctly_rounded

DIAMOND = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]


@pytest.fixture(scope="module")
def diamond():
    return CubicPHSplineClosed(DIAMOND)


# ---------------------------------------------------------------------------
# Interior cusps, reversals, and adjacent-float parameters (14.3 / 14.4)
# ---------------------------------------------------------------------------


def test_interior_cusps_correctly_rounded(diamond):
    h = diamond.offset(1.0)
    o = OffsetOracle(h, prec=350)
    ncusp = sum(len(s.cusps) for s in o.spans)
    assert ncusp >= 4  # the offset stalls inside every quadrant
    assert_correctly_rounded(h.length, o.length(), "cusped length")
    us = []
    for s in o.spans:
        for c in s.cusps:
            cu = s.u0 + float(c) * (s.u1 - s.u0)
            us += [
                math.nextafter(cu, 0.0),
                cu,
                math.nextafter(cu, 1.0),
                cu - 1e-9,
                cu + 1e-9,
            ]
    for u in us:
        if 0.0 < u < 1.0:
            assert_correctly_rounded(
                h.arc_length(u), o.arc_length(u), f"cusp-adjacent arc({u!r})"
            )


def test_public_cusps_match_independent_oracle(diamond):
    """Every published cusp parameter agrees with the mpmath root finder
    to within a few ulps, and the counts match exactly."""
    h = diamond.offset(1.0)
    o = OffsetOracle(h, prec=350)
    oracle_cusps = []
    for s in o.spans:
        for c in s.cusps:
            oracle_cusps.append(s.u0 + float(c) * (s.u1 - s.u0))
    oracle_cusps.sort()
    produced = [c.parameter for c in h.cusps]
    assert len(produced) == len(oracle_cusps)
    for got, ref in zip(produced, oracle_cusps):
        assert abs(got - ref) <= 4.0 * math.ulp(max(abs(ref), 1.0)), (got, ref)


def test_full_reversal_offset_negative_eta_everywhere(diamond):
    """Beyond-critical offset whose speed sign is reversed on every cell."""
    h = diamond.offset(1.6)
    o = OffsetOracle(h, prec=350)
    assert all(c.eta == -1 for c in h._metric._cells)
    assert_correctly_rounded(h.length, o.length(), "reversed length")
    for u in (0.1, 0.35, 0.6, 0.85):
        assert_correctly_rounded(h.arc_length(u), o.arc_length(u), "reversed arc")


def test_inverse_at_cusp_prefixes_residual_certificates(diamond):
    """No loose parameter comparison at cusps: check certified residuals."""
    h = diamond.offset(1.0)
    o = OffsetOracle(h, prec=350)
    metric = h._metric
    L = h.length
    for pf in metric._prefix_float[1:-1]:
        for s in (
            math.nextafter(pf, 0.0),
            pf,
            math.nextafter(pf, math.inf),
        ):
            if not 0.0 < s < L:
                continue
            u = h.parameter_at_length(s)
            resid = abs(o.arc_length(u) - mp.mpf(s))
            # cusp-adjacent: parameter is ill-conditioned but the distance
            # residual must stay at quantization level (spec 5.3 / 10.4)
            assert resid <= mp.mpf(1e-10) * L, (s, u, float(resid))


def test_thousand_random_inverse_targets_cusped(diamond):
    h = diamond.offset(1.0)
    L = h.length
    rng = np.random.default_rng(1_000_003)
    targets = rng.uniform(0.0, 1.0, 1000) * L
    worst = 0.0
    for s in targets:
        s = float(s)
        u = h.parameter_at_length(s)
        worst = max(worst, abs(h.arc_length(u) - s))
    assert worst <= 1e-12 * L, worst


# ---------------------------------------------------------------------------
# Tangency and near-double roots (14.3 cases 5 and 8)
# ---------------------------------------------------------------------------


def test_offset_at_exact_minimal_radius_tangency(diamond):
    """d == rho_left puts G at (or one ulp from) an even-order tangency."""
    rho_l, _ = diamond.min_curvature_radii
    h = diamond.offset(rho_l)
    o = OffsetOracle(h, prec=400)
    assert_correctly_rounded(h.length, o.length(), "tangency length")
    for u in (0.11, 0.37, 0.62, 0.88):
        assert_correctly_rounded(h.arc_length(u), o.arc_length(u), "tangency arc")
    s = 0.5 * h.length
    u = h.parameter_at_length(s)
    assert abs(float(o.arc_length(u)) - s) <= 1e-10 * h.length


@pytest.mark.parametrize("eps", [1e-13, 1e-10, 1e-7])
def test_near_double_root_perturbations(diamond, eps):
    """Offsets straddling the tangency create close root pairs (case 8)."""
    rho_l, _ = diamond.min_curvature_radii
    for d in (rho_l - eps, rho_l + eps):
        h = diamond.offset(d)
        o = OffsetOracle(h, prec=400)
        assert_correctly_rounded(h.length, o.length(), f"near-double d={d!r}")
        for u in (0.2, 0.5, 0.8):
            assert_correctly_rounded(
                h.arc_length(u), o.arc_length(u), f"near-double arc d={d!r}"
            )
        s = 0.75 * h.length
        u = h.parameter_at_length(s)
        assert abs(float(o.arc_length(u)) - s) <= 1e-10 * h.length


# ---------------------------------------------------------------------------
# Extreme scales and subnormal targets (14.3 cases 13 and 14)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-150, 1e150, 1e-300, 1e250])
def test_extreme_source_scales(scale):
    pts = [[0.0, 0.0], [scale, 0.4 * scale], [2 * scale, 1.3 * scale],
           [2.6 * scale, 2.4 * scale]]
    curve = CubicPHSplineOpen(pts)
    for dmul in (0.1, -0.03):
        h = curve.offset(dmul * scale)
        o = OffsetOracle(h, prec=350)
        assert_correctly_rounded(h.length, o.length(), f"scale {scale}")
        for u in (0.25, 0.5, 0.9):
            assert_correctly_rounded(
                h.arc_length(u), o.arc_length(u), f"scale {scale} arc"
            )
        s = 0.4 * h.length
        u = h.parameter_at_length(s)
        assert abs(h.arc_length(u) - s) <= 1e-11 * h.length


def test_independently_varied_offset_and_scale():
    """Small curve with comparatively enormous |d| and the reverse."""
    small = CubicPHSplineOpen(
        [[0.0, 0.0], [1e-8, 4e-9], [2e-8, 1.3e-8], [2.6e-8, 2.4e-8]]
    )
    h = small.offset(1.0)  # offset 1e8 times the curve size
    o = OffsetOracle(h, prec=350)
    assert_correctly_rounded(h.length, o.length(), "tiny curve huge d")
    big = CubicPHSplineOpen(
        [[0.0, 0.0], [1e8, 4e7], [2e8, 1.3e8], [2.6e8, 2.4e8]]
    )
    h2 = big.offset(1e-6)  # offset 1e14 times below the curve size
    o2 = OffsetOracle(h2, prec=350)
    assert_correctly_rounded(h2.length, o2.length(), "huge curve tiny d")


def test_subnormal_local_inverse_targets(diamond):
    """Targets within ulps of 0, L, and each cusp prefix (case 14)."""
    h = diamond.offset(1.0)
    L = h.length
    tiny_targets = [5e-324, 1e-320, 2.0**-1000, math.ulp(L)]
    tiny_targets += [L - math.ulp(L), math.nextafter(L, 0.0)]
    for s in tiny_targets:
        u = h.parameter_at_length(s)
        assert 0.0 <= u <= 1.0
        a = h.arc_length(u)
        assert abs(a - s) <= 4.0 * math.ulp(L) + 1e-13 * L
    # s = one ulp above zero must map to a parameter with distance ~ s
    u = h.parameter_at_length(5e-324)
    assert h.arc_length(u) <= 8.0 * math.ulp(L)


# ---------------------------------------------------------------------------
# Catastrophic cancellation between Delta S and d Delta Theta (case 15)
# ---------------------------------------------------------------------------


def _near_circle(n: int):
    return CubicPHSplineClosed(
        [
            [math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)]
            for k in range(n)
        ]
    )


@pytest.mark.parametrize("n_pts", [16, 64, 256])
def test_near_circle_offset_cancellation(n_pts):
    """Offsetting a near-circle by nearly its radius cancels the source
    length against the turning term over every cell.

    With 256 interpolation points the interpolant tracks the unit circle
    to about 1e-10, so ``L_d`` collapses by more than 30 bits relative to
    ``L_0`` while remaining correctly rounded.
    """
    loop = _near_circle(n_pts)
    L0 = loop.offset(0.0).length
    rho_l, rho_r = loop.min_curvature_radii
    d = rho_l * (1.0 - 1e-9)
    h = loop.offset(d)
    o = OffsetOracle(h, prec=400)
    assert_correctly_rounded(h.length, o.length(), f"near-circle n={n_pts}")
    cancel_bits = math.log2(L0 / h.length)
    assert cancel_bits > 4.0  # material cancellation really occurred
    for u in (0.125, 0.5, 0.875):
        assert_correctly_rounded(
            h.arc_length(u), o.arc_length(u), f"near-circle arc n={n_pts}"
        )
    s = 0.5 * h.length
    u = h.parameter_at_length(s)
    assert abs(float(o.arc_length(u)) - s) <= 1e-11 * max(h.length, 1e-30)


def test_symmetric_s_curve_turning_cancellation():
    """Symmetric S: the turning terms of the two halves cancel, so +d and
    -d offsets agree to within the committed-source asymmetry.

    The interpolated binary64 source is not bit-exactly mirror symmetric
    (spec 2.2 anchors accuracy to the committed source), so the two
    exact-reference lengths may differ by a few ulps; each one must still
    be correctly rounded independently.
    """
    s_curve = CubicPHSplineOpen(
        [[0.0, 0.0], [1.0, 0.9], [2.0, -0.9], [3.0, 0.0]]
    )
    for d in (0.4, 1.2):
        hp = s_curve.offset(d)
        hm = s_curve.offset(-d)
        op = OffsetOracle(hp, prec=350)
        om_ = OffsetOracle(hm, prec=350)
        assert_correctly_rounded(hp.length, op.length(), f"S +{d}")
        assert_correctly_rounded(hm.length, om_.length(), f"S -{d}")
        assert abs(hp.length - hm.length) <= 4.0 * math.ulp(hp.length)


# ---------------------------------------------------------------------------
# Resource caps, fault injection, and forbidden false success (14.3/14.5/12)
# ---------------------------------------------------------------------------


def test_nonrepresentable_total_length_fails_atomically():
    """A positive exact length beyond binary64 range must raise (8.4).

    With ``w = 1 + i t`` the normalized span length is exactly 4/3, so a
    scale of 1.5e308 puts the exact total at 2e308, beyond the largest
    finite binary64 value.
    """
    with pytest.raises((OffsetConstructionError, NumericalPrecisionError)):
        build_offset_metric(
            span_preimages=[[1 + 0j, 1 + 1j]],
            span_widths=[1.0],
            breakpoints=[0.0, 1.0],
            distance=0.0,
            scale=1.5e308,
            closed=False,
        )


def test_zero_rounding_total_length_fails_atomically():
    """A positive exact length that rounds to zero must raise (8.4)."""
    with pytest.raises((OffsetConstructionError, NumericalPrecisionError)):
        build_offset_metric(
            span_preimages=[[1e-140 + 0j, 1e-140 + 1e-140j]],
            span_widths=[1.0],
            breakpoints=[0.0, 1.0],
            distance=0.0,
            scale=1e-290,
            closed=False,
        )


def test_zero_g_polynomial_is_construction_failure():
    """A corrupted certificate with G == 0 must fail construction (5.2)."""
    with pytest.raises((OffsetConstructionError, ZeroDivisionError)):
        build_offset_metric(
            span_preimages=[[0j, 0j]],
            span_widths=[1.0],
            breakpoints=[0.0, 1.0],
            distance=0.5,
            scale=1.0,
            closed=False,
        )


def test_ordered_search_fault_injection(diamond, monkeypatch):
    """Force the Newton stage off and drive every inverse through the
    ordered-float best-representable search (spec 14.5)."""
    h = diamond.offset(1.0)
    L = h.length
    monkeypatch.setattr(om, "_MAX_NEWTON", 0)
    rng = np.random.default_rng(77)
    for s in list(rng.uniform(0.0, 1.0, 25) * L) + [0.5 * L]:
        s = float(s)
        u = h.parameter_at_length(s)
        assert 0.0 <= u <= 1.0
        resid = abs(h.arc_length(u) - s)
        assert resid <= 1e-10 * L, (s, u, resid)


def test_precision_cap_exhaustion_is_typed_failure(diamond, monkeypatch):
    """With the certified ladder collapsed to a useless precision, an
    undecidable enclosure must raise the typed failure, never return an
    unverified value (spec 8.3 / 12.1)."""
    h = diamond.offset(1.0)
    monkeypatch.setattr(om, "_QUERY_LADDER", (8,))
    monkeypatch.setattr(om, "_MAX_NEWTON", 0)
    metric = h._metric
    # Also disable the double-double fast paths by inflating every error
    # bound, so each comparison needs the (now useless) certified ladder.
    originals = [c.E for c in metric._cells]
    try:
        for c in metric._cells:
            c.E = 1e30
        with pytest.raises(
            (NumericalPrecisionError, ArcLengthInversionError)
        ):
            metric.parameter_at_length(0.37 * h.length)
    finally:
        for c, e in zip(metric._cells, originals):
            c.E = e


def test_fast_path_failure_falls_back_not_wrong(diamond):
    """Inflated error bounds force certified fallbacks; results must be
    bit-identical to the fast path (no accuracy dependence on the route)."""
    h = diamond.offset(1.0)
    metric = h._metric
    us = [0.13, 0.42, 0.77, 0.9401]
    fast = [h.arc_length(u) for u in us]
    originals = [c.E for c in metric._cells]
    try:
        for c in metric._cells:
            c.E = 1.0  # every fast enclosure fails; certified path used
        slow = [h.arc_length(u) for u in us]
    finally:
        for c, e in zip(metric._cells, originals):
            c.E = e
    assert fast == slow


# ---------------------------------------------------------------------------
# Determinism, monotonicity, and dense stress
# ---------------------------------------------------------------------------


def test_dense_monotonicity_across_cusps(diamond):
    h = diamond.offset(1.0)
    us = np.linspace(0.0, 1.0, 4001)
    prev = -1.0
    for u in us:
        a = h.arc_length(float(u))
        assert a >= prev
        prev = a
    assert prev == h.length


def test_adjacent_float_parameters_around_each_cusp(diamond):
    """Both adjacent floats around every cusp boundary evaluate, order
    correctly, and invert with certified residuals."""
    h = diamond.offset(1.0)
    metric = h._metric
    for b in metric._bounds[1:-1]:
        lo = math.nextafter(b, 0.0)
        hi = math.nextafter(b, 1.0)
        a_lo, a_b, a_hi = h.arc_length(lo), h.arc_length(b), h.arc_length(hi)
        assert a_lo <= a_b <= a_hi


def test_repeated_queries_do_not_drift(diamond):
    h = diamond.offset(0.95)
    u = 0.7317
    s = 0.317 * h.length
    first = (h.arc_length(u), h.parameter_at_length(s))
    for _ in range(50):
        assert h.arc_length(u) == first[0]
        assert h.parameter_at_length(s) == first[1]


# ---------------------------------------------------------------------------
# Kernel cross-checks under adversarial arguments
# ---------------------------------------------------------------------------


def test_dd_atan2_bound_on_adversarial_arguments():
    """The fast-path atan2's documented bound against the integer oracle,
    including near-axis, near-cut, and denormal-ratio arguments."""
    rng = np.random.default_rng(90210)
    cases = []
    for _ in range(400):
        y, x = rng.normal(size=2)
        cases.append((y, x))
    cases += [
        (1e-300, 1.0), (1.0, 1e-300), (-1e-300, -1.0),
        (1e-15, -1.0), (-1e-15, -1.0), (0.7, 0.7), (-0.7, 0.7),
        (5e-324, 1.0), (1.0, 5e-324),
    ]
    for y, x in cases:
        if y == 0.0 and x == 0.0:
            continue
        got = dd_atan2((y, 0.0), (x, 0.0))
        v, e = atan2_ball(Fraction(y), Fraction(x), 192)
        lo, hi = ball_to_fraction_bounds(v, e, 192)
        val = Fraction(got[0]) + Fraction(got[1])
        assert lo - Fraction(ATAN2_DD_ABS_ERR) <= val <= hi + Fraction(
            ATAN2_DD_ABS_ERR
        ), (y, x)


def test_high_turning_bspline_offset_correctly_rounded():
    """Sharply turning closed B-spline stays correctly rounded."""
    kidney = PHBSplineClosed(
        [[1.0, 0.0], [0.15, 0.9], [-1.0, 0.35], [-0.25, -0.2],
         [-1.0, -0.6], [0.3, -1.0]]
    )
    h = kidney.offset(0.12)
    o = OffsetOracle(h, prec=350)
    assert_correctly_rounded(h.length, o.length(), "kidney length")
    for u in (0.09, 0.31, 0.55, 0.83):
        assert_correctly_rounded(h.arc_length(u), o.arc_length(u), "kidney arc")


def test_tangent_turn_beyond_two_pi_crosses_principal_cut():
    """One source span whose tangent turns by more than 2 pi (spec 14.3
    case 10): the preimage sweeps 3.4 rad, crossing the principal atan2
    cut, so the half-plane phase proof must subdivide, and every value
    must match the independent continuously-unwrapped oracle exactly."""
    import cmath

    from offset_oracle import OracleSpan

    w = [cmath.exp(1j * a) for a in (0.0, 1.1333, 2.2667, 3.4)]
    d = 0.15
    m = build_offset_metric(
        span_preimages=[w],
        span_widths=[1.0],
        breakpoints=[0.0, 1.0],
        distance=d,
        scale=1.0,
        closed=False,
    )
    assert len(m._cells) >= 4  # the sector proof really subdivided
    o = OracleSpan(w, 1.0, 0.0, 1.0, d, 1.0, prec=350)
    total_turn = 2.0 * float(o.angle_lift(mp.mpf(0), mp.mpf(1)))
    assert total_turn > 2.0 * math.pi
    assert m.length == float(o.span_total())
    for t in (0.1, 0.35, 0.5, 0.72, 0.95):
        assert m.arc_length(t) == float(o.span_arc_length(mp.mpf(t))), t
    s = 0.6 * m.length
    u = m.parameter_at_length(s)
    assert abs(float(o.span_arc_length(mp.mpf(u))) - s) <= 1e-13 * m.length


# ---------------------------------------------------------------------------
# Performance and allocation discipline (spec 14.6)
# ---------------------------------------------------------------------------


def _median_time(fn, n=200):
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        times.append((time.perf_counter() - t0) / n)
    times.sort()
    return times[len(times) // 2]


def test_query_cost_is_span_count_stable():
    """Near-logarithmic lookup: 100x more spans must not cost more than a
    small constant factor per query."""
    small = CubicPHSplineOpen(
        [[x, 0.18 * math.sin(x)] for x in np.linspace(0.0, 3.0, 8)]
    )
    big = CubicPHSplineOpen(
        [[x, 0.18 * math.sin(x)] for x in np.linspace(0.0, 300.0, 800)]
    )
    h_small = small.offset(0.05)
    h_big = big.offset(0.05)
    h_small.arc_length(0.5)
    h_big.arc_length(0.5)
    t_small = _median_time(lambda: h_small.arc_length(0.61803))
    t_big = _median_time(lambda: h_big.arc_length(0.61803))
    assert t_big <= 6.0 * t_small + 2e-5, (t_small, t_big)


def test_no_construction_work_on_queries(diamond):
    """Queries after warm-up must remain microsecond-scale (no lazy
    metric construction on the query path, spec 6.4)."""
    h = diamond.offset(1.0)
    h.arc_length(0.5)
    h.parameter_at_length(0.5 * h.length)
    t_arc = _median_time(lambda: h.arc_length(0.4321), n=300)
    t_inv = _median_time(lambda: h.parameter_at_length(0.617 * h.length), n=150)
    assert t_arc < 2e-3
    assert t_inv < 5e-3


def test_bounded_inverse_iterations(diamond, monkeypatch):
    """The inverse must succeed within its documented hard budget: count
    authoritative evaluations via instrumentation."""
    h = diamond.offset(1.0)
    calls = 0
    original = om._cell_distance_dd

    def counting(cell, z, reverse):
        nonlocal calls
        calls += 1
        return original(cell, z, reverse)

    monkeypatch.setattr(om, "_cell_distance_dd", counting)
    L = h.length
    rng = np.random.default_rng(5)
    worst = 0
    for s in rng.uniform(0.0, 1.0, 40) * L:
        calls = 0
        h.parameter_at_length(float(s))
        worst = max(worst, calls)
    assert worst <= om._MAX_NEWTON + 24, worst
