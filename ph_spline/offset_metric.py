"""Verified traversal-distance metric certificate for exact PH offsets.

This module implements the normative addendum
``OffsetNURBS_Distance_Specification.md``: every exact offset NURBS handle
carries a *metric certificate* built here at offset construction from the
captured PH preimage data, and the four public distance queries
(``length``, ``arc_length``, ``parameter_at_length``, ``point_at_length``)
evaluate against that certificate.

Mathematical model (spec sections 4-5).  On one source span with local
parameter ``t`` and complex Bernstein preimage ``w`` of degree ``m``:

- ``sigma = |w|^2`` (degree ``2m``) and ``tau = 2 Im(conj(w) w')``
  (degree ``2m - 1``);
- the normalized cusp polynomial is ``G = h sigma^2 - (d / H) tau``
  (degree ``4m``), whose real roots are the offset cusps;
- the elementary unsigned offset distance over a constant-sign cell is
  ``eta * (H h (b - a) Q_f(x) - 2 d dphi)``, where ``Q_f`` is the compiled
  Bernstein antiderivative of the restricted speed and ``dphi`` the
  continuous preimage-angle increment (spec 4.6.2).

Phase-lift proof (spec sections 4.5 / 7.5).  Instead of the cut-crossing
winding counter, this implementation uses the explicitly permitted
half-plane subdivision proof: every metric cell is subdivided until the
exact rational Bernstein hull of its restricted preimage lies strictly in
the open half planes of both endpoint preimages.  The preimage then turns
by less than ``pi/2`` relative to either endpoint across the whole cell,
so the winding correction is identically zero and the continuous angle
increment equals one principal ``atan2`` of exact cross/dot products.

Arithmetic layers:

- construction runs in exact rational arithmetic
  (:mod:`ph_spline.exact_real`); transcendental angle terms are enclosed
  by scaled-integer balls at 320+ bits;
- ordinary queries run in double-double arithmetic
  (:mod:`ph_spline.ddouble`) against compiled per-cell polynomials with a
  stored certified error bound; a fast result is published only when its
  enclosure determines the correctly rounded public value, which implies
  the faithful rounding and the public monotonicity required by spec
  11.1 / 3.2;
- an undecided fast value escalates through the certified integer ladder
  ``(256, 512, 1024, 2048, 4096)`` bits; the documented cap of 4096 bits
  raises :class:`NumericalPrecisionError` (spec 8.3).

Representable cell boundaries.  A nonrepresentable algebraic cusp keeps
its certified isolating interval; the stored cell boundary is a binary64
parameter refined to within one ulp of the true root.  The mis-signed
micro-interval between the stored boundary and the exact root contributes
at most second order (the speed vanishes at the root) and is folded into
the cell's certified error bound, so exact-reference enclosures remain
valid.

Hard resource bounds (spec 12.2) are module constants below; hitting any
bound raises a typed exception, never returns an unverified value.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from fractions import Fraction
from math import comb, ulp
from struct import pack, unpack
from typing import NamedTuple

from ph_spline.ddouble import (
    ATAN2_DD_ABS_ERR,
    dd_add,
    dd_atan2,
    dd_div,
    dd_horner,
    dd_mul,
    dd_mul_f,
    dd_sub,
    two_sum,
)
from ph_spline.exact_real import (
    atan2_ball,
    ball_to_fraction_bounds,
    bern_antiderivative,
    bern_eval,
    bern_product,
    bern_restrict,
    bern_to_power,
    isolate_bernstein_roots,
    power_derivative,
    power_mul,
    refine_root_to_floats,
)
from ph_spline.exceptions import (
    ArcLengthInversionError,
    NumericalPrecisionError,
    OffsetConstructionError,
)

__all__ = [
    "MAX_ADAPTIVE_PRECISION_BITS",
    "OffsetCusp",
    "OffsetMetric",
    "build_offset_metric",
    "rebuild_offset_metric",
]

_EPS = 2.0**-52
_TINY = 5e-324
_ZERO = Fraction(0)

#: Construction-time escalation ladder for cell-length and prefix
#: enclosures (bits).
_BUILD_LADDER = (320, 640, 1280, 2560)

#: Query-time certified escalation ladder; the last entry is the
#: documented adaptive-precision cap of the binary64 profile (spec 8.3
#: requires at least 4096 bits).
_QUERY_LADDER = (256, 512, 1024, 2048, 4096)
MAX_ADAPTIVE_PRECISION_BITS = _QUERY_LADDER[-1]

#: Hard bound on safeguarded Newton/bisection evaluations per inverse.
_MAX_NEWTON = 48

#: Hard bound on ordered-float bisection steps (64 suffices for binary64
#: parameters in [0, 1]; the margin covers bracket bookkeeping).
_MAX_ORDERED_SEARCH = 128

#: Hard bound on half-plane sector subdivision depth per cell.
_MAX_SECTOR_DEPTH = 26

#: Deterministic exact sample points for the cell-sign certificate.
_ETA_SAMPLES = (
    Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(1, 5),
    Fraction(2, 5), Fraction(3, 5), Fraction(4, 5), Fraction(1, 7),
    Fraction(3, 7), Fraction(5, 7), Fraction(1, 11), Fraction(4, 11),
    Fraction(7, 11), Fraction(9, 11), Fraction(1, 13), Fraction(12, 13),
)

#: Safety factor applied to composed fast-path error bounds.
_E_SAFETY = 8.0


def _fail(message: str, **fields) -> OffsetConstructionError:
    return OffsetConstructionError(message, operation="offset-metric", **fields)


class OffsetCusp(NamedTuple):
    """One certified offset cusp (zero-speed parameter of the offset locus).

    ``parameter`` is a global parameter within two ulps of the exact
    stationary parameter, taken from the certified root bracket of the
    cusp polynomial ``G``.  ``multiplicity`` is the certified root
    multiplicity: odd means the offset hodograph reverses direction
    through the cusp, even means a tangential zero-speed graze without a
    reversal.  Two distinct roots closer than one representable parameter
    coalesce into a single record.
    """

    parameter: float
    multiplicity: int


# ---------------------------------------------------------------------------
# Exact per-span certificate
# ---------------------------------------------------------------------------


class _SpanExact:
    """Captured immutable exact metric data of one source span."""

    __slots__ = (
        "G",
        "angle",
        "h",
        "m",
        "mult0",
        "mult1",
        "rho",
        "tau",
        "tau_zero",
        "u0",
        "u1",
        "wim",
        "wre",
    )


def _elevate_to(c: list[Fraction], degree: int) -> list[Fraction]:
    """Exact Bernstein degree elevation to the requested degree."""
    n = len(c) - 1
    r = degree - n
    if r == 0:
        return list(c)
    out = []
    for k in range(degree + 1):
        acc = _ZERO
        for j in range(max(0, k - r), min(n, k) + 1):
            acc += comb(n, j) * comb(r, k - j) * c[j]
        out.append(acc / comb(degree, k))
    return out


def _compile_span_exact(
    preimage: list[complex],
    width: float,
    u0: float,
    u1: float,
    d_hat: Fraction,
    span_id: int,
    distance: float,
) -> _SpanExact:
    """Build and cross-verify the exact metric polynomials of one span.

    Implements the developer formulas of spec section 4.4 in exact
    rational arithmetic and verifies every coefficient identity of spec
    sections 7.2 / 14.4 against an independent power-basis construction.
    All verifications are exact equalities of rationals; any mismatch is
    an atomic construction failure.
    """
    span = _SpanExact()
    span.u0 = u0
    span.u1 = u1
    span.mult0 = 0
    span.mult1 = 0
    m = len(preimage) - 1
    span.m = m
    span.h = Fraction(width)
    if not span.h > 0:
        raise _fail(
            "Span parameter width factor must be positive",
            span_id=span_id,
            quantity="h",
            value=width,
            distance=distance,
        )
    wre = [Fraction(z.real) for z in preimage]
    wim = [Fraction(z.imag) for z in preimage]
    span.wre = wre
    span.wim = wim

    # sigma = |w|^2, spec 4.4.2 boxed formula.
    rho = []
    for k in range(2 * m + 1):
        acc = _ZERO
        for i in range(max(0, k - m), min(m, k) + 1):
            j = k - i
            acc += comb(m, i) * comb(m, j) * (wre[i] * wre[j] + wim[i] * wim[j])
        rho.append(acc / comb(2 * m, k))
    span.rho = rho

    # tau = 2 Im(conj(w) w'), spec 4.4.3 boxed formula (degree 2m - 1).
    if m == 0:
        span.tau = None
        span.tau_zero = True
    else:
        ere = [m * (wre[j + 1] - wre[j]) for j in range(m)]
        eim = [m * (wim[j + 1] - wim[j]) for j in range(m)]
        tau = []
        for k in range(2 * m):
            acc = _ZERO
            for i in range(max(0, k - m + 1), min(m, k) + 1):
                j = k - i
                acc += comb(m, i) * comb(m - 1, j) * (
                    wre[i] * eim[j] - wim[i] * ere[j]
                )
            tau.append(2 * acc / comb(2 * m - 1, k))
        span.tau = tau
        span.tau_zero = all(v == 0 for v in tau)

    # Independent power-basis verification (spec 7.2): sigma and tau
    # recomputed through a different pipeline must agree exactly.
    wre_p = bern_to_power(wre)
    wim_p = bern_to_power(wim)
    sigma_pow = [
        a + b for a, b in zip(power_mul(wre_p, wre_p), power_mul(wim_p, wim_p))
    ]
    if bern_to_power(rho) != sigma_pow:
        raise _fail(
            "Bernstein sigma coefficients disagree with the power-basis "
            "product |w|^2",
            span_id=span_id,
            quantity="sigma coefficients",
            distance=distance,
        )
    tau_pow: list[Fraction] = []
    if m >= 1:
        dre = power_derivative(wre_p)
        dim = power_derivative(wim_p)
        tau_pow = [
            2 * (a - b)
            for a, b in zip(power_mul(wre_p, dim), power_mul(wim_p, dre))
        ]
        tau_check = bern_to_power(span.tau)
        n_c = max(len(tau_pow), len(tau_check))
        tau_pow = tau_pow + [_ZERO] * (n_c - len(tau_pow))
        tau_check = tau_check + [_ZERO] * (n_c - len(tau_check))
        if tau_pow != tau_check:
            raise _fail(
                "Bernstein tau coefficients disagree with the power-basis "
                "form 2 Im(conj(w) w')",
                span_id=span_id,
                quantity="tau coefficients",
                distance=distance,
            )
        # The apparent degree-(2m-1) power coefficient must cancel exactly
        # (spec 4.4.3).
        if len(tau_check) >= 2 * m and tau_check[2 * m - 1] != 0:
            raise _fail(
                "The degree-(2m-1) leading power coefficient of tau did "
                "not cancel exactly",
                span_id=span_id,
                quantity="tau leading coefficient",
                value=float(tau_check[2 * m - 1]),
                distance=distance,
            )

    # G = h sigma^2 - d_hat tau at degree 4m (spec 4.4.4).
    sigma_sq = bern_product(rho, rho)
    if span.tau_zero or d_hat == 0:
        G = [span.h * v for v in sigma_sq]
        span.angle = False
    else:
        tau_elev = _elevate_to(span.tau, 4 * m)
        G = [span.h * s2 - d_hat * te for s2, te in zip(sigma_sq, tau_elev)]
        span.angle = True
    span.G = G
    if all(v == 0 for v in G):
        raise _fail(
            "The offset cusp polynomial vanished identically; the captured "
            "source data is corrupted (spec 5.2)",
            span_id=span_id,
            quantity="G coefficients",
            distance=distance,
        )
    # Independent power-basis verification of G.
    g_pow = [span.h * v for v in power_mul(sigma_pow, sigma_pow)]
    if span.angle:
        for k, v in enumerate(tau_pow):
            g_pow[k] -= d_hat * v
    g_check = bern_to_power(G)
    n_c = max(len(g_pow), len(g_check))
    g_pow = g_pow + [_ZERO] * (n_c - len(g_pow))
    g_check = g_check + [_ZERO] * (n_c - len(g_check))
    if g_pow != g_check:
        raise _fail(
            "Bernstein G coefficients disagree with the power-basis form "
            "h sigma^2 - d_hat tau",
            span_id=span_id,
            quantity="G coefficients",
            distance=distance,
        )
    return span


# ---------------------------------------------------------------------------
# Metric cells
# ---------------------------------------------------------------------------


class _Cell:
    """One compiled constant-sign metric cell (spec section 6.2)."""

    __slots__ = (
        "E",
        "a_loc",
        "angle",
        "b_loc",
        "ca",
        "cs",
        "cusp0",
        "cusp1",
        "eta",
        "g_rev",
        "g_scaled",
        "ga",
        "gb",
        "kv",
        "length_dd",
        "length_exact",
        "length_float",
        "lut",
        "pf",
        "pr",
        "sig_rev",
        "sig_scaled",
        "span",
        "w0",
        "w1",
        "wim_p",
        "wim_pr",
        "wre_p",
        "wre_pr",
    )


def _to_float(x: Fraction) -> float:
    """Correctly rounded float of a rational; overflow becomes infinity."""
    try:
        return float(x)
    except OverflowError:
        return math.inf if x > 0 else -math.inf


def _frac_to_dd(x: Fraction) -> tuple[float, float]:
    hi = _to_float(x)
    if not math.isfinite(hi):
        return hi, 0.0
    lo = float(x - Fraction(hi))
    return hi, lo


def _power_to_dd(
    p: list[Fraction],
) -> tuple[tuple[tuple[float, float], ...], float]:
    """Highest-first double-double power coefficients plus |coeff| sum."""
    coeffs = tuple(_frac_to_dd(c) for c in reversed(p))
    total = float(sum(abs(c) for c in p)) * (1.0 + 1e-15) + _TINY
    return coeffs, total


def _scaled_bernstein(c: list[float]) -> tuple[float, ...]:
    """Coefficients ``c_k * C(n, k)`` for the O(n) seed evaluator."""
    n = len(c) - 1
    return tuple(c[k] * comb(n, k) for k in range(n + 1))


def _bern_eval_scaled(
    scaled: tuple[float, ...], rev: tuple[float, ...], x: float
) -> float:
    """O(n) Bernstein evaluation from scaled coefficients.

    Uses the ratio form on the half interval where the ratio is at most
    one.  Seed/derivative use only; never an authoritative residual.
    """
    n = len(scaled) - 1
    if x <= 0.5:
        r = x / (1.0 - x)
        acc = scaled[n]
        for k in range(n - 1, -1, -1):
            acc = acc * r + scaled[k]
        return acc * (1.0 - x) ** n
    r = (1.0 - x) / x
    acc = rev[n]
    for k in range(n - 1, -1, -1):
        acc = acc * r + rev[k]
    return acc * x**n


def _cell_distance_dd(
    cell: _Cell, z: tuple[float, float], reverse: bool
) -> tuple[float, float]:
    """Authoritative unsigned distance from the near cell end to ``z``.

    ``z`` is the forward coordinate ``(u - ga) / (gb - ga)`` or the
    reverse coordinate ``(gb - u) / (gb - ga)`` as a double-double.
    Implements the boxed production formulas of spec 4.6.2 / 4.6.4; the
    caller pairs the value with the cell's certified error bound ``E``.
    """
    if reverse:
        qs = dd_horner(cell.pr, z)
    else:
        qs = dd_horner(cell.pf, z)
    ds = dd_mul(cell.cs, qs)
    if not cell.angle:
        return ds if cell.eta > 0 else (-ds[0], -ds[1])
    if reverse:
        wr = dd_horner(cell.wre_pr, z)
        wi = dd_horner(cell.wim_pr, z)
        ar, ai = cell.w1
        # X + iY = w(b) * conj(w(t))   (spec 4.6.4)
        x_dd = dd_add(dd_mul(ar, wr), dd_mul(ai, wi))
        y_dd = dd_sub(dd_mul(ai, wr), dd_mul(ar, wi))
    else:
        wr = dd_horner(cell.wre_p, z)
        wi = dd_horner(cell.wim_p, z)
        ar, ai = cell.w0
        # X + iY = w(t) * conj(w(a))   (spec 4.6.2)
        x_dd = dd_add(dd_mul(wr, ar), dd_mul(wi, ai))
        y_dd = dd_sub(dd_mul(wi, ar), dd_mul(wr, ai))
    theta = dd_atan2(y_dd, x_dd)
    val = dd_sub(ds, dd_mul_f(theta, cell.ca))
    return val if cell.eta > 0 else (-val[0], -val[1])


# ---------------------------------------------------------------------------
# Construction driver
# ---------------------------------------------------------------------------


def build_offset_metric(
    *,
    span_preimages: list,
    span_widths: list,
    breakpoints,
    distance: float,
    scale: float,
    closed: bool,
) -> "OffsetMetric":
    """Compile and verify the complete metric certificate.

    ``span_preimages[i]`` is the complex Bernstein preimage of source
    span ``i`` in the family's normalized frame, ``span_widths[i]`` its
    local parameter-width factor (``1.0`` for cubic spans),
    ``breakpoints`` the global span boundaries, ``distance`` the signed
    physical offset, and ``scale`` the spatial normalization ``H_x``.

    Returns a completely verified metric or raises
    :class:`OffsetConstructionError`; a partial certificate can never be
    observed (spec 6.1 / 6.4).
    """
    H = Fraction(scale)
    if not H > 0:
        raise _fail("Spatial scale must be positive", quantity="H", value=scale)
    d = float(distance)
    d_frac = Fraction(d)
    d_hat = d_frac / H

    spans: list[_SpanExact] = []
    for i in range(len(span_preimages)):
        spans.append(
            _compile_span_exact(
                [complex(z) for z in span_preimages[i]],
                float(span_widths[i]),
                float(breakpoints[i]),
                float(breakpoints[i + 1]),
                d_hat,
                i,
                d,
            )
        )

    metric = OffsetMetric()
    metric._d = d
    metric._d_hat = d_hat
    metric._H = H
    metric._scale = float(scale)
    metric._spans = spans
    metric._closed = bool(closed)

    cells: list[_Cell] = []
    cusp_records: dict[float, int] = {}
    for i, span in enumerate(spans):
        span_cells, span_cusps = _build_span_cells(span, i, H, d_frac, d)
        cells.extend(span_cells)
        for g, mult in span_cusps:
            cusp_records[g] = max(cusp_records.get(g, 0), mult)
    if not cells:
        raise _fail("No metric cells were produced", distance=d)
    metric._cusps = tuple(
        OffsetCusp(g, cusp_records[g]) for g in sorted(cusp_records)
    )

    _accumulate_prefixes(metric, cells, d)
    _publication_checks(metric, d)
    return metric


def rebuild_offset_metric(state: dict) -> OffsetMetric:
    """Rebuild and fully re-verify a metric from its serialized state."""
    return build_offset_metric(
        span_preimages=[
            [complex(re, im) for re, im in span] for span in state["preimages"]
        ],
        span_widths=state["widths"],
        breakpoints=state["breakpoints"],
        distance=state["distance"],
        scale=state["scale"],
        closed=state["closed"],
    )


def _build_span_cells(
    span: _SpanExact,
    span_id: int,
    H: Fraction,
    d_frac: Fraction,
    d: float,
) -> tuple[list[_Cell], list[tuple[float, int]]]:
    """Partition one span into verified constant-sign sector cells.

    Also returns the certified cusp records ``(global parameter,
    multiplicity)`` of this span: the refined interior roots of ``G`` plus
    any exact roots at the span endpoints.
    """
    u0, u1 = span.u0, span.u1
    boundaries: list[float] = [u0]
    cusp_mult: dict[float, int] = {}
    edge_records: list[tuple[float, int]] = []

    if span.angle:
        coeffs = span.G
        if not (all(c > 0 for c in coeffs) or all(c < 0 for c in coeffs)):
            m0, m1, interior = isolate_bernstein_roots(coeffs)
            span.mult0 = m0
            span.mult1 = m1
            for lo, hi, mult, refiner in interior:
                if lo == hi:
                    t_b = float(lo)
                else:
                    _, t_b = refine_root_to_floats(refiner, lo, hi)
                g = u0 + t_b * (u1 - u0)
                if not u0 < g < u1:
                    # An interior root whose representable parameter rounds
                    # onto the span edge: no interior cell boundary, but it
                    # stays in the public cusp list at the edge parameter.
                    edge_records.append((u0 if g <= u0 else u1, mult))
                    continue
                if g <= boundaries[-1]:
                    # Two roots within one representable parameter; the
                    # merged micro-interval is absorbed into the error
                    # bound of the adjacent cells.
                    cusp_mult[boundaries[-1]] = max(
                        cusp_mult.get(boundaries[-1], 0), mult
                    )
                    continue
                boundaries.append(g)
                cusp_mult[g] = mult
    boundaries.append(u1)

    cells: list[_Cell] = []
    stack = [
        (boundaries[k], boundaries[k + 1], 0)
        for k in range(len(boundaries) - 2, -1, -1)
    ]
    du = Fraction(u1) - Fraction(u0)
    while stack:
        ga, gb, depth = stack.pop()
        a_loc = (Fraction(ga) - Fraction(u0)) / du
        b_loc = (Fraction(gb) - Fraction(u0)) / du
        whole = a_loc == 0 and b_loc == 1
        wre_r = list(span.wre) if whole else bern_restrict(span.wre, a_loc, b_loc)
        wim_r = list(span.wim) if whole else bern_restrict(span.wim, a_loc, b_loc)
        if span.angle and not _sector_certified(wre_r, wim_r):
            if depth >= _MAX_SECTOR_DEPTH:
                raise _fail(
                    "Half-plane sector subdivision exceeded its depth bound",
                    span_id=span_id,
                    quantity="sector depth",
                    value=depth,
                    bound=f"< {_MAX_SECTOR_DEPTH}",
                    distance=d,
                )
            gm = ga + 0.5 * (gb - ga)
            if not ga < gm < gb:
                raise _fail(
                    "Sector subdivision exhausted representable breakpoints",
                    span_id=span_id,
                    quantity="cell width",
                    value=gb - ga,
                    distance=d,
                )
            stack.append((gm, gb, depth + 1))
            stack.append((ga, gm, depth + 1))
            continue
        cells.append(
            _compile_cell(
                span, span_id, ga, gb, a_loc, b_loc, wre_r, wim_r,
                H, d_frac, d, cusp_mult,
            )
        )
    cusps = list(cusp_mult.items()) + edge_records
    if span.mult0:
        cusps.append((u0, span.mult0))
    if span.mult1:
        cusps.append((u1, span.mult1))
    return cells, cusps


def _sector_certified(wre_r: list[Fraction], wim_r: list[Fraction]) -> bool:
    """Exact half-plane test of the restricted preimage hull (spec 7.5).

    Certifies that every control lies strictly inside the open half plane
    of both endpoint controls.  By the Bernstein convex-hull property the
    preimage then turns by less than ``pi/2`` relative to either endpoint
    over the whole cell, so every principal relative ``atan2`` inside the
    cell equals the continuous angle increment (winding correction zero).
    """
    for anchor_re, anchor_im in (
        (wre_r[0], wim_r[0]),
        (wre_r[-1], wim_r[-1]),
    ):
        for cr, ci in zip(wre_r, wim_r):
            if cr * anchor_re + ci * anchor_im <= 0:
                return False
    return True


def _compile_cell(
    span: _SpanExact,
    span_id: int,
    ga: float,
    gb: float,
    a_loc: Fraction,
    b_loc: Fraction,
    wre_r: list[Fraction],
    wim_r: list[Fraction],
    H: Fraction,
    d_frac: Fraction,
    d: float,
    cusp_mult: dict[float, int],
) -> _Cell:
    cell = _Cell()
    cell.span = span_id
    cell.ga = ga
    cell.gb = gb
    cell.a_loc = a_loc
    cell.b_loc = b_loc
    cell.angle = span.angle
    cell.ca = 2.0 * d

    whole = a_loc == 0 and b_loc == 1
    r = list(span.rho) if whole else bern_restrict(span.rho, a_loc, b_loc)
    g_r = list(span.G) if whole else bern_restrict(span.G, a_loc, b_loc)

    # eta: certified exact sign of G at interior sample points (spec 7.4).
    if not span.angle:
        eta = 1
    else:
        eta = 0
        for t_s in _ETA_SAMPLES:
            v = bern_eval(g_r, t_s)
            if v > 0:
                eta = 1
                break
            if v < 0:
                eta = -1
                break
        if eta == 0:
            raise _fail(
                "Could not certify the constant sign of G on a metric cell",
                span_id=span_id,
                quantity="eta samples",
                value=len(_ETA_SAMPLES),
                distance=d,
            )
    cell.eta = eta

    # Compiled antiderivatives (spec 4.6.2), forward and reverse.
    anti_f = bern_antiderivative(r)
    anti_r = bern_antiderivative(list(reversed(r)))
    if anti_f[-1] != anti_r[-1]:
        raise _fail(
            "Forward and reverse antiderivative totals disagree",
            span_id=span_id,
            quantity="antiderivative total",
            distance=d,
        )
    # Exact derivative identity R' = sigma (spec 7.8 item 7).
    qf_pow = bern_to_power(anti_f)
    r_pow = bern_to_power(r)
    dq = power_derivative(qf_pow)
    n_c = max(len(dq), len(r_pow))
    if dq + [_ZERO] * (n_c - len(dq)) != r_pow + [_ZERO] * (n_c - len(r_pow)):
        raise _fail(
            "Compiled antiderivative violates the derivative identity",
            span_id=span_id,
            quantity="Q_f' - sigma",
            distance=d,
        )
    cell.pf, sum_pf = _power_to_dd(qf_pow)
    cell.pr, _ = _power_to_dd(bern_to_power(anti_r))

    cs_frac = H * span.h * (b_loc - a_loc)
    cell.cs = _frac_to_dd(cs_frac)
    cs_abs = abs(float(cs_frac))

    # Seed evaluators (never authoritative).
    g_f = [float(v) for v in g_r]
    sig_f = [float(v) for v in r]
    cell.g_scaled = _scaled_bernstein(g_f)
    cell.g_rev = _scaled_bernstein(list(reversed(g_f)))
    cell.sig_scaled = _scaled_bernstein(sig_f)
    cell.sig_rev = _scaled_bernstein(list(reversed(sig_f)))
    cell.kv = float(H * (b_loc - a_loc))

    e_s = 8.0 * (len(cell.pf) + 2) * 2.0**-104 * cs_abs * sum_pf
    e_angle = 0.0
    theta_args = None
    if span.angle:
        cell.wre_p, w_sum_re = _power_to_dd(bern_to_power(wre_r))
        cell.wim_p, w_sum_im = _power_to_dd(bern_to_power(wim_r))
        cell.wre_pr, _ = _power_to_dd(bern_to_power(list(reversed(wre_r))))
        cell.wim_pr, _ = _power_to_dd(bern_to_power(list(reversed(wim_r))))
        cell.w0 = (_frac_to_dd(wre_r[0]), _frac_to_dd(wim_r[0]))
        cell.w1 = (_frac_to_dd(wre_r[-1]), _frac_to_dd(wim_r[-1]))
        # Certified lower bound of |w| on the cell from the half-plane
        # projection: |w(t)| >= min_j Re(c_j conj(c_0)) / |c_0|.
        proj = min(
            cr * wre_r[0] + ci * wim_r[0] for cr, ci in zip(wre_r, wim_r)
        )
        w_min_sq = proj * proj / (wre_r[0] * wre_r[0] + wim_r[0] * wim_r[0])
        w_min = math.sqrt(float(w_min_sq)) * (1.0 - 1e-14)
        if not w_min > 0.0:
            raise _fail(
                "Certified preimage lower bound collapsed",
                span_id=span_id,
                quantity="W_min",
                value=w_min,
                distance=d,
            )
        w_sum = w_sum_re + w_sum_im
        w_err = 8.0 * (len(cell.wre_p) + 4) * 2.0**-104 * w_sum
        e_angle = abs(cell.ca) * (ATAN2_DD_ABS_ERR + 4.0 * w_err / w_min)
        # Exact relative rotation across the whole cell.
        x_arg = wre_r[-1] * wre_r[0] + wim_r[-1] * wim_r[0]
        y_arg = wim_r[-1] * wre_r[0] - wre_r[-1] * wim_r[0]
        theta_args = (y_arg, x_arg)
    else:
        cell.wre_p = cell.wim_p = cell.wre_pr = cell.wim_pr = ()
        cell.w0 = cell.w1 = ((0.0, 0.0), (0.0, 0.0))

    # Sliver protection (module docstring): the mis-signed micro-interval
    # next to a *root-derived* representable boundary, second order in the
    # (there tiny) edge speed.  Exact span knots and subdivision midpoints
    # are exact boundaries and carry no sliver.
    width = gb - ga
    dx_sliver = 4.0 * ulp(max(abs(ga), abs(gb), 1e-300)) / width
    e_sliver = 0.0
    if ga in cusp_mult:
        e_sliver += (
            4.0 * dx_sliver * cell.kv * abs(g_f[0]) / max(sig_f[0], _TINY)
        )
    if gb in cusp_mult:
        e_sliver += (
            4.0 * dx_sliver * cell.kv * abs(g_f[-1]) / max(sig_f[-1], _TINY)
        )

    cell.E = _E_SAFETY * (e_s + e_angle + e_sliver)

    # Exact cell length with a certified angle enclosure (spec 4.6.3/7.6).
    ds_total = cs_frac * anti_f[-1]
    if theta_args is None:
        length_lo = length_hi = eta * ds_total
    else:
        length_lo = length_hi = None
        for prec in _BUILD_LADDER:
            v, e = atan2_ball(theta_args[0], theta_args[1], prec)
            th_lo, th_hi = ball_to_fraction_bounds(v, e, prec)
            c1 = eta * (ds_total - 2 * d_frac * th_lo)
            c2 = eta * (ds_total - 2 * d_frac * th_hi)
            length_lo, length_hi = (c1, c2) if c1 <= c2 else (c2, c1)
            if length_lo > 0:
                break
        if not length_lo > 0:
            raise _fail(
                "A metric cell length could not be certified positive",
                span_id=span_id,
                quantity="cell length enclosure",
                value=(float(length_lo), float(length_hi)),
                bound="> 0",
                distance=d,
            )
    cell.length_exact = (length_lo, length_hi)
    mid = (length_lo + length_hi) / 2
    cell.length_dd = _frac_to_dd(mid)
    cell.length_float = _to_float(mid)

    # Cusp seed data (spec 7.7): multiplicity and leading coefficient of
    # the local law D ~ C_r |dz|^(r+1).
    cell.cusp0 = cell.cusp1 = None
    mult0 = cusp_mult.get(ga, span.mult0 if a_loc == 0 else 0)
    mult1 = cusp_mult.get(gb, span.mult1 if b_loc == 1 else 0)
    if mult0:
        cell.cusp0 = (
            mult0,
            _cusp_coefficient(g_r, sig_f[0], cell.kv, mult0, False),
        )
    if mult1:
        cell.cusp1 = (
            mult1,
            _cusp_coefficient(g_r, sig_f[-1], cell.kv, mult1, True),
        )

    cell.lut = ()
    return cell


def _cusp_coefficient(
    g_r: list[Fraction], sigma_edge: float, kv: float, mult: int, right: bool
) -> float:
    """Leading coefficient of the cusp inverse-seed law (spec 7.7)."""
    n = len(g_r) - 1
    coeffs = list(reversed(g_r)) if right else g_r
    delta = _ZERO
    for j in range(mult + 1):
        term = comb(mult, j) * coeffs[j]
        delta += term if (mult - j) % 2 == 0 else -term
    lead = abs(float(comb(n, mult) * delta))
    if lead == 0.0:
        lead = _TINY
    return kv * lead / (max(sigma_edge, _TINY) * (mult + 1))


# ---------------------------------------------------------------------------
# Prefix accumulation and publication checks
# ---------------------------------------------------------------------------


def _accumulate_prefixes(metric: "OffsetMetric", cells: list[_Cell], d: float) -> None:
    """Accumulate certified extended prefixes (spec 6.3).

    Every prefix is stored three ways: exact rational enclosure at
    construction precision, a double-double anchor, and the correctly
    rounded public float.  Publication requires the rational enclosure to
    determine the rounding of every prefix and of the total length.
    """
    lo_acc = _ZERO
    hi_acc = _ZERO
    bounds = [cells[0].ga]
    pre_lo: list[Fraction] = [_ZERO]
    pre_hi: list[Fraction] = [_ZERO]
    for cell in cells:
        lo_acc = lo_acc + cell.length_exact[0]
        hi_acc = hi_acc + cell.length_exact[1]
        bounds.append(cell.gb)
        pre_lo.append(lo_acc)
        pre_hi.append(hi_acc)

    prefix_dd = []
    prefix_float = []
    for lo, hi in zip(pre_lo, pre_hi):
        f_lo = _to_float(lo)
        f_hi = _to_float(hi)
        if not math.isfinite(f_hi):
            raise _fail(
                "Exact-reference total offset length exceeds the finite "
                "binary64 range (spec 8.4)",
                quantity="prefix",
                value=f_hi,
                distance=d,
            )
        if f_lo != f_hi:
            raise NumericalPrecisionError(
                "A global prefix rounding could not be determined at "
                "construction precision",
                quantity="prefix enclosure",
                value=(f_lo, f_hi),
            )
        prefix_dd.append(_frac_to_dd((lo + hi) / 2))
        prefix_float.append(f_lo)

    total = prefix_float[-1]
    if not (math.isfinite(total) and total > 0.0):
        raise _fail(
            "Total offset length is not a finite positive binary64 value "
            "(spec 8.4)",
            quantity="length",
            value=total,
            distance=d,
        )
    for k in range(1, len(prefix_float)):
        if not pre_lo[k] > pre_hi[k - 1]:
            raise _fail(
                "Extended prefixes are not strictly increasing",
                index=k,
                quantity="prefix order",
                distance=d,
            )

    metric._cells = cells
    metric._bounds = bounds
    metric._prefix_dd = prefix_dd
    metric._prefix_float = prefix_float
    metric._prefix_exact = list(zip(pre_lo, pre_hi))
    metric._length_float = total
    metric._length_dd = prefix_dd[-1]

    # Seed tables through the authoritative evaluator (spec 7.7).
    for cell in cells:
        nodes = []
        for x in (0.25, 0.5, 0.75):
            val = _cell_distance_dd(cell, (x, 0.0), False)
            nodes.append((x, val[0] + val[1]))
        cell.lut = tuple(nodes)


def _publication_checks(metric: "OffsetMetric", d: float) -> None:
    """Structural publication battery (spec 7.8)."""
    bounds = metric._bounds
    cells = metric._cells
    if bounds[0] != cells[0].ga or not all(
        bounds[k + 1] > bounds[k] for k in range(len(bounds) - 1)
    ):
        raise _fail("Metric cell coverage is not strictly increasing", distance=d)
    for j, cell in enumerate(cells):
        if j and cells[j - 1].gb != cell.ga:
            raise _fail("Metric cells have a coverage gap", index=j, distance=d)
        if (
            j
            and cells[j - 1].span == cell.span
            and cell.angle
            and (
                cells[j - 1].w1[0] != cell.w0[0]
                or cells[j - 1].w1[1] != cell.w0[1]
            )
        ):
            raise _fail(
                "Adjacent cell preimage anchors disagree (phase continuity)",
                index=j,
                distance=d,
            )
        # Forward/reverse agreement over the full cell (spec 7.8 item 6).
        fwd = _cell_distance_dd(cell, (1.0, 0.0), False)
        rev = _cell_distance_dd(cell, (1.0, 0.0), True)
        limit = 2.0 * cell.E + 4.0 * ulp(max(cell.length_float, 1e-300)) + 1e-300
        gap = abs((fwd[0] - rev[0]) + (fwd[1] - rev[1]))
        if gap > limit:
            raise _fail(
                "Forward and reverse cell evaluations disagree",
                index=j,
                quantity="|D_f(1) - D_r(1)|",
                value=gap,
                bound=f"<= {limit:.3e}",
                distance=d,
            )
        gap_len = abs(
            (fwd[0] - cell.length_dd[0]) + (fwd[1] - cell.length_dd[1])
        )
        if gap_len > limit:
            raise _fail(
                "Cell length disagrees with the compiled evaluator",
                index=j,
                quantity="|D_f(1) - l|",
                value=gap_len,
                bound=f"<= {limit:.3e}",
                distance=d,
            )
        # Interior agreement with the certified integer oracle (item 8).
        lo, hi = _certified_cell_distance(metric, cell, Fraction(1, 2), 256)
        val = _cell_distance_dd(cell, (0.5, 0.0), False)
        v = Fraction(val[0]) + Fraction(val[1])
        slack = Fraction(cell.E + 1e-300) + (hi - lo)
        if not (lo - slack <= v <= hi + slack):
            raise _fail(
                "Fast-path cell distance disagrees with the certified "
                "oracle at the cell midpoint",
                index=j,
                quantity="|D_dd(1/2) - D_exact(1/2)|",
                value=float(abs(v - (lo + hi) / 2)),
                bound=f"<= {float(slack):.3e}",
                distance=d,
            )
        # Seed-table sanity (item 9).
        last = 0.0
        for _, s in cell.lut:
            if not (
                -cell.E <= s <= cell.length_float * (1.0 + 1e-9) + cell.E
            ):
                raise _fail(
                    "A seed node lies outside its cell", index=j, distance=d
                )
            if s < last - cell.E:
                raise _fail("Seed nodes are not monotone", index=j, distance=d)
            last = s


# ---------------------------------------------------------------------------
# Certified integer cell evaluation (fallback path)
# ---------------------------------------------------------------------------


def _certified_cell_distance(
    metric: "OffsetMetric",
    cell: _Cell,
    x: Fraction,
    prec: int,
) -> tuple[Fraction, Fraction]:
    """Exact/ball enclosure of the forward in-cell distance at ``x``.

    The source-length part is exact rational; only the angle term carries
    the ball width of the requested precision, so no floating-point
    cancellation can occur on this path.
    """
    span = metric._spans[cell.span]
    a, b = cell.a_loc, cell.b_loc
    whole = a == 0 and b == 1
    r = list(span.rho) if whole else bern_restrict(span.rho, a, b)
    anti = bern_antiderivative(r)
    ds = metric._H * span.h * (b - a) * bern_eval(anti, x)
    if not cell.angle:
        v = ds * cell.eta
        return v, v
    t = a + (b - a) * x
    wa_re = bern_eval(span.wre, a)
    wa_im = bern_eval(span.wim, a)
    wt_re = bern_eval(span.wre, t)
    wt_im = bern_eval(span.wim, t)
    x_arg = wt_re * wa_re + wt_im * wa_im
    y_arg = wt_im * wa_re - wt_re * wa_im
    if y_arg == 0 and x_arg > 0:
        v = ds * cell.eta
        return v, v
    v, e = atan2_ball(y_arg, x_arg, prec)
    th_lo, th_hi = ball_to_fraction_bounds(v, e, prec)
    two_d = 2 * Fraction(metric._d)
    c1 = cell.eta * (ds - two_d * th_lo)
    c2 = cell.eta * (ds - two_d * th_hi)
    return (c1, c2) if c1 <= c2 else (c2, c1)


# ---------------------------------------------------------------------------
# Ordered-float helpers (valid for nonnegative finite floats)
# ---------------------------------------------------------------------------


def _float_to_ordered(f: float) -> int:
    return unpack("<q", pack("<d", f))[0]


def _ordered_to_float(i: int) -> float:
    return unpack("<d", pack("<q", i))[0]


# ---------------------------------------------------------------------------
# The metric object
# ---------------------------------------------------------------------------


class OffsetMetric:
    """Complete verified distance certificate of one offset handle.

    Instances are produced only by :func:`build_offset_metric`.  All
    state is written once during construction; queries never mutate the
    object, so it is safe under concurrent readers, copying, and
    pickling (via :meth:`state` and :func:`rebuild_offset_metric`).
    """

    __slots__ = (
        "_bounds",
        "_cells",
        "_closed",
        "_cusps",
        "_d",
        "_d_hat",
        "_H",
        "_length_dd",
        "_length_float",
        "_prefix_dd",
        "_prefix_exact",
        "_prefix_float",
        "_scale",
        "_spans",
    )

    # -- public queries (validated scalars supplied by the handle) --------

    @property
    def length(self) -> float:
        return self._length_float

    @property
    def cusps(self) -> tuple[OffsetCusp, ...]:
        return self._cusps

    def arc_length(self, u: float) -> float:
        """Correctly rounded traversal distance from 0 to ``u``."""
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return self._length_float
        j = bisect_right(self._bounds, u) - 1
        if j >= len(self._cells):
            j = len(self._cells) - 1
        if u == self._bounds[j]:
            return self._prefix_float[j]
        val, err = self._eval_at(self._cells[j], j, u)
        lo = val[0] + (val[1] - err)
        hi = val[0] + (val[1] + err)
        if lo == hi:
            return lo
        return self._arc_length_certified(j, u)

    def parameter_at_length(self, s: float) -> float:
        """Unique parameter with ``arc_length(u) = s`` (spec section 10)."""
        if s <= 0.0:
            return 0.0
        if s >= self._length_float:
            return 1.0
        j = self._locate_prefix(s)
        cell = self._cells[j]
        p_lo = self._prefix_dd[j]
        p_hi = self._prefix_dd[j + 1]
        s1, e1 = two_sum(s, -p_lo[0])
        sf = dd_add((s1, e1), (-p_lo[1], 0.0))
        s2, e2 = two_sum(p_hi[0], -s)
        sr = dd_add((s2, e2), (p_hi[1], 0.0))
        if sf[0] + sf[1] <= 0.0:
            return cell.ga
        if sr[0] + sr[1] <= 0.0:
            return cell.gb
        return self._invert_in_cell(j, sf, sr, s)

    # -- internals ---------------------------------------------------------

    def _eval_at(self, cell: _Cell, j: int, u: float):
        """Double-double arc length at interior ``u`` with error bound."""
        ga, gb = cell.ga, cell.gb
        den = two_sum(gb, -ga)
        if u - ga <= gb - u:
            z = dd_div(two_sum(u, -ga), den)
            dval = _cell_distance_dd(cell, z, False)
            val = dd_add(self._prefix_dd[j], dval)
        else:
            z = dd_div(two_sum(gb, -u), den)
            dval = _cell_distance_dd(cell, z, True)
            val = dd_sub(self._prefix_dd[j + 1], dval)
        err = cell.E + 2.0**-100 * abs(val[0]) + 2.0 * _TINY
        return val, err

    def _arc_length_certified(self, j: int, u: float) -> float:
        """Adaptive certified evaluation ladder (spec 8.3)."""
        cell = self._cells[j]
        ga = Fraction(cell.ga)
        gb = Fraction(cell.gb)
        x = (Fraction(u) - ga) / (gb - ga)
        p_lo, p_hi = self._prefix_exact[j]
        for prec in _QUERY_LADDER:
            d_lo, d_hi = _certified_cell_distance(self, cell, x, prec)
            lo = float(p_lo + d_lo)
            hi = float(p_hi + d_hi)
            if lo == hi:
                return lo
        raise NumericalPrecisionError(
            "arc_length could not be certified within the adaptive "
            "precision cap",
            quantity="arc length enclosure",
            value=u,
            bound=f"{MAX_ADAPTIVE_PRECISION_BITS} bits",
        )

    def _locate_prefix(self, s: float) -> int:
        """Deterministic extended-prefix owner of a public target (10.1)."""
        j = bisect_right(self._prefix_float, s) - 1
        j = min(max(j, 0), len(self._cells) - 1)
        while j > 0:
            p = self._prefix_dd[j]
            if p[0] > s or (p[0] == s and p[1] > 0.0):
                j -= 1
            else:
                break
        while j < len(self._cells) - 1:
            p = self._prefix_dd[j + 1]
            if p[0] < s or (p[0] == s and p[1] <= 0.0):
                j += 1
            else:
                break
        return j

    def _speed_local(self, cell: _Cell, z: float, reverse: bool) -> float:
        """Seed derivative ``dD/dz`` (never authoritative)."""
        x = 1.0 - z if reverse else z
        g = _bern_eval_scaled(cell.g_scaled, cell.g_rev, x)
        sg = _bern_eval_scaled(cell.sig_scaled, cell.sig_rev, x)
        if not sg > 0.0:
            return 0.0
        return cell.kv * abs(g) / sg

    def _invert_in_cell(
        self,
        j: int,
        sf: tuple[float, float],
        sr: tuple[float, float],
        s_public: float,
    ) -> float:
        """Safeguarded bracketed Newton inverse in one cell (spec 10)."""
        cell = self._cells[j]
        reverse = sr[0] < sf[0]
        target = sr if reverse else sf
        y = target[0] + target[1]
        gate = cell.E + 4.0 * ulp(abs(y) + _TINY) + 32.0 * _EPS * abs(y)

        # Seed priority (spec 10.2): cusp law, seed table, proportional.
        cusp = cell.cusp1 if reverse else cell.cusp0
        z = -1.0
        if cusp is not None and cusp[1] > 0.0:
            r_m, c_r = cusp
            z = (y / c_r) ** (1.0 / (r_m + 1.0))
        if not 0.0 < z < 1.0:
            z = self._lut_seed(cell, y, reverse)
        if not 0.0 < z < 1.0:
            z = min(
                max(y / max(cell.length_float, _TINY), 2.0**-60),
                1.0 - 2.0**-60,
            )

        lo_b, hi_b = 0.0, 1.0
        accepted = None
        for _ in range(_MAX_NEWTON):
            val = _cell_distance_dd(cell, (z, 0.0), reverse)
            g = dd_add(val, (-target[0], -target[1]))
            g_f = g[0] + g[1]
            if abs(g_f) <= gate:
                accepted = z
                break
            if g_f < 0.0:
                lo_b = z
            else:
                hi_b = z
            v = self._speed_local(cell, z, reverse)
            z_new = z - g_f / v if v > 0.0 else math.nan
            if not (math.isfinite(z_new) and lo_b < z_new < hi_b):
                z_new = lo_b + 0.5 * (hi_b - lo_b)
            if z_new == z:
                z_new = lo_b + 0.5 * (hi_b - lo_b)
                if z_new == z:
                    break
            z = z_new
        if accepted is None:
            return self._ordered_search(j, s_public)

        u = self._z_to_u(cell, accepted, reverse)
        # Postcondition (spec 10.5): re-verify through the public kernel;
        # the extra ulp terms admit the parameter-rounding displacement.
        a_pub = self.arc_length(u)
        v_loc = self._speed_local(cell, accepted, reverse)
        width = cell.gb - cell.ga
        gate_pub = (
            gate
            + 32.0 * _EPS * abs(s_public)
            + 4.0 * v_loc / max(width, _TINY) * ulp(max(abs(u), _TINY))
        )
        if abs(a_pub - s_public) <= gate_pub:
            return u
        return self._ordered_search(j, s_public)

    def _lut_seed(self, cell: _Cell, y: float, reverse: bool) -> float:
        lut = cell.lut
        if not lut:
            return -1.0
        target = cell.length_float - y if reverse else y
        x_prev, s_prev = 0.0, 0.0
        x_seed = -1.0
        for x_node, s_node in lut + ((1.0, cell.length_float),):
            if target <= s_node:
                if s_node > s_prev:
                    x_seed = x_prev + (x_node - x_prev) * (
                        target - s_prev
                    ) / (s_node - s_prev)
                break
            x_prev, s_prev = x_node, s_node
        if not 0.0 < x_seed < 1.0:
            return -1.0
        return 1.0 - x_seed if reverse else x_seed

    def _z_to_u(self, cell: _Cell, z: float, reverse: bool) -> float:
        width = two_sum(cell.gb, -cell.ga)
        step = dd_mul_f(width, z)
        if reverse:
            val = dd_add((cell.gb, 0.0), (-step[0], -step[1]))
        else:
            val = dd_add((cell.ga, 0.0), step)
        u = val[0] + val[1]
        return min(max(u, cell.ga), cell.gb)

    def _ordered_search(self, j: int, s: float) -> float:
        """Best-representable-parameter search over public floats (10.4).

        Bisects the ordered binary64 encodings of the global parameter
        inside the owning cell.  Every comparison is either a decided
        double-double sign or a certified integer enclosure; an
        enclosure that cannot be decided within the adaptive cap raises
        the typed failure required by the specification.
        """
        cell = self._cells[j]
        lo_u, hi_u = cell.ga, cell.gb
        i_lo = _float_to_ordered(lo_u)
        i_hi = _float_to_ordered(hi_u)
        steps = 0
        while i_hi - i_lo > 1:
            steps += 1
            if steps > _MAX_ORDERED_SEARCH:
                raise ArcLengthInversionError(
                    "Ordered-float search exceeded its hard iteration "
                    "bound",
                    quantity="ordered search steps",
                    value=steps,
                    bound=f"<= {_MAX_ORDERED_SEARCH}",
                )
            u_mid = _ordered_to_float((i_lo + i_hi) // 2)
            cmp = self._compare_distance(j, u_mid, s)
            if cmp == 0:
                return u_mid
            if cmp < 0:
                i_lo = _float_to_ordered(u_mid)
                lo_u = u_mid
            else:
                i_hi = _float_to_ordered(u_mid)
                hi_u = u_mid
        # Adjacent representable parameters bracket the target: return the
        # one with the certifiably smaller residual; break exact ties
        # toward the smaller parameter.
        r_lo = self._residual_bounds(j, lo_u, s)
        r_hi = self._residual_bounds(j, hi_u, s)
        if r_lo[1] <= r_hi[0]:
            return lo_u
        if r_hi[1] < r_lo[0]:
            return hi_u
        sfr = Fraction(s)
        for prec in _QUERY_LADDER:
            a_lo = self._certified_bounds(j, lo_u, prec)
            a_hi = self._certified_bounds(j, hi_u, prec)
            lo_res = _abs_interval(a_lo[0] - sfr, a_lo[1] - sfr)
            hi_res = _abs_interval(a_hi[0] - sfr, a_hi[1] - sfr)
            if lo_res[1] <= hi_res[0]:
                return lo_u
            if hi_res[1] < lo_res[0]:
                return hi_u
        raise NumericalPrecisionError(
            "Adjacent-parameter comparison could not be certified within "
            "the adaptive precision cap",
            quantity="distance comparison",
            value=(lo_u, hi_u),
            bound=f"{MAX_ADAPTIVE_PRECISION_BITS} bits",
        )

    def _compare_distance(self, j: int, u: float, s: float) -> int:
        """Certified sign of ``arc_length(u) - s``; 0 = acceptably equal."""
        cell = self._cells[j]
        val, err = self._eval_at(cell, j, u)
        diff = dd_add(val, (-s, 0.0))
        d_f = diff[0] + diff[1]
        if abs(d_f) > err:
            return -1 if d_f < 0.0 else 1
        sfr = Fraction(s)
        # Acceptance-by-equality requires the certified residual to pass
        # the public residual gate of spec 10.4 (with the certified path's
        # evaluation error equal to the enclosure width itself).
        gate = Fraction(4.0 * ulp(abs(s) + _TINY) + 32.0 * _EPS * abs(s))
        for prec in _QUERY_LADDER:
            lo, hi = self._certified_bounds(j, u, prec)
            if hi < sfr:
                return -1
            if lo > sfr:
                return 1
            if hi - lo <= gate:
                # The distance provably meets the residual gate: accept
                # the parameter with a certified residual certificate.
                return 0
        raise NumericalPrecisionError(
            "A distance comparison could not be certified within the "
            "adaptive precision cap",
            quantity="distance comparison enclosure",
            value=u,
            bound=f"{MAX_ADAPTIVE_PRECISION_BITS} bits",
        )

    def _residual_bounds(self, j: int, u: float, s: float) -> tuple[float, float]:
        val, err = self._eval_at(self._cells[j], j, u)
        diff = dd_add(val, (-s, 0.0))
        d_f = abs(diff[0] + diff[1])
        return (max(d_f - err, 0.0), d_f + err)

    def _certified_bounds(
        self, j: int, u: float, prec: int
    ) -> tuple[Fraction, Fraction]:
        cell = self._cells[j]
        ga = Fraction(cell.ga)
        gb = Fraction(cell.gb)
        x = (Fraction(u) - ga) / (gb - ga)
        p_lo, p_hi = self._prefix_exact[j]
        d_lo, d_hi = _certified_cell_distance(self, cell, x, prec)
        return p_lo + d_lo, p_hi + d_hi

    # -- read-only views for the area addendum ---------------------------
    #
    # ``ClosedSpline_Area_Specification.md`` (section 12.2) requires the
    # metric to expose its exact phase cells and exact source-speed state
    # without duplicating any algorithm.  Both views return captured
    # immutable construction data; they never compute anything.

    def exact_source_state(self):
        """Exact captured source state ``(spans, H, d, closed)``.

        ``spans`` is the ordered tuple of per-span exact certificates
        (rational preimages ``wre``/``wim``, speed coefficients ``rho``,
        width ``h`` and preimage degree ``m``), ``H`` the exact rational
        normalization scale, ``d`` the accepted signed offset distance and
        ``closed`` the verified source topology.
        """
        return tuple(self._spans), self._H, self._d, self._closed

    def fill_cells(self):
        """All verified metric cells as ``(span, a_loc, b_loc, eta)``.

        ``eta`` is the certified constant sign of the cusp polynomial
        ``G`` on the cell, which is also the sign factor between the
        offset tangent direction and ``w**2``.  Captured construction
        data only; nothing is computed.
        """
        return tuple(
            (cell.span, cell.a_loc, cell.b_loc, cell.eta)
            for cell in self._cells
        )

    def phase_cells(self):
        """Verified nonconstant-phase cells as ``(span, a_loc, b_loc)``.

        Each triple names one half-plane-certified metric cell by its span
        index and exact rational local bounds; the continuous preimage
        phase change across such a cell is provably below ``pi/2`` in
        magnitude, so a principal ``atan2`` of the exact endpoint
        preimages equals the continuous increment.  Constant-phase spans
        contribute no cells.
        """
        return tuple(
            (cell.span, cell.a_loc, cell.b_loc)
            for cell in self._cells
            if cell.angle
        )

    # -- serialization support -------------------------------------------

    def state(self) -> dict:
        """Raw immutable source state sufficient to rebuild the metric."""
        return {
            "preimages": [
                [(float(re), float(im)) for re, im in zip(s.wre, s.wim)]
                for s in self._spans
            ],
            "widths": [float(s.h) for s in self._spans],
            "breakpoints": [self._spans[0].u0] + [s.u1 for s in self._spans],
            "distance": self._d,
            "scale": self._scale,
            "closed": self._closed,
        }


def _abs_interval(lo: Fraction, hi: Fraction) -> tuple[Fraction, Fraction]:
    """Interval absolute value."""
    if lo >= 0:
        return lo, hi
    if hi <= 0:
        return -hi, -lo
    return _ZERO, max(-lo, hi)
