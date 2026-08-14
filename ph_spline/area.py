"""Closed-spline signed-area kernels (``ClosedSpline_Area_Specification.md``).

This module owns every area algorithm of the specification:

- exact rational pairwise coefficient tables ``K_ab`` (spec 5.2), generated
  from integer binomial coefficients and cached per degree on first use;
- the certified double-double *ball* fast path for the composite Bernstein
  area sum (spec 5.4 / 10.3), with error-free ``two_prod`` determinants and
  per-operation propagated radii;
- the exact-rational fallback (spec 10.4) whose ``Fraction`` result is
  converted once with correct rounding, including exact-zero detection and
  explicit overflow rejection;
- the residual join-closure convention for stored chains (spec 4.4);
- exact source-length extraction from the captured offset-metric state
  (spec 8.4) and the certified integer tangent turning number from the
  metric's verified preimage phase cells (spec 9);
- the adaptive certified evaluation of ``A_d = R + C * pi`` (spec 10.5); and
- the immutable :class:`OffsetAreaProvenance` record carried by closed
  offset handles (spec 8.5).

Topology-specific properties, caches and provenance wiring live in
``cubic.py``, ``bspline.py`` and ``nurbs.py``; no spline class is imported
here.  All arithmetic runs on normalized position controls; the user origin
never occurs (spec 5.5 / 10.2).
"""

from __future__ import annotations

import math
from fractions import Fraction
from math import comb

import numpy as np
from numpy.typing import NDArray

from ph_spline.ddouble import dd_add, dd_mul, two_prod
from ph_spline.exact_real import (
    atan2_ball,
    ball_to_fraction_bounds,
    bern_eval,
    pi_ball,
)
from ph_spline.exceptions import NumericalPrecisionError, OffsetConstructionError

__all__ = [
    "AREA_PRECISION_LADDER",
    "OffsetAreaProvenance",
    "area_coefficients",
    "offset_signed_area",
    "reset_statistics",
    "source_signed_area",
    "span_contribution_ball",
    "statistics",
    "turning_number",
]

#: Adaptive certified precision ladder shared by the turning-number and the
#: ``R + C * pi`` evaluations (spec 9.3 / 10.5).
AREA_PRECISION_LADDER = (256, 512, 1024, 2048, 4096)

#: Unit roundoff of one double-double operation (spec 10.3, ``eps_dd``).
_EPS_DD = 2.0**-104

#: Two-prod products smaller than this may carry a subnormal residual that
#: the FMA cannot represent exactly; a few ulps of absolute slack cover it.
_SUBNORMAL_GUARD = 1.0e-290
_SUBNORMAL_SLACK = 4.0e-323

_ZERO = Fraction(0)
_MAX_BINARY64 = math.nextafter(math.inf, 0.0)

#: Lazy per-degree coefficient tables (exact rational and double-double
#: enclosure forms).  Publication is a single atomic dict assignment; a
#: duplicated first computation is deterministic and harmless (spec 11.2).
_FRACTION_TABLES: dict[int, tuple[tuple[int, int, Fraction], ...]] = {}
_BALL_TABLES: dict[int, tuple[tuple[int, int, float, float, float], ...]] = {}

#: Query diagnostics for tests and benchmarks (spec 16).  Counters only;
#: never consulted by any production decision.
statistics = {
    "fast_accepted": 0,
    "exact_fallback": 0,
    "span_contributions": 0,
    "span_reused": 0,
    "turning_max_precision": 0,
    "last_condition": 0.0,
}


def reset_statistics() -> None:
    """Reset all diagnostic counters to their initial state."""
    statistics.update(
        fast_accepted=0,
        exact_fallback=0,
        span_contributions=0,
        span_reused=0,
        turning_max_precision=0,
        last_condition=0.0,
    )


def _area_error(message: str, **fields) -> NumericalPrecisionError:
    return NumericalPrecisionError(message, operation="area", **fields)


# ---------------------------------------------------------------------------
# Exact pairwise coefficient tables (spec 5.2 / 13.1)
# ---------------------------------------------------------------------------


def area_coefficients(p: int) -> tuple[tuple[int, int, Fraction], ...]:
    """Exact strictly positive coefficients ``(a, b, K_ab)`` for degree ``p``.

    ``K_ab = (b - a) C(p, a) C(p, b) / (2 (2p - 1) C(2p - 2, a + b - 1))``
    from exact integer binomial coefficients (spec 5.2).  The table depends
    only on the degree and is cached on first area use.
    """
    table = _FRACTION_TABLES.get(p)
    if table is not None:
        return table
    if p < 1:
        raise _area_error(
            "Area coefficient tables require degree >= 1",
            quantity="degree",
            value=p,
            bound=">= 1",
        )
    rows: list[tuple[int, int, Fraction]] = []
    for a in range(p + 1):
        for b in range(a + 1, p + 1):
            coefficient = Fraction(
                (b - a) * comb(p, a) * comb(p, b),
                2 * (2 * p - 1) * comb(2 * p - 2, a + b - 1),
            )
            if not coefficient > 0:
                raise _area_error(
                    "A generated area coefficient is not strictly positive",
                    quantity=f"K[{a},{b}]",
                    value=float(coefficient),
                    bound="> 0",
                )
            rows.append((a, b, coefficient))
    table = tuple(rows)
    _FRACTION_TABLES[p] = table
    return table


def _ball_table(p: int) -> tuple[tuple[int, int, float, float, float], ...]:
    """Double-double enclosures ``(a, b, k_hi, k_lo, radius)`` of the exact
    coefficients (spec 10.3: never one unchecked binary64 coefficient)."""
    table = _BALL_TABLES.get(p)
    if table is not None:
        return table
    rows: list[tuple[int, int, float, float, float]] = []
    for a, b, coefficient in area_coefficients(p):
        hi = float(coefficient)
        lo = float(coefficient - Fraction(hi))
        remainder = abs(coefficient - Fraction(hi) - Fraction(lo))
        radius = 0.0 if remainder == 0 else math.nextafter(float(remainder), math.inf)
        rows.append((a, b, hi, lo, radius))
    table = tuple(rows)
    _BALL_TABLES[p] = table
    return table


# ---------------------------------------------------------------------------
# Double-double ball arithmetic (spec 10.3)
# ---------------------------------------------------------------------------
#
# A ball is ``(hi, lo, r)``: the exact value lies within ``r`` of the exact
# real ``hi + lo``.  Radii follow the conservative recurrences (10.1); every
# radius result gets one extra upward slop step so nearest rounding of the
# bound arithmetic itself can never understate it.


def _bound_up(value: float) -> float:
    return math.nextafter(value * (1.0 + 2.0**-40), math.inf)


def _ball_add(a, b):
    ah, al, ar = a
    bh, bl, br = b
    hi, lo = dd_add((ah, al), (bh, bl))
    magnitude = abs(ah) + abs(al) + abs(bh) + abs(bl)
    return (hi, lo, _bound_up(ar + br + 4.0 * _EPS_DD * magnitude))


def _ball_mul(a, b):
    ah, al, ar = a
    bh, bl, br = b
    hi, lo = dd_mul((ah, al), (bh, bl))
    ma = abs(ah) + abs(al)
    mb = abs(bh) + abs(bl)
    radius = (
        ma * br
        + mb * ar
        + ar * br
        + 8.0 * _EPS_DD * (ma + ar) * (mb + br)
    )
    return (hi, lo, _bound_up(radius))


def _det_ball(ax: float, ay: float, bx: float, by: float):
    """Certified ball of the determinant ``ax * by - ay * bx``.

    Both products are error-free ``two_prod`` transforms; the double-double
    subtraction contributes ``4 eps_dd`` relative error, and products deep
    in the subnormal range receive a small absolute slack because the FMA
    residual is then no longer exact.
    """
    p1, e1 = two_prod(ax, by)
    p2, e2 = two_prod(ay, bx)
    hi, lo = dd_add((p1, e1), (-p2, -e2))
    radius = 4.0 * _EPS_DD * (abs(p1) + abs(p2))
    if ax != 0.0 and by != 0.0 and abs(p1) < _SUBNORMAL_GUARD:
        radius += _SUBNORMAL_SLACK
    if ay != 0.0 and bx != 0.0 and abs(p2) < _SUBNORMAL_GUARD:
        radius += _SUBNORMAL_SLACK
    return (hi, lo, _bound_up(radius))


def _ball_finite(ball) -> bool:
    return (
        math.isfinite(ball[0])
        and math.isfinite(ball[1])
        and math.isfinite(ball[2])
    )


def _pairwise_sum(balls):
    """Balanced pairwise ball reduction (magnitude-growth aware)."""
    items = list(balls)
    if not items:
        return (0.0, 0.0, 0.0)
    while len(items) > 1:
        merged = [
            _ball_add(items[i], items[i + 1])
            for i in range(0, len(items) - 1, 2)
        ]
        if len(items) % 2:
            merged.append(items[-1])
        items = merged
    return items[0]


# ---------------------------------------------------------------------------
# Certified fast composite path (spec 5.4 / 10.3)
# ---------------------------------------------------------------------------


def span_contribution_ball(controls: NDArray[np.float64]):
    """Certified ball of one normalized span's contribution (5.3).

    ``controls`` is the read-only ``(p + 1, 2)`` normalized position array.
    Returns ``None`` when a nonfinite intermediate appears; the caller then
    uses the exact fallback (spec 10.3: no empirical rescaling).
    """
    statistics["span_contributions"] += 1
    p = controls.shape[0] - 1
    terms = []
    condition = 0.0
    for a, b, k_hi, k_lo, k_r in _ball_table(p):
        det = _det_ball(
            float(controls[a, 0]),
            float(controls[a, 1]),
            float(controls[b, 0]),
            float(controls[b, 1]),
        )
        term = _ball_mul((k_hi, k_lo, k_r), det)
        if not _ball_finite(term):
            return None
        condition += abs(term[0])
        terms.append(term)
    total = _pairwise_sum(terms)
    statistics["last_condition"] = condition
    return total if _ball_finite(total) else None


def _join_balls(spans):
    """Residual connector line-integral balls, cyclically (spec 4.4)."""
    half = (0.5, 0.0, 0.0)
    count = len(spans)
    balls = []
    for index in range(count):
        end = spans[index][-1]
        start = spans[(index + 1) % count][0]
        det = _det_ball(
            float(end[0]), float(end[1]), float(start[0]), float(start[1])
        )
        balls.append(_ball_mul(half, det))
    return balls


def _accept_ball(ball) -> float | None:
    """Publish the ball's value iff its enclosure fixes one binary64.

    The double-double center is decoded exactly, the radius is expanded
    outward by one ``nextafter`` step, and both exact interval endpoints
    are correctly rounded; monotonicity of correct rounding then makes an
    equal pair the correctly rounded value of every real in the enclosure.
    """
    if not _ball_finite(ball):
        return None
    center = Fraction(ball[0]) + Fraction(ball[1])
    radius = Fraction(math.nextafter(ball[2], math.inf))
    try:
        lo = float(center - radius)
        hi = float(center + radius)
    except OverflowError:
        return None
    if lo == hi and math.isfinite(lo):
        return lo
    return None


def _aggregate_fast(spans, span_balls, scale: float) -> float | None:
    """Fast certified composite (5.4)-(5.5); ``None`` demands the fallback."""
    total = _pairwise_sum(list(span_balls) + _join_balls(spans))
    if not _ball_finite(total):
        return None
    # H * (H * A_hat): never H * H first (spec 5.5).
    h_ball = (scale, 0.0, 0.0)
    total = _ball_mul(_ball_mul(total, h_ball), h_ball)
    return _accept_ball(total)


# ---------------------------------------------------------------------------
# Exact rational fallback (spec 10.4)
# ---------------------------------------------------------------------------


def _normalized_area_exact(spans) -> Fraction:
    """Exact rational normalized composite area (5.4), joins included."""
    total = _ZERO
    for controls in spans:
        p = len(controls) - 1
        xs = [Fraction(float(row[0])) for row in controls]
        ys = [Fraction(float(row[1])) for row in controls]
        for a, b, coefficient in area_coefficients(p):
            total += coefficient * (xs[a] * ys[b] - ys[a] * xs[b])
    count = len(spans)
    for index in range(count):
        end = spans[index][-1]
        start = spans[(index + 1) % count][0]
        ex, ey = Fraction(float(end[0])), Fraction(float(end[1]))
        qx, qy = Fraction(float(start[0])), Fraction(float(start[1]))
        total += (ex * qy - ey * qx) / 2
    return total


def _publish_float(value: Fraction) -> float:
    """One correctly rounded rational-to-binary64 conversion (spec 10.4).

    Exact zero publishes as positive zero; a nonzero value that underflows
    keeps its IEEE 754 sign.  A value beyond binary64 range raises the
    typed failure instead of returning infinity (spec 10.1).
    """
    if value == 0:
        return 0.0
    try:
        result = float(value)
    except OverflowError:
        result = math.inf if value > 0 else -math.inf
    if not math.isfinite(result):
        raise _area_error(
            "The exact signed area is outside the finite binary64 range",
            quantity="signed area",
            bound=f"|A| <= {_MAX_BINARY64!r}",
        )
    return result


# ---------------------------------------------------------------------------
# Public source-area kernel
# ---------------------------------------------------------------------------


def _validate_source(spans, scale: float) -> None:
    """Structural wiring invariants (spec 14.1); computes no area."""
    if len(spans) < 1:
        raise _area_error(
            "A closed area query requires at least one source span",
            quantity="span count",
            value=len(spans),
            bound=">= 1",
        )
    if not (math.isfinite(scale) and scale > 0.0):
        raise _area_error(
            "The normalization scale must be finite and positive",
            quantity="H",
            value=scale,
            bound="> 0",
        )
    for index, controls in enumerate(spans):
        shape = getattr(controls, "shape", None)
        if shape is None or len(shape) != 2 or shape[1] != 2 or shape[0] < 2:
            raise _area_error(
                "A source position span violates its (p + 1, 2) contract",
                span_id=index,
                quantity="position shape",
                value=shape,
            )
        if not np.all(np.isfinite(controls)):
            raise _area_error(
                "A source position span contains a nonfinite coefficient",
                span_id=index,
                quantity="position controls",
            )


def source_signed_area(spans, scale: float, *, span_balls=None) -> float:
    """Correctly rounded signed area of one closed normalized chain.

    ``spans`` is the ordered sequence of read-only ``(p + 1, 2)`` normalized
    position-control arrays of one committed closed state and ``scale`` its
    normalization ``H``.  ``span_balls`` optionally supplies precomputed
    certified span contributions (PH B-spline local-edit reuse, spec 7.3);
    a ``None`` entry forces the exact path.

    The certified double-double fast path is attempted first; whenever its
    enclosure does not fix one binary64 value, the exact rational fallback
    decides the result (spec 10.3 / 10.4).
    """
    _validate_source(spans, scale)
    if span_balls is None:
        span_balls = [span_contribution_ball(controls) for controls in spans]
    if all(ball is not None for ball in span_balls):
        fast = _aggregate_fast(spans, span_balls, scale)
        if fast is not None:
            statistics["fast_accepted"] += 1
            return fast
    statistics["exact_fallback"] += 1
    exact = Fraction(scale) * Fraction(scale) * _normalized_area_exact(spans)
    return _publish_float(exact)


# ---------------------------------------------------------------------------
# Certified tangent turning number (spec 9)
# ---------------------------------------------------------------------------


def _complex_square(re: Fraction, im: Fraction) -> tuple[Fraction, Fraction]:
    return (re * re - im * im, 2 * re * im)


def turning_number(metric) -> int:
    """Certified integer tangent turning number of the captured source.

    Reuses the verified constant-phase metric cells: each nonconstant-phase
    cell contributes twice the principal preimage rotation between its exact
    rational endpoint preimages (9.2), and every cyclic squared-tangent join
    contributes its canonical shortest correction (9.1).  The quotient by
    the simultaneous ``pi`` enclosure must isolate exactly one integer on
    the adaptive precision ladder (9.3).
    """
    spans, _, _, closed = metric.exact_source_state()
    if not closed:
        raise _area_error(
            "A turning number is defined only for a closed source",
            quantity="topology",
        )
    arguments: list[tuple[Fraction, Fraction, int]] = []
    for span_index, a_loc, b_loc in metric.phase_cells():
        span = spans[span_index]
        wa_re = bern_eval(span.wre, a_loc)
        wa_im = bern_eval(span.wim, a_loc)
        wb_re = bern_eval(span.wre, b_loc)
        wb_im = bern_eval(span.wim, b_loc)
        x_arg = wb_re * wa_re + wb_im * wa_im
        y_arg = wb_im * wa_re - wb_re * wa_im
        if x_arg == 0 and y_arg == 0:
            raise _area_error(
                "A phase cell endpoint preimage vanished",
                span_id=span_index,
                quantity="(X, Y)",
                bound="!= (0, 0)",
            )
        arguments.append((y_arg, x_arg, 2))
    count = len(spans)
    for index in range(count):
        left = spans[index]
        right = spans[(index + 1) % count]
        ql = _complex_square(left.wre[-1], left.wim[-1])
        qr = _complex_square(right.wre[0], right.wim[0])
        x_arg = ql[0] * qr[0] + ql[1] * qr[1]
        y_arg = ql[0] * qr[1] - ql[1] * qr[0]
        if not x_arg > 0:
            raise _area_error(
                "A cyclic squared-tangent join contradicts the verified G1 "
                "continuity contract",
                index=index,
                quantity="q- . q+",
                value=float(x_arg),
                bound="> 0",
            )
        arguments.append((y_arg, x_arg, 1))
    for precision in AREA_PRECISION_LADDER:
        v_theta = 0
        e_theta = 0
        for y_arg, x_arg, weight in arguments:
            value, error = atan2_ball(y_arg, x_arg, precision)
            v_theta += weight * value
            e_theta += weight * error
        v_pi, e_pi = pi_ball(precision)
        numerator = (v_theta - e_theta, v_theta + e_theta)
        denominator = (2 * (v_pi - e_pi), 2 * (v_pi + e_pi))
        quotients = [
            Fraction(n, d) for n in numerator for d in denominator
        ]
        low = math.ceil(min(quotients))
        high = math.floor(max(quotients))
        if low == high:
            statistics["turning_max_precision"] = max(
                statistics["turning_max_precision"], precision
            )
            return low
    raise _area_error(
        "The turning-number enclosure could not isolate one integer within "
        "the adaptive precision cap",
        quantity="turning number enclosure",
        bound=f"{AREA_PRECISION_LADDER[-1]} bits",
    )


# ---------------------------------------------------------------------------
# Closed offset area (spec 8 / 10.5)
# ---------------------------------------------------------------------------


class OffsetAreaProvenance:
    """Immutable captured source state of one closed offset (spec 8.5).

    Owns read-only snapshots of the ordered normalized source position
    controls and shares the exact-reference preimages, widths, scale,
    distance and closed topology with the handle's verified offset-metric
    certificate.  Construction captures and validates only; it never
    evaluates an area, a length, or a turning number.
    """

    __slots__ = ("metric", "position_spans")

    def __init__(self, *, position_spans, metric) -> None:
        captured = []
        for array in position_spans:
            copy = np.array(array, dtype=np.float64, copy=True)
            copy.setflags(write=False)
            captured.append(copy)
        self.position_spans = tuple(captured)
        self.metric = metric
        self._validate()

    def _validate(self) -> None:
        """Structural provenance invariants (spec 14.1); no area work."""

        def fail(message: str, **fields) -> OffsetConstructionError:
            return OffsetConstructionError(
                message, operation="offset", quantity="area provenance", **fields
            )

        if self.metric is None:
            raise fail("Closed offset area provenance lacks its metric")
        spans, scale, distance, closed = self.metric.exact_source_state()
        if not closed:
            raise fail("Area provenance requires a closed source topology")
        if not math.isfinite(distance):
            raise fail("Area provenance distance is not finite", value=distance)
        if not scale > 0:
            raise fail("Area provenance scale is not positive")
        if not (len(self.position_spans) >= 1
                and len(self.position_spans) == len(spans)):
            raise fail(
                "Position and preimage span counts disagree",
                value=(len(self.position_spans), len(spans)),
            )
        for index, (controls, span) in enumerate(
            zip(self.position_spans, spans)
        ):
            if (
                controls.ndim != 2
                or controls.shape[1] != 2
                or controls.shape[0] != 2 * span.m + 2
            ):
                raise fail(
                    "Provenance position degree disagrees with the "
                    "preimage degree",
                    span_id=index,
                    value=controls.shape,
                    bound=f"({2 * span.m + 2}, 2)",
                )
            if not np.all(np.isfinite(controls)):
                raise fail(
                    "Provenance position controls are not finite",
                    span_id=index,
                )
            if not span.h > 0:
                raise fail(
                    "Provenance span width is not positive", span_id=index
                )


def _exact_source_length(spans, scale: Fraction) -> Fraction:
    """Exact source length (8.3) from the captured metric coefficients."""
    total = _ZERO
    for span in spans:
        total += span.h * sum(span.rho, _ZERO) / (2 * span.m + 1)
    return scale * total


def offset_signed_area(provenance: OffsetAreaProvenance) -> float:
    """Correctly rounded signed area of one exact closed PH offset.

    Implements the query procedure of spec 8.6: exact source area from the
    captured position controls, exact source length from the metric
    coefficients, the certified integer turning number, and the adaptive
    certified evaluation of ``A_d = R + C * pi`` (10.5).
    """
    metric = provenance.metric
    spans, scale, distance, _ = metric.exact_source_state()
    area_source = (
        scale * scale * _normalized_area_exact(provenance.position_spans)
    )
    if distance == 0.0:
        return _publish_float(area_source)
    length_source = _exact_source_length(spans, scale)
    nu = turning_number(metric)
    d_exact = Fraction(distance)
    residue = area_source - d_exact * length_source
    pi_factor = nu * d_exact * d_exact
    if pi_factor == 0:
        return _publish_float(residue)
    for precision in AREA_PRECISION_LADDER:
        v_pi, e_pi = pi_ball(precision)
        pi_lo, pi_hi = ball_to_fraction_bounds(v_pi, e_pi, precision)
        if pi_factor > 0:
            bounds = (residue + pi_factor * pi_lo, residue + pi_factor * pi_hi)
        else:
            bounds = (residue + pi_factor * pi_hi, residue + pi_factor * pi_lo)
        try:
            lo = float(bounds[0])
            hi = float(bounds[1])
        except OverflowError:
            raise _area_error(
                "The exact offset signed area is outside the finite "
                "binary64 range",
                quantity="signed area",
                bound=f"|A| <= {_MAX_BINARY64!r}",
            ) from None
        if lo == hi:
            if not math.isfinite(lo):
                raise _area_error(
                    "The exact offset signed area is outside the finite "
                    "binary64 range",
                    quantity="signed area",
                    bound=f"|A| <= {_MAX_BINARY64!r}",
                )
            return lo
    raise _area_error(
        "The offset area enclosure could not determine one binary64 "
        "rounding within the adaptive precision cap",
        quantity="R + C * pi enclosure",
        bound=f"{AREA_PRECISION_LADDER[-1]} bits",
    )
