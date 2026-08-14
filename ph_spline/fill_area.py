"""Nonzero-winding fill area kernels (``ClosedSpline_FillArea_Specification.md``).

This module owns the complete certified fill-area pipeline:

- certified monotone pieces of the exact-reference locus (source spans,
  or offset metric cells carrying the constant sign of the cusp
  polynomial), each provably injective through its tangent-hull
  direction certificate;
- complete certified self-crossing enumeration over piece pairs: exact
  rational position-hull exclusion, cone-separation uniqueness (a common
  chord of two crossings would lie in both tangent cones), separable
  Miranda existence through univariate Bernstein hull comparisons, exact
  bisection localization with deterministic jittered splits, shared-joint
  corner certificates, and sound micro-block absorption of unresolved
  contacts (perturbation argument: replacing the curve inside a box of
  diameter ``delta`` changes the fill by at most the box area);
- Seifert smoothing of the parameter circle at the certified crossings
  into simple sub-loops, rational-interval Green integrals per sub-loop
  (exact Bernstein antiderivatives; for offsets the four-term identity
  with exact arc-length antiderivatives and certified phase ``atan2``
  enclosures), laminar containment via certified ray casting, and the
  winding-gated face-area sum of specification equation (3.1);
- the correct-rounding acceptance gate with the absorbed-slack budget.

Everything runs on exact rationals or rational intervals in the
normalized frame; the user origin never occurs and the ``H**2`` scaling
is applied once at publication.  No spline class is imported here.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ph_spline.exact_real import (
    atan2_ball,
    ball_to_fraction_bounds,
    bern_antiderivative,
    bern_elevate,
    bern_eval,
    bern_product,
    bern_restrict,
    isolate_bernstein_roots,
    sign_at,
)
from ph_spline.exceptions import NumericalPrecisionError

__all__ = [
    "fill_area_offset",
    "fill_area_source",
    "reset_statistics",
    "statistics",
]

_ZERO = Fraction(0)
_ONE = Fraction(1)
_HALF = Fraction(1, 2)

#: Hard resource bounds of the reference binary64 profile.  Block
#: recursions that converge on one unresolved point feature form chains,
#: not trees, so the depth cap can safely exceed the micro-absorption
#: scale (about 60 combined halvings) by a margin.
_MAX_PIECE_DEPTH = 26
_MAX_BLOCK_DEPTH = 160
_MAX_REFINE_STEPS = 420
_MAX_TEST_POINTS = 16
_MAX_ROOT_STEPS = 130

#: Micro-block qualification: absorbed boxes must be smaller than the
#: locus diameter by this factor; the summed quadratic slack then stays
#: near ``2**-68`` relative, far below one ulp of any fill value built
#: from the same geometry.
_MICRO_FACTOR = Fraction(1, 2**34)

#: Crossing enclosures are refined toward this parameter width.
_REFINE_WIDTH = Fraction(1, 2**100)

#: Deterministic jittered split fractions (a crossing cannot lie on all).
_SPLITS = (_HALF, Fraction(9, 16), Fraction(7, 16), Fraction(17, 32),
           Fraction(15, 32))

#: Phase-enclosure precisions (bits): first pass, then one retry.
_PHASE_PRECISIONS = (256, 512)

statistics = {
    "queries": 0,
    "simple_fast_path": 0,
    "crossings_certified": 0,
    "loops_decomposed": 0,
    "micro_blocks_absorbed": 0,
    "containment_tests": 0,
}


def reset_statistics() -> None:
    statistics.update(
        queries=0,
        simple_fast_path=0,
        crossings_certified=0,
        loops_decomposed=0,
        micro_blocks_absorbed=0,
        containment_tests=0,
    )


def _fail(message: str, **fields) -> NumericalPrecisionError:
    return NumericalPrecisionError(message, operation="fill-area", **fields)


# ---------------------------------------------------------------------------
# Rational intervals
# ---------------------------------------------------------------------------


def _iv(lo: Fraction, hi: Fraction):
    return (lo, hi) if lo <= hi else (hi, lo)


def _iv_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _iv_sub(a, b):
    return (a[0] - b[1], a[1] - b[0])


def _iv_mul(a, b):
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(products), max(products))


def _iv_div(a, b):
    if b[0] <= 0 <= b[1]:
        raise _fail("Interval division by an interval containing zero")
    quotients = (a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1])
    return (min(quotients), max(quotients))


def _iv_scale(a, factor: Fraction):
    return _iv(a[0] * factor, a[1] * factor)


def _iv_det(ax, ay, bx, by):
    return _iv_sub(_iv_mul(ax, by), _iv_mul(ay, bx))


def _bern_range(coeffs, lo: Fraction, hi: Fraction):
    """Certified range of a Bernstein polynomial over ``[lo, hi]``."""
    if lo == hi:
        value = bern_eval(coeffs, lo)
        return (value, value)
    restricted = (
        coeffs if (lo, hi) == (0, 1) else bern_restrict(coeffs, lo, hi)
    )
    return (min(restricted), max(restricted))


# ---------------------------------------------------------------------------
# Locus model
# ---------------------------------------------------------------------------


class _Span:
    """Exact per-span geometry of one closed locus in normalized frame."""

    __slots__ = (
        "cells", "fnum_x", "fnum_y", "fweight", "green", "h", "num_x",
        "num_y", "px", "py", "rho", "rho_anti", "rho_elev", "wre", "wim",
    )


class _Piece:
    """One certified monotone piece (injective graph over ``direction``)."""

    __slots__ = ("a", "b", "box", "eta", "span")


class _Locus:
    __slots__ = (
        "d_hat", "micro_limit", "offset", "pieces", "scale", "spans",
        "trim_slack",
    )


def _complex_square(re, im):
    """Bernstein coefficients of ``(re + i im)**2``."""
    rr = bern_product(re, re)
    ii = bern_product(im, im)
    ri = bern_product(re, im)
    return [a - b for a, b in zip(rr, ii)], [2 * a for a in ri]


def build_locus(position_spans, preimages, widths, scale, d_hat, cells):
    """Compile the exact locus: spans, offset numerators, monotone pieces.

    ``cells[i]`` lists ``(a_loc, b_loc, eta)`` triples covering span ``i``
    in order (sources pass one full cell with ``eta = +1``).
    """
    locus = _Locus()
    locus.scale = Fraction(float(scale))
    locus.d_hat = d_hat
    locus.offset = d_hat != 0
    locus.spans = []
    for index, controls in enumerate(position_spans):
        span = _Span()
        span.px = [Fraction(float(row[0])) for row in controls]
        span.py = [Fraction(float(row[1])) for row in controls]
        preimage = preimages[index]
        span.wre = [Fraction(float(complex(z).real)) for z in preimage]
        span.wim = [Fraction(float(complex(z).imag)) for z in preimage]
        span.h = Fraction(float(widths[index]))
        span.cells = [
            (Fraction(a), Fraction(b), int(eta)) for a, b, eta in cells[index]
        ]
        degree = len(span.px) - 1
        dpx = [degree * (b - a) for a, b in zip(span.px, span.px[1:])]
        dpy = [degree * (b - a) for a, b in zip(span.py, span.py[1:])]
        integrand = [
            u - v
            for u, v in zip(
                bern_product(span.px, dpy), bern_product(span.py, dpx)
            )
        ]
        span.green = bern_antiderivative(integrand)
        rr = bern_product(span.wre, span.wre)
        ii = bern_product(span.wim, span.wim)
        span.rho = [a + b for a, b in zip(rr, ii)]
        span.rho_anti = bern_antiderivative(span.rho)
        if locus.offset:
            w2re, w2im = _complex_square(span.wre, span.wim)
            target = degree + len(span.rho) - 1
            lift = target - (len(w2re) - 1)
            w2re = bern_elevate(w2re, lift)
            w2im = bern_elevate(w2im, lift)
            span.num_x = [
                a - d_hat * b
                for a, b in zip(bern_product(span.px, span.rho), w2im)
            ]
            span.num_y = [
                a + d_hat * b
                for a, b in zip(bern_product(span.py, span.rho), w2re)
            ]
            # The weight elevated to the numerator degree: every
            # coefficientwise combination ``num - k * weight`` below
            # requires equal-degree Bernstein lists.
            span.rho_elev = bern_elevate(
                span.rho, len(span.num_x) - len(span.rho)
            )
        else:
            span.num_x = span.px
            span.num_y = span.py
            span.rho_elev = None
        # Float shadows for the Newton seeding of crossing refinement
        # (never authoritative; every acceptance is exact).
        span.fnum_x = [float(c) for c in span.num_x]
        span.fnum_y = [float(c) for c in span.num_y]
        span.fweight = (
            [float(c) for c in span.rho] if locus.offset else None
        )
        locus.spans.append(span)
    locus.trim_slack = _ZERO
    if locus.offset:
        _trim_cusp_strips(locus)
    _build_pieces(locus)
    diameter = _ZERO
    for piece in locus.pieces:
        box = _position_range(locus, piece.span, piece.a, piece.b)
        piece.box = box
        diameter = max(
            diameter, box[0][1] - box[0][0] + box[1][1] - box[1][0]
        )
    locus.micro_limit = max(diameter, Fraction(1, 10**300)) * _MICRO_FACTOR
    return locus


def _trim_cusp_strips(locus) -> None:
    """Exclude guard strips around cusp-derived cell boundaries.

    Metric cell boundaries at cusps are binary64 roots within two ulps of
    the exact root, so a mis-signed micro strip next to each boundary
    carries the wrong constant-sign factor and would invalidate the
    monotone-piece certificates (the exact tangent has already flipped
    there).  Each strip is removed from the crossing search and its
    position box is absorbed into the slack budget: replacing the curve
    inside the box changes the fill by at most the box area (spec 5.4).
    """

    def guard(boundary: Fraction) -> Fraction:
        return Fraction(math.ulp(max(abs(float(boundary)), 1e-300))) * 64

    def strip_slack(span_index, lo, hi) -> Fraction:
        box = _position_range(locus, span_index, lo, hi)
        diameter = (box[0][1] - box[0][0]) + (box[1][1] - box[1][0])
        return diameter * diameter

    for span_index, span in enumerate(locus.spans):
        cells = list(span.cells)
        for k in range(len(cells) - 1):
            if cells[k][2] == cells[k + 1][2]:
                continue
            boundary = cells[k][1]
            g = guard(boundary)
            locus.trim_slack += strip_slack(
                span_index, boundary - g, boundary + g
            )
            cells[k] = (cells[k][0], boundary - g, cells[k][2])
            cells[k + 1] = (boundary + g, cells[k + 1][1], cells[k + 1][2])
        span.cells = cells
    count = len(locus.spans)
    for span_index in range(count):
        left = locus.spans[span_index]
        right = locus.spans[(span_index + 1) % count]
        if left.cells[-1][2] == right.cells[0][2]:
            continue
        g = guard(_ONE)
        locus.trim_slack += strip_slack(
            span_index, _ONE - g, _ONE
        ) + strip_slack((span_index + 1) % count, _ZERO, g)
        a, b, eta = left.cells[-1]
        left.cells[-1] = (a, b - g, eta)
        a, b, eta = right.cells[0]
        right.cells[0] = (a + g, b, eta)


def _tangent_generators(locus, span_index, a, b, eta):
    """Bernstein generators of ``eta * w**2`` restricted to ``[a, b]``."""
    span = locus.spans[span_index]
    whole = (a, b) == (0, 1)
    wre = span.wre if whole else bern_restrict(span.wre, a, b)
    wim = span.wim if whole else bern_restrict(span.wim, a, b)
    w2re, w2im = _complex_square(wre, wim)
    return [(eta * gx, eta * gy) for gx, gy in zip(w2re, w2im)]


def _gens_certified(gens):
    """Common strict half-plane direction of the generators, or None."""
    dx = sum(g[0] for g in gens)
    dy = sum(g[1] for g in gens)
    if dx == 0 and dy == 0:
        return None
    for gx, gy in gens:
        if not gx * dx + gy * dy > 0:
            return None
    return (dx, dy)


def _position_range(locus, span_index, a, b):
    """Exact rational x/y ranges of the locus over one span subinterval."""
    span = locus.spans[span_index]
    if locus.offset:
        weight = _bern_range(span.rho, a, b)
        if not weight[0] > 0:
            raise _fail(
                "A locus weight range lost positivity", span_id=span_index
            )
        return (
            _iv_div(_bern_range(span.num_x, a, b), weight),
            _iv_div(_bern_range(span.num_y, a, b), weight),
        )
    return (_bern_range(span.px, a, b), _bern_range(span.py, a, b))


def _build_pieces(locus) -> None:
    pieces = []
    for span_index, span in enumerate(locus.spans):
        for a0, b0, eta in span.cells:
            stack = [(a0, b0, 0)]
            local = []
            while stack:
                a, b, depth = stack.pop()
                gens = _tangent_generators(locus, span_index, a, b, eta)
                certified = _gens_certified(gens) is not None
                if certified and locus.offset:
                    whole = (a, b) == (0, 1)
                    rho_r = (
                        span.rho if whole else bern_restrict(span.rho, a, b)
                    )
                    certified = all(c > 0 for c in rho_r)
                if certified:
                    piece = _Piece()
                    piece.span = span_index
                    piece.a = a
                    piece.b = b
                    piece.eta = eta
                    piece.box = None
                    local.append(piece)
                    continue
                if depth >= _MAX_PIECE_DEPTH:
                    raise _fail(
                        "Monotone-piece certification exceeded its depth "
                        "bound",
                        span_id=span_index,
                        quantity="piece depth",
                        bound=f"< {_MAX_PIECE_DEPTH}",
                    )
                mid = a + (b - a) / 2
                stack.append((mid, b, depth + 1))
                stack.append((a, mid, depth + 1))
            local.sort(key=lambda piece: piece.a)
            pieces.extend(local)
    locus.pieces = pieces


# ---------------------------------------------------------------------------
# Crossing enumeration
# ---------------------------------------------------------------------------


class _Crossing:
    __slots__ = ("point", "s_iv", "span_p", "span_q", "t_iv")


class _Engine:
    def __init__(self, locus):
        self.locus = locus
        self.crossings: list[_Crossing] = []
        self.slack = _ZERO  # normalized absorbed box-area budget

    # -- primitives -------------------------------------------------------

    @staticmethod
    def _boxes_disjoint(box_p, box_q) -> bool:
        return (
            box_p[0][1] < box_q[0][0]
            or box_q[0][1] < box_p[0][0]
            or box_p[1][1] < box_q[1][0]
            or box_q[1][1] < box_p[1][0]
        )

    def _absorb(self, box_p, box_q) -> bool:
        xs_lo = min(box_p[0][0], box_q[0][0])
        xs_hi = max(box_p[0][1], box_q[0][1])
        ys_lo = min(box_p[1][0], box_q[1][0])
        ys_hi = max(box_p[1][1], box_q[1][1])
        diameter = (xs_hi - xs_lo) + (ys_hi - ys_lo)
        if diameter <= self.locus.micro_limit:
            self.slack += diameter * diameter
            statistics["micro_blocks_absorbed"] += 1
            return True
        return False

    def _scalar(self, span_index, direction, a, b):
        """Bernstein data of ``direction . locus`` over a span interval.

        Returns ``(numerator, weight)``; polynomial loci have ``None``
        weight.
        """
        span = self.locus.spans[span_index]
        numerator = [
            direction[0] * cx + direction[1] * cy
            for cx, cy in zip(span.num_x, span.num_y)
        ]
        if (a, b) != (0, 1):
            numerator = bern_restrict(numerator, a, b)
        if not self.locus.offset:
            return numerator, None
        weight = (
            span.rho_elev
            if (a, b) == (0, 1)
            else bern_restrict(span.rho_elev, a, b)
        )
        return numerator, weight

    @staticmethod
    def _scalar_end(scalar, at_end: bool) -> Fraction:
        numerator, weight = scalar
        value = numerator[-1] if at_end else numerator[0]
        if weight is None:
            return value
        return value / (weight[-1] if at_end else weight[0])

    @staticmethod
    def _range_minus(scalar, anchor: Fraction):
        """Tight range enclosure of ``scalar - anchor`` over its block.

        Recentring happens exactly in the Bernstein coefficients before
        the single hull division, so the enclosure width tracks the
        variation of the scalar itself rather than its absolute value.
        Without this, rational-hull division slop (proportional to the
        coordinate magnitude) would swallow the cubically small branch
        separation next to every offset cusp.
        """
        numerator, weight = scalar
        if weight is None:
            shifted = [c - anchor for c in numerator]
            return (min(shifted), max(shifted))
        shifted = [c - anchor * w for c, w in zip(numerator, weight)]
        return _iv_div(
            (min(shifted), max(shifted)), (min(weight), max(weight))
        )

    @staticmethod
    def _miranda_pairings(dir_p, dir_q):
        """Candidate Miranda direction pairs for one separated block.

        The perpendicular pairing certifies ordinary transversal
        crossings; the common-tangent pairings certify the shallow
        nearly parallel and nearly anti-parallel crossings of offset
        cusp slivers, where both perpendiculars degenerate to the same
        transverse direction.
        """
        pairs = [((-dir_q[1], dir_q[0]), (-dir_p[1], dir_p[0]))]
        for tangent in (
            (dir_p[0] - dir_q[0], dir_p[1] - dir_q[1]),
            (dir_p[0] + dir_q[0], dir_p[1] + dir_q[1]),
        ):
            if tangent != (0, 0):
                pairs.append((tangent, (-tangent[1], tangent[0])))
                pairs.append(((-tangent[1], tangent[0]), tangent))
        return pairs

    def _miranda(self, sp, sa, sb, sq, ta, tb, dir_a, dir_b) -> bool:
        """Separable Miranda existence test on one block."""
        u = self._scalar(sp, dir_a, sa, sb)
        v = self._scalar(sq, dir_a, ta, tb)
        u_lo = self._scalar_end(u, False)
        u_hi = self._scalar_end(u, True)
        v_minus_lo = self._range_minus(v, u_lo)
        v_minus_hi = self._range_minus(v, u_hi)
        if not (
            (v_minus_lo[0] > 0 and v_minus_hi[1] < 0)
            or (v_minus_lo[1] < 0 and v_minus_hi[0] > 0)
        ):
            return False
        p = self._scalar(sp, dir_b, sa, sb)
        q = self._scalar(sq, dir_b, ta, tb)
        q_lo = self._scalar_end(q, False)
        q_hi = self._scalar_end(q, True)
        p_minus_lo = self._range_minus(p, q_lo)
        p_minus_hi = self._range_minus(p, q_hi)
        return (
            (p_minus_lo[0] > 0 and p_minus_hi[1] < 0)
            or (p_minus_lo[1] < 0 and p_minus_hi[0] > 0)
        )

    @staticmethod
    def _separated(gens_p, gens_q) -> bool:
        """One strict determinant sign over all generator pairs."""
        sign = 0
        for px, py in gens_p:
            for qx, qy in gens_q:
                det = px * qy - py * qx
                if det == 0:
                    return False
                current = 1 if det > 0 else -1
                if sign == 0:
                    sign = current
                elif current != sign:
                    return False
        return True

    # -- block recursion --------------------------------------------------

    def _direction_separated(self, sp, sa, sb, sq, ta, tb, direction):
        """Prune when the two arcs' scalar ranges along ``direction`` are
        disjoint.  This is the essential exclusion for nearly parallel
        arcs, whose axis boxes overlap over long stretches while their
        transverse coordinates never meet."""
        scalar_p = self._scalar(sp, direction, sa, sb)
        scalar_q = self._scalar(sq, direction, ta, tb)
        anchor = self._scalar_end(scalar_q, False)
        range_p = self._range_minus(scalar_p, anchor)
        range_q = self._range_minus(scalar_q, anchor)
        return range_p[1] < range_q[0] or range_q[1] < range_p[0]

    def _block(self, sp, sa, sb, ep, sq, ta, tb, eq, depth):
        box_p = _position_range(self.locus, sp, sa, sb)
        box_q = _position_range(self.locus, sq, ta, tb)
        if self._boxes_disjoint(box_p, box_q):
            return
        gens_p = _tangent_generators(self.locus, sp, sa, sb, ep)
        gens_q = _tangent_generators(self.locus, sq, ta, tb, eq)
        dir_p = _gens_certified(gens_p)
        dir_q = _gens_certified(gens_q)
        if dir_p is not None and self._direction_separated(
            sp, sa, sb, sq, ta, tb, (-dir_p[1], dir_p[0])
        ):
            return
        if dir_q is not None and self._direction_separated(
            sp, sa, sb, sq, ta, tb, (-dir_q[1], dir_q[0])
        ):
            return
        if (
            dir_p is not None
            and dir_q is not None
            and self._separated(gens_p, gens_q)
        ):
            for dir_a, dir_b in self._miranda_pairings(dir_p, dir_q):
                if self._miranda(sp, sa, sb, sq, ta, tb, dir_a, dir_b):
                    if self._certify(sp, sa, sb, sq, ta, tb, dir_p, dir_q):
                        return
                    # The certified crossing resisted localization (a
                    # near-tangential kiss): keep subdividing so the
                    # contact isolates into tightly absorbed micro
                    # blocks and the remainder prunes.
                    break
            # Single-signed pairwise determinants bound the pair to at
            # most one crossing, but a failed Miranda is not yet
            # emptiness; subdivide below.
        if self._absorb(box_p, box_q):
            return
        if depth >= _MAX_BLOCK_DEPTH:
            raise _fail(
                "Crossing certification exceeded its block depth bound",
                quantity="block depth",
                bound=f"< {_MAX_BLOCK_DEPTH}",
            )
        split = _SPLITS[depth % len(_SPLITS)]
        if sb - sa >= tb - ta:
            mid = sa + (sb - sa) * split
            self._block(sp, sa, mid, ep, sq, ta, tb, eq, depth + 1)
            self._block(sp, mid, sb, ep, sq, ta, tb, eq, depth + 1)
        else:
            mid = ta + (tb - ta) * split
            self._block(sp, sa, sb, ep, sq, ta, mid, eq, depth + 1)
            self._block(sp, sa, sb, ep, sq, mid, tb, eq, depth + 1)

    def _fgap(self, sp, s_value: float, sq, t_value: float):
        """Float position gap ``P(s) - Q(t)`` for Newton seeding only."""

        def evaluate(span, value):
            def decast(coeffs):
                work = list(coeffs)
                complement = 1.0 - value
                for width in range(len(work) - 1, 0, -1):
                    for k in range(width):
                        work[k] = complement * work[k] + value * work[k + 1]
                return work[0]

            x = decast(span.fnum_x)
            y = decast(span.fnum_y)
            if span.fweight is None:
                return x, y
            w = decast(span.fweight)
            return x / w, y / w

        px, py = evaluate(self.locus.spans[sp], s_value)
        qx, qy = evaluate(self.locus.spans[sq], t_value)
        return px - qx, py - qy

    def _scalar_at(self, span_index, direction, at: Fraction) -> Fraction:
        """Exact value of ``direction . locus`` at one parameter."""
        span = self.locus.spans[span_index]
        value = direction[0] * bern_eval(span.num_x, at) + direction[
            1
        ] * bern_eval(span.num_y, at)
        if not self.locus.offset:
            return value
        return value / bern_eval(span.rho, at)

    def _certify(self, sp, sa, sb, sq, ta, tb, dir_p, dir_q) -> None:
        """Monotone bisection refinement of one certified crossing.

        On the separated block, ``A . P`` is strictly monotone in ``s``
        for ``A = perp(dir_q)`` (every tangent generator has one strict
        sign against ``A``), so comparing ``A . P(m)`` with the recentred
        range of ``A . Q`` decides which side of a trial split holds the
        crossing; symmetrically for ``t``.  A straddling comparison
        retries a jittered split.
        """
        dir_a = (-dir_q[1], dir_q[0])
        dir_b = (-dir_p[1], dir_p[0])
        sign_f = 1 if (dir_a[0] * dir_p[0] + dir_a[1] * dir_p[1]) > 0 else -1
        sign_g = 1 if (dir_b[0] * dir_q[0] + dir_b[1] * dir_q[1]) > 0 else -1
        def monotone_s():
            nonlocal sa, sb
            for split in _SPLITS:
                mid = sa + (sb - sa) * split
                value = self._scalar_at(sp, dir_a, mid)
                other = self._scalar(sq, dir_a, ta, tb)
                gap = self._range_minus(other, value)
                if gap[1] < 0:  # A.Q < A.P(mid): f(mid, .) > 0
                    if sign_f > 0:
                        sb = mid
                    else:
                        sa = mid
                    return True
                if gap[0] > 0:
                    if sign_f > 0:
                        sa = mid
                    else:
                        sb = mid
                    return True
            return False

        def monotone_t():
            nonlocal ta, tb
            for split in _SPLITS:
                mid = ta + (tb - ta) * split
                value = self._scalar_at(sq, dir_b, mid)
                other = self._scalar(sp, dir_b, sa, sb)
                gap = self._range_minus(other, value)
                if gap[1] < 0:  # B.P < B.Q(mid): g(., mid) < 0
                    if sign_g > 0:
                        tb = mid
                    else:
                        ta = mid
                    return True
                if gap[0] > 0:
                    if sign_g > 0:
                        ta = mid
                    else:
                        tb = mid
                    return True
            return False

        def miranda_halving():
            # Both axes wide: shrink to whichever half re-certifies
            # existence (the crossing is unique, so at most one does).
            nonlocal sa, sb, ta, tb
            pairings = self._miranda_pairings(dir_p, dir_q)
            for split in _SPLITS:
                if sb - sa >= tb - ta:
                    mid = sa + (sb - sa) * split
                    halves = ((sa, mid, ta, tb), (mid, sb, ta, tb))
                else:
                    mid = ta + (tb - ta) * split
                    halves = ((sa, sb, ta, mid), (sa, sb, mid, tb))
                for half in halves:
                    for pair_a, pair_b in pairings:
                        if self._miranda(
                            sp, half[0], half[1], sq, half[2], half[3],
                            pair_a, pair_b,
                        ):
                            sa, sb, ta, tb = half
                            return True
            return False

        def newton_restart():
            # Float Newton seed followed by exact tiny-box verification;
            # rigor lives entirely in the exact Miranda acceptance.
            nonlocal sa, sb, ta, tb
            s0 = float(sa + (sb - sa) / 2)
            t0 = float(ta + (tb - ta) / 2)
            for _ in range(40):
                fx, fy = self._fgap(sp, s0, sq, t0)
                if not (math.isfinite(fx) and math.isfinite(fy)):
                    return False
                step = 1.0e-7
                axx, axy = self._fgap(sp, s0 + step, sq, t0)
                ayx, ayy = self._fgap(sp, s0, sq, t0 + step)
                j11 = (axx - fx) / step
                j21 = (axy - fy) / step
                j12 = (ayx - fx) / step
                j22 = (ayy - fy) / step
                det = j11 * j22 - j12 * j21
                if det == 0.0 or not math.isfinite(det):
                    return False
                ds = (fx * j22 - fy * j12) / det
                dt = (fy * j11 - fx * j21) / det
                s0 -= ds
                t0 -= dt
                if abs(ds) + abs(dt) < 1.0e-15:
                    break
            pairings = self._miranda_pairings(dir_p, dir_q)
            for exponent in (36, 30, 24):
                h = Fraction(1, 2**exponent)
                s_mid = Fraction(round(s0 * 2**48), 2**48)
                t_mid = Fraction(round(t0 * 2**48), 2**48)
                box = (
                    max(sa, s_mid - h), min(sb, s_mid + h),
                    max(ta, t_mid - h), min(tb, t_mid + h),
                )
                if not (box[0] < box[1] and box[2] < box[3]):
                    continue
                for pair_a, pair_b in pairings:
                    if self._miranda(
                        sp, box[0], box[1], sq, box[2], box[3],
                        pair_a, pair_b,
                    ):
                        sa, sb, ta, tb = box
                        return True
            return False

        # One float-Newton shot first: a well-conditioned crossing lands
        # in a certified 2**-36 box immediately, and the bisection below
        # only polishes it to the acceptance width.  Blocks that are
        # already tiny descend from an earlier failed Newton, so the
        # retry is skipped for them.
        try_newton = sb - sa > Fraction(1, 2**20) or tb - ta > Fraction(
            1, 2**20
        )
        if try_newton:
            newton_restart()
        retried = not try_newton
        for _ in range(_MAX_REFINE_STEPS):
            if sb - sa <= _REFINE_WIDTH and tb - ta <= _REFINE_WIDTH:
                break
            if sb - sa > tb - ta or tb - ta <= _REFINE_WIDTH:
                advanced = monotone_s() or monotone_t()
            else:
                advanced = monotone_t() or monotone_s()
            if not advanced:
                advanced = miranda_halving()
            if not advanced and not retried:
                retried = True
                advanced = newton_restart()
            if not advanced:
                break
        if sb - sa > _REFINE_WIDTH or tb - ta > _REFINE_WIDTH:
            return False
        crossing = _Crossing()
        crossing.span_p = sp
        crossing.s_iv = (sa, sb)
        crossing.span_q = sq
        crossing.t_iv = (ta, tb)
        crossing.point = _position_range(self.locus, sp, sa, sb)
        self.crossings.append(crossing)
        statistics["crossings_certified"] += 1
        return True

    # -- adjacency --------------------------------------------------------

    def _process_adjacent(self, piece_p, piece_q) -> None:
        """Shared-joint pair: certify the corner, then sweep the rest."""
        fraction = _ONE
        p_cut = piece_p.b
        q_cut = piece_q.a
        for _ in range(_MAX_BLOCK_DEPTH):
            fraction = fraction / 2
            p_cut = piece_p.b - (piece_p.b - piece_p.a) * fraction
            q_cut = piece_q.a + (piece_q.b - piece_q.a) * fraction
            gens = _tangent_generators(
                self.locus, piece_p.span, p_cut, piece_p.b, piece_p.eta
            ) + _tangent_generators(
                self.locus, piece_q.span, piece_q.a, q_cut, piece_q.eta
            )
            if _gens_certified(gens) is not None:
                break
            box_p = _position_range(
                self.locus, piece_p.span, p_cut, piece_p.b
            )
            box_q = _position_range(
                self.locus, piece_q.span, piece_q.a, q_cut
            )
            if self._absorb(box_p, box_q):
                break
        else:
            raise _fail(
                "A shared-joint corner could not be certified injective",
                span_id=piece_p.span,
            )
        if p_cut > piece_p.a:
            self._block(
                piece_p.span, piece_p.a, p_cut, piece_p.eta,
                piece_q.span, piece_q.a, piece_q.b, piece_q.eta, 0,
            )
        if q_cut < piece_q.b:
            self._block(
                piece_p.span, p_cut, piece_p.b, piece_p.eta,
                piece_q.span, q_cut, piece_q.b, piece_q.eta, 0,
            )

    def _dedupe(self) -> None:
        """Merge crossing enclosures certified from overlapping blocks.

        A crossing next to a piece boundary can be certified from both
        touching pairs; the merged record hulls the parameter and point
        enclosures (spec 5.5).
        """

        def sides_overlap(span_a, iv_a, span_b, iv_b):
            return span_a == span_b and not (
                iv_a[1] < iv_b[0] or iv_b[1] < iv_a[0]
            )

        def hull(iv_a, iv_b):
            return (min(iv_a[0], iv_b[0]), max(iv_a[1], iv_b[1]))

        merged = True
        while merged:
            merged = False
            for i in range(len(self.crossings)):
                for j in range(i + 1, len(self.crossings)):
                    a, b = self.crossings[i], self.crossings[j]
                    direct = sides_overlap(
                        a.span_p, a.s_iv, b.span_p, b.s_iv
                    ) and sides_overlap(a.span_q, a.t_iv, b.span_q, b.t_iv)
                    swapped = sides_overlap(
                        a.span_p, a.s_iv, b.span_q, b.t_iv
                    ) and sides_overlap(a.span_q, a.t_iv, b.span_p, b.s_iv)
                    if not (direct or swapped):
                        continue
                    if swapped and not direct:
                        b.span_p, b.span_q = b.span_q, b.span_p
                        b.s_iv, b.t_iv = b.t_iv, b.s_iv
                    a.s_iv = hull(a.s_iv, b.s_iv)
                    a.t_iv = hull(a.t_iv, b.t_iv)
                    a.point = (
                        hull(a.point[0], b.point[0]),
                        hull(a.point[1], b.point[1]),
                    )
                    del self.crossings[j]
                    statistics["crossings_certified"] -= 1
                    merged = True
                    break
                if merged:
                    break

    def _shares_endpoint(self, piece_p, piece_q) -> bool:
        """True when piece_q starts exactly where piece_p ends.

        Cusp-guard trimming leaves a gap between the pieces around every
        eta flip; those pairs are swept by the ordinary far engine and
        their strips are covered by the absorbed trim slack.
        """
        if piece_p.span == piece_q.span:
            return piece_p.b == piece_q.a
        return piece_p.b == 1 and piece_q.a == 0

    def run(self):
        self.slack += self.locus.trim_slack
        pieces = self.locus.pieces
        count = len(pieces)
        for i in range(count):
            for j in range(i + 1, count):
                if j == i + 1 or (i == 0 and j == count - 1):
                    piece_p, piece_q = (
                        (pieces[i], pieces[j])
                        if j == i + 1
                        else (pieces[j], pieces[i])
                    )
                    if self._shares_endpoint(piece_p, piece_q):
                        self._process_adjacent(piece_p, piece_q)
                        continue
                if not self._boxes_disjoint(pieces[i].box, pieces[j].box):
                    self._block(
                        pieces[i].span, pieces[i].a, pieces[i].b,
                        pieces[i].eta,
                        pieces[j].span, pieces[j].a, pieces[j].b,
                        pieces[j].eta, 0,
                    )
        self._dedupe()
        return self.crossings, self.slack


# ---------------------------------------------------------------------------
# Decomposition (Seifert smoothing at the certified crossings)
# ---------------------------------------------------------------------------


class _Cut:
    __slots__ = ("partner", "point", "span", "t_iv")

    def sort_key(self):
        return (self.span, (self.t_iv[0] + self.t_iv[1]) / 2)


def _make_cuts(crossings):
    cuts = []
    for crossing in crossings:
        first = _Cut()
        first.span = crossing.span_p
        first.t_iv = crossing.s_iv
        first.point = crossing.point
        second = _Cut()
        second.span = crossing.span_q
        second.t_iv = crossing.t_iv
        second.point = crossing.point
        first.partner = second
        second.partner = first
        cuts.extend((first, second))
    cuts.sort(key=_Cut.sort_key)
    for index in range(len(cuts) - 1):
        a, b = cuts[index], cuts[index + 1]
        if a.span == b.span and a.t_iv[1] > b.t_iv[0]:
            raise _fail(
                "Two crossing enclosures overlap on the parameter circle",
                span_id=a.span,
            )
    return cuts


def _segments_between(locus, cut_a, cut_b):
    """Forward arc segments ``(span, lo_iv, hi_iv)`` from cut_a to cut_b."""
    spans = len(locus.spans)
    segments = []
    span = cut_a.span
    if cut_b.span == span and cut_b.t_iv[0] >= cut_a.t_iv[1]:
        segments.append((span, cut_a.t_iv, cut_b.t_iv))
        return segments
    segments.append((span, cut_a.t_iv, (_ONE, _ONE)))
    span = (span + 1) % spans
    guard = 0
    while span != cut_b.span:
        segments.append((span, (_ZERO, _ZERO), (_ONE, _ONE)))
        span = (span + 1) % spans
        guard += 1
        if guard > spans + 1:
            raise _fail("Arc walking failed to terminate")
    segments.append((span, (_ZERO, _ZERO), cut_b.t_iv))
    return segments


def _decompose(locus, crossings):
    cuts = _make_cuts(crossings)
    count = len(cuts)
    index_of = {id(cut): index for index, cut in enumerate(cuts)}
    visited = set()
    loops = []
    for start in range(count):
        if start in visited:
            continue
        sequence = []
        current = start
        while current not in visited:
            visited.add(current)
            sequence.append(current)
            arrival = (current + 1) % count
            current = index_of[id(cuts[arrival].partner)]
        loops.append(sequence)
    loop_records = []
    for sequence in loops:
        segments = []
        for arc_index in sequence:
            cut_a = cuts[arc_index]
            cut_b = cuts[(arc_index + 1) % count]
            segments.extend(_segments_between(locus, cut_a, cut_b))
        loop_records.append(segments)
    statistics["loops_decomposed"] += len(loop_records)
    return loop_records


# ---------------------------------------------------------------------------
# Loop areas (rational intervals, normalized frame)
# ---------------------------------------------------------------------------


def _w_box(span, lo, hi):
    return (_bern_range(span.wre, lo, hi), _bern_range(span.wim, lo, hi))


def _angle_box(x_iv, y_iv, precision):
    """Enclosure of ``atan2`` over a rational box with ``x > 0``."""
    if not x_iv[0] > 0:
        raise _fail("A phase box reached the closed left half plane")
    lo = None
    hi = None
    for x_arg in x_iv:
        for y_arg in y_iv:
            value, error = atan2_ball(y_arg, x_arg, precision)
            b_lo, b_hi = ball_to_fraction_bounds(value, error, precision)
            lo = b_lo if lo is None else min(lo, b_lo)
            hi = b_hi if hi is None else max(hi, b_hi)
    return (lo, hi)


def _phase_increment(locus, span_index, lo_iv, hi_iv, precision):
    """Tangent-angle increment enclosure over one span subinterval.

    Telescopes principal preimage rotations between the arc bounds and
    every interior metric-cell boundary; each step stays below ``pi/2``
    by the cell sector certificates, so the principal branch equals the
    continuous lift (specification 3.2).
    """
    span = locus.spans[span_index]
    stops = [lo_iv]
    for _a_loc, b_loc, _eta in span.cells[:-1]:
        if lo_iv[1] < b_loc < hi_iv[0]:
            stops.append((b_loc, b_loc))
    stops.append(hi_iv)
    total = (_ZERO, _ZERO)
    for start, end in zip(stops, stops[1:]):
        wa = _w_box(span, start[0], start[1])
        wb = _w_box(span, end[0], end[1])
        x_iv = _iv_add(_iv_mul(wb[0], wa[0]), _iv_mul(wb[1], wa[1]))
        y_iv = _iv_sub(_iv_mul(wb[1], wa[0]), _iv_mul(wb[0], wa[1]))
        angle = _angle_box(x_iv, y_iv, precision)
        total = _iv_add(total, _iv_scale(angle, Fraction(2)))
    return total


def _zn_term(locus, span_index, t_iv):
    """Enclosure of ``[z, N]`` at one parameter interval."""
    span = locus.spans[span_index]
    zx = _bern_range(span.px, t_iv[0], t_iv[1])
    zy = _bern_range(span.py, t_iv[0], t_iv[1])
    wre, wim = _w_box(span, t_iv[0], t_iv[1])
    w2re = _iv_sub(_iv_mul(wre, wre), _iv_mul(wim, wim))
    w2im = _iv_scale(_iv_mul(wre, wim), Fraction(2))
    norm = _iv_add(_iv_mul(wre, wre), _iv_mul(wim, wim))
    nx = _iv_div((-w2im[1], -w2im[0]), norm)
    ny = _iv_div(w2re, norm)
    return _iv_det(zx, zy, nx, ny)


def _segment_double_area(locus, segment, precision):
    """Enclosure of ``integral [z_d, z_d'] dt`` over one arc segment."""
    span_index, lo_iv, hi_iv = segment
    span = locus.spans[span_index]
    green = _iv_sub(
        _bern_range(span.green, hi_iv[0], hi_iv[1]),
        _bern_range(span.green, lo_iv[0], lo_iv[1]),
    )
    if not locus.offset:
        return green
    arc = _iv_scale(
        _iv_sub(
            _bern_range(span.rho_anti, hi_iv[0], hi_iv[1]),
            _bern_range(span.rho_anti, lo_iv[0], lo_iv[1]),
        ),
        span.h,
    )
    boundary = _iv_sub(
        _zn_term(locus, span_index, hi_iv), _zn_term(locus, span_index, lo_iv)
    )
    theta = _phase_increment(locus, span_index, lo_iv, hi_iv, precision)
    total = _iv_add(
        green,
        _iv_scale(
            _iv_sub(boundary, _iv_scale(arc, Fraction(2))), locus.d_hat
        ),
    )
    return _iv_add(total, _iv_scale(theta, locus.d_hat * locus.d_hat))


def _joint_phase_correction(locus, left_span, right_span, precision):
    """Squared-tangent join correction between consecutive spans (9.1)."""
    left = locus.spans[left_span]
    right = locus.spans[right_span]
    ql_re = left.wre[-1] * left.wre[-1] - left.wim[-1] * left.wim[-1]
    ql_im = 2 * left.wre[-1] * left.wim[-1]
    qr_re = right.wre[0] * right.wre[0] - right.wim[0] * right.wim[0]
    qr_im = 2 * right.wre[0] * right.wim[0]
    x_arg = ql_re * qr_re + ql_im * qr_im
    y_arg = ql_re * qr_im - ql_im * qr_re
    if not x_arg > 0:
        raise _fail(
            "A cyclic tangent join contradicts the verified continuity "
            "contract",
            index=right_span,
        )
    value, error = atan2_ball(y_arg, x_arg, precision)
    return ball_to_fraction_bounds(value, error, precision)


def _position_point(locus, span_index, t_iv):
    return _position_range(locus, span_index, t_iv[0], t_iv[1])


def _loop_area(locus, segments, precision):
    """Certified normalized algebraic area enclosure of one sub-loop."""
    total = (_ZERO, _ZERO)
    d_sq = locus.d_hat * locus.d_hat
    count = len(segments)
    for index, segment in enumerate(segments):
        total = _iv_add(total, _segment_double_area(locus, segment, precision))
        nxt = segments[(index + 1) % count]
        joint = segment[2] == (_ONE, _ONE) and nxt[1] == (_ZERO, _ZERO)
        end_pos = _position_point(locus, segment[0], segment[2])
        start_pos = _position_point(locus, nxt[0], nxt[1])
        total = _iv_add(
            total,
            _iv_det(end_pos[0], end_pos[1], start_pos[0], start_pos[1]),
        )
        if joint and locus.offset:
            correction = _joint_phase_correction(
                locus, segment[0], nxt[0], precision
            )
            total = _iv_add(total, _iv_scale(correction, d_sq))
    return _iv_scale(total, _HALF)


# ---------------------------------------------------------------------------
# Containment by certified ray casting
# ---------------------------------------------------------------------------

_TEST_FRACTIONS = (
    Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(2, 5),
    Fraction(3, 5), Fraction(1, 7), Fraction(3, 7), Fraction(5, 7),
    Fraction(4, 11), Fraction(7, 11), Fraction(2, 13), Fraction(11, 13),
    Fraction(5, 17), Fraction(12, 17), Fraction(8, 19), Fraction(15, 19),
)


class _RayAmbiguity(Exception):
    """The chosen test point produced an undecidable ray configuration."""


def _widest_segment(segments):
    best = None
    width = Fraction(-1)
    for segment in segments:
        current = segment[2][0] - segment[1][1]
        if current > width:
            width = current
            best = segment
    return best


def _test_point(locus, segments, attempt):
    segment = _widest_segment(segments)
    span_index, lo_iv, hi_iv = segment
    fraction = _TEST_FRACTIONS[attempt % len(_TEST_FRACTIONS)]
    t = lo_iv[1] + (hi_iv[0] - lo_iv[1]) * fraction
    span = locus.spans[span_index]
    if locus.offset:
        weight = bern_eval(span.rho, t)
        return (
            bern_eval(span.num_x, t) / weight,
            bern_eval(span.num_y, t) / weight,
        )
    return (bern_eval(span.px, t), bern_eval(span.py, t))


def _segment_ray_winding(locus, segment, point):
    """Signed +x ray-crossing count of one arc segment about ``point``."""
    span_index, lo_iv, hi_iv = segment
    span = locus.spans[span_index]
    px, py = point
    if locus.offset:
        vertical = [c - py * w for c, w in zip(span.num_y, span.rho_elev)]
        horizontal = [c - px * w for c, w in zip(span.num_x, span.rho_elev)]
    else:
        vertical = [c - py for c in span.py]
        horizontal = [c - px for c in span.px]
    if all(c > 0 for c in vertical) or all(c < 0 for c in vertical):
        return 0
    degree = len(vertical) - 1
    derivative = [degree * (b - a) for a, b in zip(vertical, vertical[1:])]
    m0, m1, interior = isolate_bernstein_roots(vertical)
    if m0 or m1:
        raise _RayAmbiguity
    winding = 0
    for lo, hi, multiplicity, _refiner in interior:
        if multiplicity % 2 == 0:
            continue  # tangential graze: no winding contribution
        s_lo = sign_at(vertical, lo)
        for _ in range(_MAX_ROOT_STEPS):
            if hi <= lo_iv[0] or lo >= hi_iv[1]:
                break  # certified outside the segment
            if lo >= lo_iv[1] and hi <= hi_iv[0]:
                break  # certified inside the segment
            mid = lo + (hi - lo) / 2
            s_mid = sign_at(vertical, mid)
            if s_mid == 0:
                lo = hi = mid
            elif s_mid == s_lo:
                lo = mid
            else:
                hi = mid
        if hi <= lo_iv[0] or lo >= hi_iv[1]:
            continue
        if not (lo >= lo_iv[1] and hi <= hi_iv[0]):
            raise _RayAmbiguity  # root inside a cut-uncertainty zone

        contribution = None
        for _ in range(_MAX_ROOT_STEPS):
            x_range = _bern_range(horizontal, lo, hi)
            if x_range[1] < 0:
                contribution = 0
                break
            if x_range[0] > 0:
                d_range = _bern_range(derivative, lo, hi)
                if d_range[0] > 0:
                    contribution = 1
                    break
                if d_range[1] < 0:
                    contribution = -1
                    break
            if lo == hi:
                raise _RayAmbiguity
            mid = lo + (hi - lo) / 2
            s_mid = sign_at(vertical, mid)
            if s_mid == 0:
                lo = hi = mid
            elif s_mid == s_lo:
                lo = mid
            else:
                hi = mid
        if contribution is None:
            raise _RayAmbiguity
        winding += contribution
    return winding


def _loop_winding(locus, segments, point):
    return sum(
        _segment_ray_winding(locus, segment, point) for segment in segments
    )


def _containment_parents(locus, loop_records, areas):
    """Laminar parent assignment by certified point-in-loop winding."""
    count = len(loop_records)
    contains = [[False] * count for _ in range(count)]
    for inner in range(count):
        for outer in range(count):
            if inner == outer:
                continue
            statistics["containment_tests"] += 1
            for attempt in range(_MAX_TEST_POINTS):
                point = _test_point(locus, loop_records[inner], attempt)
                try:
                    winding = _loop_winding(
                        locus, loop_records[outer], point
                    )
                except _RayAmbiguity:
                    continue
                contains[outer][inner] = winding != 0
                break
            else:
                raise _fail(
                    "No test point produced a decidable containment ray",
                    quantity="containment",
                )
    parents = [None] * count
    for inner in range(count):
        best = None
        for outer in range(count):
            if contains[outer][inner] and (
                best is None
                or areas[outer][1] - areas[outer][0] + abs(
                    areas[outer][0] + areas[outer][1]
                ) < areas[best][1] - areas[best][0] + abs(
                    areas[best][0] + areas[best][1]
                )
            ):
                best = outer
        parents[inner] = best
    return parents


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


def _accept(lo: Fraction, hi: Fraction) -> float | None:
    try:
        f_lo = float(lo)
        f_hi = float(hi)
    except OverflowError:
        raise _fail(
            "The exact fill area is outside the finite binary64 range",
            quantity="fill area",
        ) from None
    if f_lo == f_hi and math.isfinite(f_lo):
        return f_lo
    return None


def _fill_from_locus(locus, area_value: float) -> float:
    engine = _Engine(locus)
    crossings, slack = engine.run()
    h_sq = locus.scale * locus.scale
    if not crossings:
        statistics["simple_fast_path"] += 1
        if slack:
            budget = Fraction(math.ulp(max(area_value, 5e-324))) / 4
            if h_sq * 2 * slack > budget:
                raise _fail(
                    "Absorbed micro blocks exceed the simple-path slack "
                    "budget",
                    quantity="slack",
                )
        return area_value
    loop_records = _decompose(locus, crossings)
    negligible_budget = locus.micro_limit * locus.micro_limit
    for precision in _PHASE_PRECISIONS:
        all_areas = [
            _loop_area(locus, segments, precision)
            for segments in loop_records
        ]
        # A loop whose area enclosure straddles zero at negligible
        # magnitude (a micro loop at enclosure resolution) contributes
        # between zero and its magnitude bound to the fill; it is dropped
        # and its bound joins the slack budget.  A large-magnitude
        # unresolved orientation is a genuine failure.
        records = []
        areas = []
        signs = []
        loop_slack = _ZERO
        ambiguous = False
        for record, area in zip(loop_records, all_areas):
            if area[0] > 0:
                sign = 1
            elif area[1] < 0:
                sign = -1
            else:
                magnitude = max(-area[0], area[1])
                if magnitude <= negligible_budget:
                    loop_slack += magnitude
                    continue
                ambiguous = True
                break
            records.append(record)
            areas.append(area)
            signs.append(sign)
        if ambiguous:
            continue  # escalate the precision ladder
        loop_records_active = records
        parents = _containment_parents(locus, loop_records_active, areas)
        children = [[] for _ in areas]
        for index, parent in enumerate(parents):
            if parent is not None:
                children[parent].append(index)

        def winding_of(index):
            total = signs[index]
            parent = parents[index]
            while parent is not None:
                total += signs[parent]
                parent = parents[parent]
            return total

        def magnitude_of(index):
            area = areas[index]
            return area if signs[index] > 0 else (-area[1], -area[0])

        fill = (_ZERO, _ZERO)
        for index in range(len(areas)):
            if winding_of(index) == 0:
                continue
            face = magnitude_of(index)
            for child in children[index]:
                face = _iv_sub(face, magnitude_of(child))
            fill = _iv_add(fill, face)
        lo = h_sq * (fill[0] - 2 * slack - loop_slack)
        hi = h_sq * (fill[1] + 2 * slack + loop_slack)
        result = _accept(lo, hi)
        if result is not None:
            if result < 0.0:
                raise _fail(
                    "The certified fill enclosure produced a negative value",
                    quantity="fill area",
                    value=result,
                )
            return result
    raise _fail(
        "The fill enclosure could not determine one binary64 rounding "
        "within the precision ladder",
        quantity="fill enclosure",
        bound=f"{_PHASE_PRECISIONS[-1]} bits",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def fill_area_source(
    position_spans, preimages, widths, scale, area_value
) -> float:
    """Nonzero-winding fill area of one closed polynomial PH source.

    ``area_value`` is the already correctly rounded ``abs(signed_area)``
    of the same committed state; it is returned bitwise when the crossing
    engine certifies simplicity.
    """
    statistics["queries"] += 1
    cells = [((_ZERO, _ONE, 1),) for _ in position_spans]
    locus = build_locus(position_spans, preimages, widths, scale, _ZERO, cells)
    return _fill_from_locus(locus, area_value)


def fill_area_offset(provenance, area_value) -> float:
    """Nonzero-winding fill area of one exact closed PH offset."""
    metric = provenance.metric
    spans, scale, distance, _closed = metric.exact_source_state()
    preimages = [
        [complex(float(re), float(im)) for re, im in zip(s.wre, s.wim)]
        for s in spans
    ]
    widths = [float(s.h) for s in spans]
    if distance == 0.0:
        return fill_area_source(
            provenance.position_spans, preimages, widths, float(scale),
            area_value,
        )
    statistics["queries"] += 1
    d_hat = Fraction(distance) / scale
    cell_map = [[] for _ in spans]
    for span_index, a_loc, b_loc, eta in metric.fill_cells():
        cell_map[span_index].append((a_loc, b_loc, eta))
    locus = build_locus(
        provenance.position_spans, preimages, widths, float(scale), d_hat,
        cell_map,
    )
    return _fill_from_locus(locus, area_value)
