"""Exact rational Bernstein algebra and certified real analysis kernels.

Everything in this module operates on exact values.  Binary64 inputs are
dyadic rationals, so every polynomial coefficient operation below is free
of rounding error, and every transcendental value is produced as a scaled
integer *ball* ``(v, e)`` at precision ``p``: the exact real lies within
``[(v - e) / 2**p, (v + e) / 2**p]``.

The module supplies:

- exact Bernstein products, degree elevation, splitting, restriction,
  evaluation, and power-basis conversion over :class:`fractions.Fraction`;
- complete certified real-root isolation on ``[0, 1]`` (Descartes /
  de Casteljau bisection with an exact square-free fallback via Yun's
  algorithm), including multiplicities;
- adaptive-precision ``atan`` / ``atan2`` / ``pi`` enclosures used by the
  certified offset-distance path and by the construction-time metric
  verification.

None of these routines is on the ordinary query fast path; they run at
offset construction and inside rare certified fallbacks.
"""

from __future__ import annotations

import math
from fractions import Fraction
from math import comb, isqrt

from ph_spline.exceptions import ResourceLimitError

__all__ = [
    "MAX_ISOLATION_DEPTH",
    "atan2_ball",
    "atan_ball",
    "bern_antiderivative",
    "bern_elevate",
    "bern_eval",
    "bern_product",
    "bern_restrict",
    "bern_split",
    "bern_to_power",
    "isolate_bernstein_roots",
    "pi_ball",
    "power_eval",
    "power_to_bern",
    "refine_root_to_floats",
]

#: Hard bisection-depth bound of the primary Descartes isolation pass.
MAX_ISOLATION_DEPTH = 72

#: Hard bisection-depth bound after the exact square-free fallback.
MAX_SQUAREFREE_DEPTH = 1024

#: Hard bound on root-refinement bisection steps.
MAX_REFINE_STEPS = 220

_HALF = Fraction(1, 2)
_ZERO = Fraction(0)


# ---------------------------------------------------------------------------
# Exact Bernstein algebra
# ---------------------------------------------------------------------------


def bern_product(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    """Exact Bernstein coefficients of the product of two polynomials."""
    na = len(a) - 1
    nb = len(b) - 1
    n = na + nb
    out = []
    for k in range(n + 1):
        acc = _ZERO
        for i in range(max(0, k - nb), min(na, k) + 1):
            acc += comb(na, i) * comb(nb, k - i) * a[i] * b[k - i]
        out.append(acc / comb(n, k))
    return out


def bern_elevate(c: list[Fraction], r: int) -> list[Fraction]:
    """Exact degree elevation by ``r``."""
    n = len(c) - 1
    m = n + r
    out = []
    for k in range(m + 1):
        acc = _ZERO
        for j in range(max(0, k - r), min(n, k) + 1):
            acc += comb(n, j) * comb(r, k - j) * c[j]
        out.append(acc / comb(m, k))
    return out


def bern_split(
    c: list[Fraction], t: Fraction
) -> tuple[list[Fraction], list[Fraction]]:
    """Exact de Casteljau split at ``t``; returns (left, right) controls."""
    n = len(c) - 1
    work = list(c)
    left = [work[0]]
    right = [work[n]]
    s = 1 - t
    for level in range(1, n + 1):
        work = [s * work[i] + t * work[i + 1] for i in range(len(work) - 1)]
        left.append(work[0])
        right.append(work[-1])
    right.reverse()
    return left, right


def bern_restrict(
    c: list[Fraction], a: Fraction, b: Fraction
) -> list[Fraction]:
    """Exact restriction of Bernstein controls to ``[a, b]`` in ``[0, 1]``."""
    if not 0 <= a < b <= 1:
        raise ValueError("restriction interval must satisfy 0 <= a < b <= 1")
    if a == 0:
        if b == 1:
            return list(c)
        return bern_split(c, b)[0]
    right = bern_split(c, a)[1]
    if b == 1:
        return right
    local_b = (b - a) / (1 - a)
    return bern_split(right, local_b)[0]


def bern_eval(c: list[Fraction], t: Fraction) -> Fraction:
    """Exact de Casteljau evaluation."""
    work = list(c)
    s = 1 - t
    while len(work) > 1:
        work = [s * work[i] + t * work[i + 1] for i in range(len(work) - 1)]
    return work[0]


def bern_antiderivative(c: list[Fraction]) -> list[Fraction]:
    """Forward antiderivative controls: ``f_0 = 0``, ``f_{j+1} = f_j + c_j/(n+1)``."""
    n = len(c) - 1
    out = [_ZERO]
    for j in range(n + 1):
        out.append(out[-1] + c[j] / (n + 1))
    return out


def bern_to_power(c: list[Fraction]) -> list[Fraction]:
    """Exact power coefficients ``p_k`` with ``p(x) = sum p_k x^k``."""
    n = len(c) - 1
    out = []
    for k in range(n + 1):
        delta = _ZERO
        for j in range(k + 1):
            term = comb(k, j) * c[j]
            delta += term if (k - j) % 2 == 0 else -term
        out.append(comb(n, k) * delta)
    return out


def power_to_bern(p: list[Fraction]) -> list[Fraction]:
    """Exact Bernstein controls of a power polynomial on ``[0, 1]``."""
    n = len(p) - 1
    out = []
    for j in range(n + 1):
        acc = _ZERO
        for k in range(j + 1):
            acc += Fraction(comb(j, k), comb(n, k)) * p[k]
        out.append(acc)
    return out


def power_eval(p: list[Fraction], x: Fraction) -> Fraction:
    acc = _ZERO
    for c in reversed(p):
        acc = acc * x + c
    return acc


def power_mul(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    out = [_ZERO] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
    return out


def power_derivative(p: list[Fraction]) -> list[Fraction]:
    if len(p) <= 1:
        return [_ZERO]
    return [k * p[k] for k in range(1, len(p))]


# ---------------------------------------------------------------------------
# Exact integer polynomial helpers (square-free machinery)
# ---------------------------------------------------------------------------


def _to_int_poly(p: list[Fraction]) -> list[int]:
    """Scale a rational polynomial by a positive constant to integers."""
    while p and p[-1] == 0:
        p = p[:-1]
    if not p:
        return []
    lcm = 1
    for c in p:
        lcm = lcm * c.denominator // math.gcd(lcm, c.denominator)
    out = [int(c * lcm) for c in p]
    content = 0
    for c in out:
        content = math.gcd(content, abs(c))
    if content > 1:
        out = [c // content for c in out]
    return out


def _int_poly_gcd(a: list[int], b: list[int]) -> list[int]:
    """Primitive-PRS polynomial GCD of two integer polynomials."""

    def prem(f: list[int], g: list[int]) -> list[int]:
        f = list(f)
        dg = len(g) - 1
        lg = g[-1]
        while len(f) - 1 >= dg and any(f):
            df = len(f) - 1
            if f[-1] == 0:
                f.pop()
                continue
            lf = f[-1]
            f = [c * lg for c in f]
            shift = df - dg
            for i in range(dg + 1):
                f[shift + i] -= lf * g[i]
            while f and f[-1] == 0:
                f.pop()
        return f

    def primitive(f: list[int]) -> list[int]:
        content = 0
        for c in f:
            content = math.gcd(content, abs(c))
        return [c // content for c in f] if content > 1 else list(f)

    a = primitive([c for c in a])
    b = primitive([c for c in b])
    if len(a) < len(b):
        a, b = b, a
    while b:
        r = prem(a, b)
        if not any(r):
            return primitive(b)
        a, b = b, primitive(r)
    return primitive(a)


def _int_poly_div_exact(a: list[int], b: list[int]) -> list[Fraction]:
    """Exact division ``a / b`` (must divide) returning rational coefficients."""
    fa = [Fraction(c) for c in a]
    fb = [Fraction(c) for c in b]
    out = [_ZERO] * (len(fa) - len(fb) + 1)
    while len(fa) >= len(fb) and any(fa):
        while fa and fa[-1] == 0:
            fa.pop()
        if len(fa) < len(fb):
            break
        k = len(fa) - len(fb)
        q = fa[-1] / fb[-1]
        out[k] = q
        for i in range(len(fb)):
            fa[k + i] -= q * fb[i]
        fa.pop()
    if any(fa):
        raise ArithmeticError("exact polynomial division failed")
    return out


def squarefree_decomposition(
    p: list[Fraction],
) -> list[tuple[list[Fraction], int]]:
    """Yun's algorithm: return ``[(factor, multiplicity), ...]``.

    Every returned factor is square-free, factors are pairwise coprime,
    and ``prod factor^multiplicity`` equals ``p`` up to a constant.
    """
    ip = _to_int_poly(p)
    if len(ip) <= 1:
        return []
    d = [k * ip[k] for k in range(1, len(ip))]
    g = _int_poly_gcd(ip, d)
    if len(g) == 1:
        return [([Fraction(c) for c in ip], 1)]
    out: list[tuple[list[Fraction], int]] = []
    c = _int_poly_div_exact(ip, g)
    dd_ = _int_poly_div_exact(d, g)
    i = 1
    while True:
        cd = power_derivative(c)
        y = [dd_[k] - (cd[k] if k < len(cd) else _ZERO) for k in range(len(dd_))]
        while y and y[-1] == 0:
            y.pop()
        ic = _to_int_poly(c)
        iy = _to_int_poly(y) if y else []
        if not iy:
            if len(ic) > 1:
                out.append(([Fraction(v) for v in ic], i))
            break
        f = _int_poly_gcd(ic, iy)
        if len(f) > 1:
            out.append(([Fraction(v) for v in f], i))
        c = _int_poly_div_exact(ic, f)
        dd_ = _int_poly_div_exact(iy, f)
        i += 1
        if not any(c[1:]):
            break
    return out


# ---------------------------------------------------------------------------
# Certified real-root isolation on [0, 1]
# ---------------------------------------------------------------------------


class _IsolationStall(Exception):
    """Primary Descartes bisection could not separate a root cluster."""


def _variations(c: list[Fraction]) -> int:
    count = 0
    last = 0
    for v in c:
        if v == 0:
            continue
        s = 1 if v > 0 else -1
        if last and s != last:
            count += 1
        last = s
    return count


def _deflate_left(c: list[Fraction]) -> tuple[int, list[Fraction]]:
    """Strip a root at ``t = 0``: return (multiplicity, deflated controls)."""
    n = len(c) - 1
    k = 0
    while k <= n and c[k] == 0:
        k += 1
    if k == 0:
        return 0, c
    out = [
        c[i + k] * Fraction(comb(n, i + k), comb(n - k, i))
        for i in range(n - k + 1)
    ]
    return k, out


def _deflate_right(c: list[Fraction]) -> tuple[int, list[Fraction]]:
    """Strip a root at ``t = 1``: return (multiplicity, deflated controls)."""
    n = len(c) - 1
    k = 0
    while k <= n and c[n - k] == 0:
        k += 1
    if k == 0:
        return 0, c
    out = [
        c[i] * Fraction(comb(n, i), comb(n - k, i)) for i in range(n - k + 1)
    ]
    return k, out


def _isolate_recursive(
    c: list[Fraction],
    lo: Fraction,
    hi: Fraction,
    depth: int,
    max_depth: int,
    out: list[tuple[Fraction, Fraction, int]],
) -> None:
    v = _variations(c)
    if v == 0:
        return
    if v == 1:
        # One sign variation certifies exactly one root, of multiplicity
        # one, and the endpoint values (first and last controls) have
        # opposite signs, so the interval is a refinable bracket.
        out.append((lo, hi, 1))
        return
    if depth >= max_depth:
        raise _IsolationStall
    mid = lo + (hi - lo) / 2
    left, right = bern_split(c, _HALF)
    if left[-1] == 0:
        kl, left = _deflate_right(left)
        kr, right = _deflate_left(right)
        if kl != kr:
            raise AssertionError("inconsistent midpoint multiplicity")
        out.append((mid, mid, kl))
    _isolate_recursive(left, lo, mid, depth + 1, max_depth, out)
    _isolate_recursive(right, mid, hi, depth + 1, max_depth, out)


def isolate_bernstein_roots(
    c: list[Fraction],
) -> tuple[int, int, list[tuple[Fraction, Fraction, int, list[Fraction] | None]]]:
    """Complete certified real-root isolation on ``[0, 1]``.

    Returns ``(mult_at_0, mult_at_1, interior)`` where ``interior`` is an
    ordered list of ``(lo, hi, multiplicity, refiner)`` records with
    dyadic endpoints.  ``lo == hi`` marks an exact dyadic root (``refiner``
    is ``None``); otherwise the open interval ``(lo, hi)`` contains
    exactly one distinct real root of the stated multiplicity, the
    complement of all records contains no root, and ``refiner`` is a
    Bernstein polynomial on ``[0, 1]`` with certified opposite signs at
    ``lo`` and ``hi`` whose unique root in the bracket is the same root
    (the polynomial itself for an odd-multiplicity root of the primary
    pass, the square-free factor otherwise).  The zero polynomial is
    rejected.

    The primary pass is exact Descartes/de Casteljau bisection; a stalled
    cluster (for example an even-multiplicity tangency) triggers the
    exact square-free fallback, which terminates for every nonzero input.
    """
    if all(v == 0 for v in c):
        raise ValueError("cannot isolate roots of the zero polynomial")
    m0, c0 = _deflate_left(c)
    m1, c1 = _deflate_right(c0)
    if len(c1) == 1:
        return m0, m1, []
    interior: list[tuple[Fraction, Fraction, int]] = []
    try:
        _isolate_recursive(
            c1, _ZERO, Fraction(1), 0, MAX_ISOLATION_DEPTH, interior
        )
        interior.sort(key=lambda r: r[0])
        return m0, m1, [
            (lo, hi, m, None if lo == hi else c1) for lo, hi, m in interior
        ]
    except _IsolationStall:
        pass
    # Exact square-free fallback.
    power = bern_to_power(c1)
    records: list[list] = []  # [lo, hi, multiplicity, factor_bernstein]
    for factor, mult in squarefree_decomposition(power):
        if len(factor) <= 1:
            continue
        bern_f = power_to_bern(factor)
        f0, f1, roots_f = _squarefree_isolate(bern_f)
        if f0 or f1:
            raise AssertionError("deflated polynomial re-grew endpoint roots")
        for lo, hi, _ in roots_f:
            records.append([lo, hi, mult, bern_f])
    records.sort(key=lambda r: (r[0], r[1]))
    _separate_records(records)
    return m0, m1, [
        (r[0], r[1], r[2], None if r[0] == r[1] else r[3]) for r in records
    ]


def _squarefree_isolate(
    c: list[Fraction],
) -> tuple[int, int, list[tuple[Fraction, Fraction, int]]]:
    m0, c0 = _deflate_left(c)
    m1, c1 = _deflate_right(c0)
    out: list[tuple[Fraction, Fraction, int]] = []
    if len(c1) > 1:
        try:
            _isolate_recursive(
                c1, _ZERO, Fraction(1), 0, MAX_SQUAREFREE_DEPTH, out
            )
        except _IsolationStall as exc:
            raise ResourceLimitError(
                "Certified root isolation exceeded its hard bisection depth",
                operation="offset-metric",
                quantity="square-free bisection depth",
                value=MAX_SQUAREFREE_DEPTH,
            ) from exc
    return m0, m1, out


def _halve_record(rec: list) -> None:
    """Halve one open sign-change bracket against its own factor, in place."""
    lo, hi, factor = rec[0], rec[1], rec[3]
    if lo == hi:
        return  # exact dyadic root; nothing to shrink
    mid = lo + (hi - lo) / 2
    s_mid = sign_at(factor, mid)
    if s_mid == 0:
        rec[0] = rec[1] = mid
        return
    if s_mid == sign_at(factor, lo):
        rec[0] = mid
    else:
        rec[1] = mid


def _separate_records(records: list[list]) -> None:
    """Refine factor-tagged isolating records until pairwise disjoint.

    The records hold distinct real roots (Yun factors are pairwise
    coprime), so repeated halving of the overlapping brackets against
    their own factors terminates; the hard budget converts an unexpected
    non-termination into a typed failure.
    """
    for _ in range(MAX_REFINE_STEPS):
        records.sort(key=lambda r: (r[0], r[1]))
        overlap = False
        for i in range(len(records) - 1):
            if records[i][1] > records[i + 1][0] or (
                records[i][1] == records[i + 1][0]
                and records[i][0] == records[i][1] == records[i + 1][1]
            ):
                overlap = True
                _halve_record(records[i])
                _halve_record(records[i + 1])
        if not overlap:
            return
    raise ResourceLimitError(
        "Isolating intervals from square-free factors could not be "
        "separated within the refinement budget",
        operation="offset-metric",
        quantity="interval separation steps",
        value=MAX_REFINE_STEPS,
    )


def sign_at(c: list[Fraction], t: Fraction) -> int:
    """Exact sign of a Bernstein polynomial at ``t``."""
    v = bern_eval(c, t)
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def refine_root_to_floats(
    c: list[Fraction], lo: Fraction, hi: Fraction
) -> tuple[float, float]:
    """Refine one certified sign-change bracket to adjacent binary64 floats.

    ``c`` must have opposite nonzero signs at ``lo`` and ``hi``.  Returns
    ``(f_lo, f_hi)`` with ``f_lo <= root <= f_hi`` and ``f_hi`` equal to
    or adjacent to ``f_lo``; both endpoint signs are certified by exact
    evaluation.  If the bracket encloses an exactly representable root the
    two floats are equal.
    """
    s_lo = sign_at(c, lo)
    s_hi = sign_at(c, hi)
    if s_lo == 0:
        f = float(lo)
        return f, f
    if s_hi == 0:
        f = float(hi)
        return f, f
    if s_lo == s_hi:
        raise ValueError("bracket endpoints must have opposite signs")
    for _ in range(MAX_REFINE_STEPS):
        f_lo = float(lo)
        f_hi = float(hi)
        if f_lo == f_hi or math.nextafter(f_lo, math.inf) >= f_hi:
            return f_lo, f_hi
        mid = lo + (hi - lo) / 2
        s_mid = sign_at(c, mid)
        if s_mid == 0:
            f = float(mid)
            return f, f
        if s_mid == s_lo:
            lo = mid
        else:
            hi = mid
    raise ResourceLimitError(
        "Root refinement exceeded its hard bisection budget",
        operation="offset-metric",
        quantity="refinement steps",
        value=MAX_REFINE_STEPS,
    )


# ---------------------------------------------------------------------------
# Scaled-integer ball arithmetic and certified transcendentals
# ---------------------------------------------------------------------------

_PI_CACHE: dict[int, tuple[int, int]] = {}


def _atan_inv_int(x: int, p: int) -> tuple[int, int]:
    """Ball for ``atan(1/x) * 2**p`` with ``x >= 2`` an integer."""
    total = 0
    xx = x * x
    power = x
    k = 0
    terms = 0
    while True:
        term = (1 << p) // ((2 * k + 1) * power)
        if term == 0:
            break
        total += term if k % 2 == 0 else -term
        power *= xx
        k += 1
        terms += 1
    return total, terms + 1


def pi_ball(p: int) -> tuple[int, int]:
    """Certified ball for ``pi * 2**p`` (Machin's formula)."""
    cached = _PI_CACHE.get(p)
    if cached is not None:
        return cached
    a5, e5 = _atan_inv_int(5, p + 8)
    a239, e239 = _atan_inv_int(239, p + 8)
    v = 16 * a5 - 4 * a239
    e = 16 * e5 + 4 * e239
    out = (v >> 8, (e >> 8) + 2)
    _PI_CACHE[p] = out
    return out


def _ball_sqrt(v: int, e: int, p: int) -> tuple[int, int]:
    """Ball square root at scale ``2**-p`` for ``v - e > 0``."""
    s = isqrt(v << p)
    # |s - sqrt(v * 2**p)| <= 1; input error propagates by e / (2 sqrt(v)).
    prop = (e << p) // (2 * s) + 2 if s else e + 2
    return s, prop


def atan_ball(x: Fraction, p: int) -> tuple[int, int]:
    """Certified ball for ``atan(x) * 2**p`` with ``0 <= x <= 1``.

    Sixteen cotangent half-angle reductions shrink the argument below
    ``2**-16``; the alternating Taylor series then converges at 32 bits
    per term, and its truncation error is bounded by the first omitted
    term.  All operations are directed integer arithmetic with explicit
    error counters.
    """
    if not 0 <= x <= 1:
        raise ValueError("atan_ball requires x in [0, 1]")
    if x == 0:
        return 0, 0
    g = p + 48  # guard bits
    one = 1 << g
    v = (x.numerator << g) // x.denominator
    e = 1
    for _ in range(16):
        # x <- x / (1 + sqrt(1 + x^2))
        sq = (v * v) >> g
        sq_e = ((2 * v * e) >> g) + e * e + 2
        s, s_e = _ball_sqrt(one + sq, sq_e, g)
        den = one + s
        den_e = s_e
        v_new = (v << g) // den
        e = ((e << g) + v_new * den_e) // den + 2
        v = v_new
    # Taylor: atan(t) = t - t^3/3 + t^5/5 - ...  with 0 <= t < 2**-16, so
    # successive terms shrink by more than 2**-32 and ``g // 32 + 4`` terms
    # push the first omitted term below one ulp of the guard scale.
    t2 = (v * v) >> g
    t2_e = ((2 * v * e) >> g) + e * e + 2
    total = v
    err = e + 1
    term = v
    term_e = e
    for k in range(1, g // 32 + 5):
        term = (term * t2) >> g
        term_e = ((term_e * t2 + t2_e * term) >> g) + term_e * t2_e + 3
        contrib = term // (2 * k + 1)
        if k % 2 == 1:
            total -= contrib
        else:
            total += contrib
        err += term_e // (2 * k + 1) + 1
        if term == 0:
            break
    # Truncation: the alternating remainder is bounded by the first omitted
    # term, itself bounded by the last computed term's ball.
    err += term + term_e + 1
    total <<= 16  # undo the sixteen half-angle reductions
    err <<= 16
    shift = g - p
    return total >> shift, (err >> shift) + 2


def atan2_ball(y: Fraction, x: Fraction, p: int) -> tuple[int, int]:
    """Certified ball for the principal ``atan2(y, x) * 2**p``.

    The quadrant and octant decisions are exact rational comparisons, so
    the branch structure is decided without rounding; only the magnitude
    of the reduced arctangent carries a ball error.
    """
    if y == 0:
        if x > 0:
            return 0, 0
        if x < 0:
            return pi_ball(p)
        raise ZeroDivisionError("atan2(0, 0) is undefined")
    if x == 0:
        # pi_ball(p - 1) is the integer (pi / 2) * 2**p.
        v, e = pi_ball(p - 1)
        return (v, e + 1) if y > 0 else (-v, e + 1)
    ax = abs(x)
    ay = abs(y)
    if ay <= ax:
        v, e = atan_ball(ay / ax, p)
    else:
        # atan2 = pi/2 - atan(ax / ay); pi_ball(p - 1) is (pi/2) * 2**p.
        b, be = atan_ball(ax / ay, p)
        hp, hpe = pi_ball(p - 1)
        v = hp - b
        e = hpe + be + 1
    if x < 0:
        pv, pe = pi_ball(p)
        v = pv - v
        e = pe + e
    if y < 0:
        v = -v
    return v, e


def ball_to_fraction_bounds(v: int, e: int, p: int) -> tuple[Fraction, Fraction]:
    """Lower and upper rational bounds of a ball."""
    scale = Fraction(1, 1 << p)
    return (v - e) * scale, (v + e) * scale
