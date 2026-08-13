"""Double-double (compensated binary64 pair) arithmetic for offset metrics.

A double-double value is an unevaluated sum ``hi + lo`` of two binary64
numbers with ``|lo| <= ulp(hi) / 2``.  It carries about 106 significant
bits.  This module provides the error-free transformations, the standard
arithmetic kernels, and a quadrant-aware ``atan2`` whose absolute error is
bounded and verified against the exact integer evaluator in
:mod:`ph_spline.exact_real`.

Documented worst-case relative error bounds (in units of ``2**-104``, from
the standard double-word analyses of Joldes, Muller and Popescu, rounded up
to lax powers of two):

- ``dd_add``     : 4
- ``dd_mul``     : 8
- ``dd_div``     : 16
- ``dd_atan2``   : absolute error ``<= ATAN2_DD_ABS_ERR`` for arguments in
  the certified table range (see below)

The offset-metric fast path composes these bounds into per-cell error
certificates; whenever a certificate cannot decide the requested public
rounding the caller escalates to the exact integer path, so no bound here
is trusted beyond its role as a fast-path filter.

All functions accept and return plain ``(hi, lo)`` tuples.  Hot paths in
:mod:`ph_spline.offset_metric` inline the error-free transformations.
"""

from __future__ import annotations

from math import fma

__all__ = [
    "ATAN2_DD_ABS_ERR",
    "DD_EPS",
    "PI_DD",
    "PI_2_DD",
    "dd_abs",
    "dd_add",
    "dd_atan2",
    "dd_div",
    "dd_horner",
    "dd_mul",
    "dd_mul_f",
    "dd_sub",
    "two_prod",
    "two_sum",
]

#: Unit roundoff of one double-double operation (documented, conservative).
DD_EPS: float = 2.0**-104

#: pi as a double-double (error about 3e-33, i.e. 2**-108 relative).
PI_DD: tuple[float, float] = (3.141592653589793, 1.2246467991473532e-16)

#: pi / 2 as a double-double (halving is exact in binary64).
PI_2_DD: tuple[float, float] = (1.5707963267948966, 6.123233995736766e-17)

#: Documented absolute error bound of :func:`dd_atan2` (see the module
#: docstring); verified against the exact integer evaluator by the test
#: suite over deterministic and adversarial inputs.
ATAN2_DD_ABS_ERR: float = 2.0**-98


# ---------------------------------------------------------------------------
# Error-free transformations
# ---------------------------------------------------------------------------


def two_sum(a: float, b: float) -> tuple[float, float]:
    """Knuth two-sum: exact ``a + b = s + e``."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def two_prod(a: float, b: float) -> tuple[float, float]:
    """FMA two-product: exact ``a * b = p + e``."""
    p = a * b
    return p, fma(a, b, -p)


def _quick_two_sum(a: float, b: float) -> tuple[float, float]:
    """Renormalization two-sum, valid for ``|a| >= |b|`` or ``a == 0``."""
    s = a + b
    return s, b - (s - a)


# ---------------------------------------------------------------------------
# Double-double kernels
# ---------------------------------------------------------------------------


def dd_abs(a: tuple[float, float]) -> tuple[float, float]:
    return a if (a[0] > 0.0 or (a[0] == 0.0 and a[1] >= 0.0)) else (-a[0], -a[1])


def dd_add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Accurate double-double addition (relative error ``<= 4 * DD_EPS``)."""
    s1, s2 = two_sum(a[0], b[0])
    t1, t2 = two_sum(a[1], b[1])
    s2 += t1
    s1, s2 = _quick_two_sum(s1, s2)
    s2 += t2
    return _quick_two_sum(s1, s2)


def dd_sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return dd_add(a, (-b[0], -b[1]))


def dd_mul(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Double-double product (relative error ``<= 8 * DD_EPS``)."""
    p = a[0] * b[0]
    e = fma(a[0], b[0], -p)
    e += a[0] * b[1] + a[1] * b[0]
    return _quick_two_sum(p, e)


def dd_mul_f(a: tuple[float, float], b: float) -> tuple[float, float]:
    p = a[0] * b
    e = fma(a[0], b, -p)
    e += a[1] * b
    return _quick_two_sum(p, e)


def dd_div(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Double-double quotient (relative error ``<= 16 * DD_EPS``)."""
    q1 = a[0] / b[0]
    # r = a - q1 * b, computed in double-double.
    p = q1 * b[0]
    e = fma(q1, b[0], -p)
    e += q1 * b[1]
    r = dd_add(a, (-p, -e))
    q2 = (r[0] + r[1]) / (b[0] + b[1])
    return _quick_two_sum(q1, q2)


def dd_horner(
    coeffs: tuple[tuple[float, float], ...], x: tuple[float, float]
) -> tuple[float, float]:
    """Compensated Horner evaluation of a power polynomial at ``x``.

    ``coeffs`` are double-double power coefficients ordered from the
    highest degree down to the constant term.  Relative error of each
    fused step is ``<= 12 * DD_EPS``; the caller composes the complete
    bound from the stored coefficient-magnitude sum.
    """
    xh, xl = x
    rh, rl = coeffs[0]
    for ch, cl in coeffs[1:]:
        # r = r * x
        p = rh * xh
        e = fma(rh, xh, -p)
        e += rh * xl + rl * xh
        rh = p + e
        rl = e - (rh - p)
        # r = r + c
        s1, s2 = two_sum(rh, ch)
        t1, t2 = two_sum(rl, cl)
        s2 += t1
        s = s1 + s2
        s2 = s2 - (s - s1)
        s2 += t2
        rh = s + s2
        rl = s2 - (rh - s)
    return rh, rl


# ---------------------------------------------------------------------------
# atan2 in double-double
# ---------------------------------------------------------------------------

#: Table granularity: nodes ``k / _ATAN_TABLE_N`` on ``[0, 1]``.
_ATAN_TABLE_N = 64

_ATAN_TABLE: list[tuple[float, float]] | None = None


def _build_atan_table() -> list[tuple[float, float]]:
    """Build ``atan(k / 64)`` nodes as double-doubles.

    The nodes are computed by the exact integer evaluator of
    :mod:`ph_spline.exact_real` at 160 fractional bits, so every stored
    pair encloses its exact node value to well below ``DD_EPS``.
    """
    from fractions import Fraction

    from ph_spline.exact_real import atan_ball

    table: list[tuple[float, float]] = []
    prec = 160
    scale = Fraction(1, 1 << prec)
    for k in range(_ATAN_TABLE_N + 1):
        value, err = atan_ball(Fraction(k, _ATAN_TABLE_N), prec)
        exact = Fraction(value, 1 << prec)
        hi = float(exact)
        lo = float(exact - Fraction(hi))
        assert err * scale < Fraction(1, 1 << 140)
        table.append((hi, lo))
    return table


def _atan_reduced(eps: tuple[float, float]) -> tuple[float, float]:
    """Taylor ``atan`` for ``|eps| <= 2**-6`` with error ``< 2**-104``.

    Terms through ``eps**15`` leave a truncation remainder below
    ``|eps|**17 / 17 <= 2**-106 |eps|``.
    """
    e2 = dd_mul(eps, eps)
    # Horner in eps^2 over 1 - x/3 + x^2/5 - ... +- x^7/15.
    acc = (-0.06666666666666667, -9.251858538542971e-19)  # -1/15
    for c in (
        (0.07692307692307693, -4.270088556250602e-18),  # 1/13
        (-0.09090909090909091, 2.523234146875356e-18),  # -1/11
        (0.1111111111111111, 6.1679056923619804e-18),  # 1/9
        (-0.14285714285714285, -7.93016446160826e-18),  # -1/7
        (0.2, -1.1102230246251566e-17),  # 1/5
        (-0.3333333333333333, -1.850371707708594e-17),  # -1/3
        (1.0, 0.0),
    ):
        acc = dd_add(dd_mul(acc, e2), c)
    return dd_mul(eps, acc)


def dd_atan2(y: tuple[float, float], x: tuple[float, float]) -> tuple[float, float]:
    """Quadrant-aware arctangent of double-double arguments.

    Returns the principal value in ``(-pi, pi]`` with absolute error
    bounded by ``ATAN2_DD_ABS_ERR`` (documented; verified in the test
    suite against the exact integer evaluator).  Exact axis inputs return
    the exact axis results ``0``, ``pi/2``, ``-pi/2`` and the
    double-double ``pi``.  The pair ``(0, 0)`` is a caller error and
    raises ``ZeroDivisionError``.
    """
    global _ATAN_TABLE
    yh = y[0] + y[1]
    xh = x[0] + x[1]
    if yh == 0.0 and y[0] == 0.0:
        if xh > 0.0:
            return (0.0, 0.0)
        if xh < 0.0:
            return PI_DD
        raise ZeroDivisionError("atan2(0, 0) is undefined")
    if xh == 0.0 and x[0] == 0.0:
        return PI_2_DD if yh > 0.0 else (-PI_2_DD[0], -PI_2_DD[1])

    if _ATAN_TABLE is None:
        _ATAN_TABLE = _build_atan_table()

    ax = dd_abs(x)
    ay = dd_abs(y)
    swap = ay[0] > ax[0] or (ay[0] == ax[0] and ay[1] > ax[1])
    if swap:
        num, den = ax, ay
    else:
        num, den = ay, ax
    r = dd_div(num, den)  # in [0, 1]
    k = int(r[0] * _ATAN_TABLE_N + 0.5)
    if k < 0:
        k = 0
    elif k > _ATAN_TABLE_N:
        k = _ATAN_TABLE_N
    t = k / _ATAN_TABLE_N  # exact binary64 (k / 64)
    if k == 0:
        base = _atan_reduced(r)
    else:
        # atan(r) = atan(t) + atan((num - t*den) / (den + t*num))
        rn = dd_add(num, dd_mul_f(den, -t))
        rd = dd_add(den, dd_mul_f(num, t))
        base = dd_add(_ATAN_TABLE[k], _atan_reduced(dd_div(rn, rd)))
    if swap:
        base = dd_sub(PI_2_DD, base)
    if xh < 0.0:
        base = dd_sub(PI_DD, base)
    if yh < 0.0 or (yh == 0.0 and y[0] < 0.0):
        base = (-base[0], -base[1])
    return base


