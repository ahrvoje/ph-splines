"""Stable exact local arc length and its elementary inverse.

All quantities are in normalized coordinates.  For one PH segment the
parametric speed is the quadratic

    sigma(t) = A t^2 + B t + C,      A = |b|^2, B = 2 Re(conj(a) b), C = |a|^2,

and the local arc length is exactly the cubic

    S(t) = A t^3 / 3 + B t^2 / 2 + C t.

The inverse of ``S`` is computed by the closed-form depressed-cubic
reduction of specification section 13 (hyperbolic form, with a stable
scaled Cardano fallback), followed by safeguarded Newton polishing against
the exact polynomial.  No generic polynomial root finder is ever used.
"""

from __future__ import annotations

import math

from ph_spline._constants import EPS, G_HYPERBOLIC_MIN
from ph_spline.exceptions import ArcLengthInversionError

try:  # pragma: no cover - availability depends on the Python version
    from math import fma as _fma
except ImportError:  # pragma: no cover

    def _fma(x: float, y: float, z: float) -> float:
        return x * y + z


__all__ = [
    "arc_length_at",
    "compensated_prefix_sums",
    "invert_arc_length",
    "speed",
]

#: Relative threshold below which the quadratic coefficient ``A`` is treated
#: as negligible for conditioning purposes (spec section 13.5).  This is a
#: floating-point strategy only; the stored curve is unchanged.
_A_NEGLIGIBLE = 64.0 * EPS
_B_NEGLIGIBLE = 64.0 * EPS


def speed(A: float, B: float, C: float, t: float) -> float:
    """Parametric speed ``sigma(t)`` evaluated by Horner's scheme."""
    return _fma(_fma(A, t, B), t, C)


def arc_length_at(A: float, B: float, C: float, t: float) -> float:
    """Exact local arc length ``S(t)``.

    Uses the fused-multiply-add Horner form.  When the linear coefficient is
    negative and the quadratic coefficient is material, the equivalent
    completed-square form is used instead because the plain Horner form can
    cancel there (spec section 12).
    """
    if B < 0.0 and A > _A_NEGLIGIBLE * max(abs(B), C):
        h = 0.5 * B / A
        g2 = C / A - h * h  # == (|chi| / A)^2 by the PH identity
        g2 = max(g2, 0.0)
        th = t + h
        return A * t * (((th * th + th * h) + h * h) / 3.0 + g2)
    return t * _fma(t, _fma(t, A / 3.0, 0.5 * B), C)


def _depressed_cubic_root(g: float, R: float) -> float:
    """Unique real root of ``y^3 + 3 g^2 y = R`` with ``g >= 0``.

    The equation is scaled so that its principal dimensionless quantities
    are near unit magnitude (spec section 13.4), then solved with the
    hyperbolic closed form.  When the scaled ``G`` is too small for the
    hyperbolic argument, the stable scaled Cardano form is used; the smaller
    Cardano term is obtained through the product identity, never by
    subtracting nearly equal radicals.
    """
    if R == 0.0:
        return 0.0
    q = max(g, math.cbrt(abs(R)))
    if q == 0.0:
        return 0.0
    G = g / q
    Rq = ((R / q) / q) / q
    if G >= G_HYPERBOLIC_MIN:
        arg = Rq / (2.0 * G * G * G)
        Y = 2.0 * G * math.sinh(math.asinh(arg) / 3.0)
    else:
        # After scaling, G < G_HYPERBOLIC_MIN implies |Rq| == 1 up to
        # rounding, so no cancellation can occur in U + V.
        Q = 0.5 * Rq
        Hq = math.hypot(Q, G * G * G)
        if Q >= 0.0:
            U = math.cbrt(Q + Hq)
            V = -(G * G) / U if U != 0.0 else 0.0
        else:
            V = math.cbrt(Q - Hq)
            U = -(G * G) / V if V != 0.0 else 0.0
        Y = U + V
    return q * Y


def _elementary_estimate(A: float, B: float, C: float, chi: float, s: float) -> float:
    """Closed-form elementary estimate of the local inverse (spec 13.2-13.5)."""
    if A > _A_NEGLIGIBLE * max(abs(B), C):
        h = 0.5 * B / A
        g = abs(chi) / A
        three_s_over_A = 3.0 * s / A
        R = (h * h + 3.0 * g * g) * h + three_s_over_A
        y = _depressed_cubic_root(g, R)
        # Cancellation-resistant recovery of t (spec 13.3): never y - h.
        denom = (y * y + y * h) + (h * h + 3.0 * g * g)
        if denom > 0.0 and math.isfinite(denom):
            return three_s_over_A / denom
        return 0.5
    if abs(B) <= _B_NEGLIGIBLE * C:
        # Nearly constant speed: linear initializer.
        return s / C
    # Stable quadratic initializer for (B/2) t^2 + C t = s.
    disc = _fma(C, C, 2.0 * B * s)
    disc = max(disc, 0.0)
    denom = C + math.sqrt(disc)
    if denom > 0.0:
        return 2.0 * s / denom
    return s / C


def _polish(A: float, B: float, C: float, length: float, s: float, t0: float) -> float:
    """Safeguarded Newton correction against the exact cubic (spec 13.6)."""
    tol = 64.0 * EPS * length + 4.0 * math.ulp(s)
    t = min(max(t0, 0.0), 1.0)
    f = arc_length_at(A, B, C, t) - s
    if abs(f) <= tol:
        return t
    lo, hi = 0.0, 1.0
    for _ in range(67):
        if f < 0.0:
            lo = t
        else:
            hi = t
        sigma = speed(A, B, C, t)
        if sigma > 0.0:
            t_new = t - f / sigma
        else:
            t_new = math.nan
        if not (math.isfinite(t_new) and lo < t_new < hi):
            t_new = 0.5 * (lo + hi)
        f_new = arc_length_at(A, B, C, t_new) - s
        if abs(f_new) <= tol:
            return t_new
        if abs(f_new) >= abs(f):
            # The Newton proposal increased the residual: fall back to the
            # midpoint of the updated bracket.
            if f_new < 0.0:
                lo = t_new
            else:
                hi = t_new
            t_new = 0.5 * (lo + hi)
            f_new = arc_length_at(A, B, C, t_new) - s
            if abs(f_new) <= tol:
                return t_new
        t, f = t_new, f_new
    raise ArcLengthInversionError(
        "Safeguarded arc-length inversion did not reach its residual bound",
        quantity="|S(t) - s|",
        value=abs(f),
        bound=tol,
    )


def _invert_forward(
    A: float, B: float, C: float, chi: float, length: float, s: float
) -> float:
    t0 = _elementary_estimate(A, B, C, chi, s)
    return _polish(A, B, C, length, s, t0)


def invert_arc_length(
    A: float, B: float, C: float, chi: float, length: float, s: float
) -> float:
    """Unique ``t`` in ``[0, 1]`` with ``S(t) = s`` for ``s`` in ``[0, length]``.

    Targets closer to the segment end than its start are inverted from the
    reversed end using the reversed preimage invariants, which preserves
    relative accuracy near ``t = 1`` (spec section 13.3).
    """
    if s <= 0.0:
        return 0.0
    if s >= length:
        return 1.0
    if s > 0.5 * length:
        t_rev = _invert_forward(
            A, -(2.0 * A + B), (A + B) + C, -chi, length, length - s
        )
        return 1.0 - t_rev
    return _invert_forward(A, B, C, chi, length, s)


def compensated_prefix_sums(lengths: list[float]) -> list[float]:
    """Neumaier-compensated prefix sums ``[0, L0, L0+L1, ...]``."""
    prefix = [0.0]
    total = 0.0
    comp = 0.0
    for value in lengths:
        t = total + value
        if abs(total) >= abs(value):
            comp += (total - t) + value
        else:
            comp += (value - t) + total
        total = t
        prefix.append(total + comp)
    return prefix
