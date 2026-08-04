"""Immutable cubic PH segment representation (specification section 4).

Each segment is stored in both cubic Bezier form (for de Casteljau point
evaluation) and PH preimage form (for tangents, curvature and arc length).
All stored quantities are in normalized coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cubic_ph_spline._constants import EPS
from cubic_ph_spline.arclength import arc_length_at, invert_arc_length, speed

__all__ = ["PHSegment"]


@dataclass(frozen=True, slots=True)
class PHSegment:
    """One regular cubic PH segment in normalized coordinates.

    Attributes
    ----------
    index:
        Index in the assembled segment list (which includes auxiliary spans).
    w0, w1:
        Endpoint preimages; ``z'(t) = ((1 - t) w0 + t w1)^2``.
    a, b:
        ``a = w0`` and ``b = w1 - w0`` so that ``w(t) = a + b t``.
    A, B, C, chi:
        Stored scalar invariants ``A = |b|^2``, ``B = 2 Re(conj(a) b)``,
        ``C = |a|^2`` and ``chi = Im(conj(a) b)``.
    ctrl:
        Read-only ``(4, 2)`` Bezier control net, endpoint-snapped to the
        normalized input points.
    length:
        Exact normalized segment arc length ``S(1)``.
    """

    index: int
    w0: complex
    w1: complex
    chi_stable: float | None = None
    a: complex = field(init=False)
    b: complex = field(init=False)
    A: float = field(init=False)
    B: float = field(init=False)
    C: float = field(init=False)
    chi: float = field(init=False)
    ctrl: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    length: float = field(init=False)

    def __post_init__(self) -> None:
        a = self.w0
        b = self.w1 - self.w0
        conj_ab = a.conjugate() * b
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "A", abs(b) ** 2)
        object.__setattr__(self, "B", 2.0 * conj_ab.real)
        object.__setattr__(self, "C", abs(a) ** 2)
        # chi = Im(conj(a) b) mathematically; the caller may supply the
        # cancellation-free product form 3 tau sqrt(lam0 lam1) sin(beta1),
        # which preserves full relative accuracy for tiny tangent turns.
        chi = self.chi_stable if self.chi_stable is not None else conj_ab.imag
        object.__setattr__(self, "chi", chi)
        ctrl = np.array(self.ctrl, dtype=np.float64, copy=True)
        ctrl.setflags(write=False)
        object.__setattr__(self, "ctrl", ctrl)
        object.__setattr__(self, "length", arc_length_at(self.A, self.B, self.C, 1.0))

    # -- Bezier control-net construction ---------------------------------

    @staticmethod
    def control_net(p0: complex, p1: complex, w0: complex, w1: complex) -> np.ndarray:
        """Control net from the PH legs, endpoint-snapped to ``p0`` and ``p1``.

        The interior control points are ``B1 = B0 + w0^2 / 3`` and
        ``B2 = B1 + w0 w1 / 3``; the final point is snapped to the exact
        normalized input point after the reconstruction residual
        ``|B2 + w1^2 / 3 - p1|`` has been verified by the caller.
        """
        b0 = p0
        b1 = b0 + (w0 * w0) / 3.0
        b2 = b1 + (w0 * w1) / 3.0
        b3 = p1
        return np.array(
            [
                [b0.real, b0.imag],
                [b1.real, b1.imag],
                [b2.real, b2.imag],
                [b3.real, b3.imag],
            ],
            dtype=np.float64,
        )

    # -- local evaluation -------------------------------------------------

    def point_local(self, t: float) -> tuple[float, float]:
        """De Casteljau evaluation of the normalized cubic at local ``t``."""
        c = self.ctrl
        u = 1.0 - t
        x01 = u * c[0, 0] + t * c[1, 0]
        y01 = u * c[0, 1] + t * c[1, 1]
        x12 = u * c[1, 0] + t * c[2, 0]
        y12 = u * c[1, 1] + t * c[2, 1]
        x23 = u * c[2, 0] + t * c[3, 0]
        y23 = u * c[2, 1] + t * c[3, 1]
        x012 = u * x01 + t * x12
        y012 = u * y01 + t * y12
        x123 = u * x12 + t * x23
        y123 = u * y12 + t * y23
        return (u * x012 + t * x123, u * y012 + t * y123)

    def w_at(self, t: float) -> complex:
        """Preimage ``w(t) = a + b t``; exact endpoint values at t = 0, 1."""
        if t == 0.0:
            return self.w0
        if t == 1.0:
            return self.w1
        return complex(self.a.real + self.b.real * t, self.a.imag + self.b.imag * t)

    def sigma(self, t: float) -> float:
        """Parametric speed ``sigma(t) = |w(t)|^2`` (spec section 11.4).

        Evaluated from the preimage rather than the polynomial
        ``A t^2 + B t + C``, which preserves relative accuracy at segment
        ends with strongly unequal edge lengths.
        """
        w = self.w_at(t)
        return w.real * w.real + w.imag * w.imag

    def sigma_poly(self, t: float) -> float:
        """Polynomial speed used as the arc-length derivative."""
        return speed(self.A, self.B, self.C, t)

    def tangent_local(self, t: float) -> tuple[float, float]:
        """Unit tangent ``T = (w / |w|)^2`` (spec section 11.2)."""
        w = self.w_at(t)
        n = math.hypot(w.real, w.imag)
        r = w.real / n
        s = w.imag / n
        tx = (r - s) * (r + s)
        ty = 2.0 * r * s
        nrm2 = tx * tx + ty * ty
        if abs(nrm2 - 1.0) > 64.0 * EPS:
            nrm = math.sqrt(nrm2)
            tx /= nrm
            ty /= nrm
        return (tx, ty)

    def curvature_local(self, t: float) -> float:
        """Normalized signed curvature ``2 chi / sigma(t)^2``."""
        s = self.sigma(t)
        return 2.0 * self.chi / (s * s)

    def arc_length_local(self, t: float) -> float:
        """Exact normalized local arc length ``S(t)``."""
        return arc_length_at(self.A, self.B, self.C, t)

    def invert_arc_length_local(self, s: float) -> float:
        """Unique local parameter with ``S(t) = s`` for ``s`` in ``[0, length]``."""
        return invert_arc_length(self.A, self.B, self.C, self.chi, self.length, s)

    # -- regularity data ---------------------------------------------------

    def sigma_extremes(self) -> tuple[float, float]:
        """``(sigma_min, sigma_end_max)`` over ``[0, 1]`` (spec section 10.1)."""
        s0 = self.sigma(0.0)
        s1 = self.sigma(1.0)
        end_max = max(s0, s1)
        if self.A > 0.0:
            t_star = -0.5 * self.B / self.A
            if t_star < 0.0:
                t_star = 0.0
            elif t_star > 1.0:
                t_star = 1.0
            s_min = min(self.sigma(t_star), s0, s1)
        else:
            s_min = min(s0, s1)
        return (s_min, end_max)
