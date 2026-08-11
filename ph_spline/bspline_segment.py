"""Immutable compiled PH B-spline span kernels.

The authoritative representation is a complex Bernstein preimage ``w``.
Position, speed, and arc-length coefficients are derived from it by finite
Bernstein product and antiderivative identities; no quadrature contributes to
the stored geometry or metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.typing import NDArray

from ph_spline.bspline_types import InversePolicy
from ph_spline.exceptions import ArcLengthInversionError

_EPS = np.finfo(np.float64).eps


def _readonly(array: NDArray) -> NDArray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def de_casteljau(values: NDArray, u: float):
    """Evaluate scalar, complex, or vector Bernstein data stably."""

    work = np.array(values, copy=True)
    if u == 0.0:
        return work[0].copy() if work.ndim > 1 else work[0]
    if u == 1.0:
        return work[-1].copy() if work.ndim > 1 else work[-1]
    v = 1.0 - u
    for width in range(work.shape[0] - 1, 0, -1):
        work[:width] = v * work[:width] + u * work[1 : width + 1]
    return work[0].copy() if work.ndim > 1 else work[0]


def derivative_controls(values: NDArray, order: int) -> NDArray:
    """Bernstein controls of an ordinary local-parameter derivative."""

    degree = values.shape[0] - 1
    if order == 0:
        return np.asarray(values)
    if order > degree:
        shape = (1,) + values.shape[1:]
        return np.zeros(shape, dtype=values.dtype)
    result = np.array(values, copy=True)
    for j in range(order):
        result = (degree - j) * np.diff(result, axis=0)
    return result


@cache
def _power_matrix(degree: int) -> NDArray[np.float64]:
    """Cached Bernstein-to-power transform for one modest degree."""

    result = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    for order in range(degree + 1):
        factor = math.comb(degree, order)
        for index in range(order + 1):
            result[order, index] = (
                factor * (-1.0) ** (order - index) * math.comb(order, index)
            )
    result.setflags(write=False)
    return result


def _horner(coefficients: NDArray, value: float):
    result = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        result = result * value + coefficient
    return result


@cache
def _product_terms(degree: int) -> tuple[tuple[tuple[int, int, float], ...], ...]:
    """Exact combinatorial Bernstein-product weights grouped by output index."""

    groups: list[list[tuple[int, int, float]]] = [[] for _ in range(2 * degree + 1)]
    for a in range(degree + 1):
        ca = math.comb(degree, a)
        for b in range(degree + 1):
            weight = ca * math.comb(degree, b) / math.comb(2 * degree, a + b)
            groups[a + b].append((a, b, float(weight)))
    return tuple(tuple(group) for group in groups)


def bernstein_square(values: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Bernstein coefficients of ``w**2``."""

    degree = values.size - 1
    result = np.empty(2 * degree + 1, dtype=np.complex128)
    for k, terms in enumerate(_product_terms(degree)):
        result[k] = sum(weight * values[a] * values[b] for a, b, weight in terms)
    return result


def bernstein_abs_square(values: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Real Bernstein coefficients of ``|w|**2`` with an imaginary check."""

    degree = values.size - 1
    result = np.empty(2 * degree + 1, dtype=np.float64)
    for k, terms in enumerate(_product_terms(degree)):
        value = sum(
            weight * values[a] * values[b].conjugate() for a, b, weight in terms
        )
        scale = max(1.0, abs(value.real))
        if abs(value.imag) > 256.0 * _EPS * scale:
            raise ArithmeticError("PH speed product acquired a material imaginary part")
        result[k] = value.real
    return result


def _general_product(a: NDArray, b: NDArray) -> NDArray:
    """Bernstein coefficients of the product of two Bernstein polynomials."""

    degree_a = a.size - 1
    degree_b = b.size - 1
    out = np.zeros(degree_a + degree_b + 1, dtype=np.result_type(a, b))
    for i in range(degree_a + 1):
        factor = math.comb(degree_a, i) * a[i]
        for j in range(degree_b + 1):
            out[i + j] += (
                factor
                * math.comb(degree_b, j)
                / math.comb(degree_a + degree_b, i + j)
                * b[j]
            )
    return out


def curvature_extremes(
    preimage: NDArray[np.complex128], parameter_width: float
) -> tuple[float, float]:
    """Certified signed extremes of the normalized curvature on one span.

    With ``kappa_hat(t) = 2 Im(conj(w) w_u) / (h |w|^4) = 2 P / (h Q)``,
    interior extrema are sign-crossing roots of ``E = P'Q - PQ'`` (degree
    ``6m - 2``).  A Bernstein interval whose coefficients keep one strict
    sign contains no crossing, and a tangential zero of ``E`` is a
    curvature inflection rather than an extremum, so recursive midpoint
    subdivision with the coefficient sign test enumerates every extremum
    candidate.  Subdivision to depth 30 resolves each root to ``~1e-9`` in
    ``t``, which is second-order small in the extremum value.
    """

    w = np.asarray(preimage, dtype=np.complex128)
    dw = (w.size - 1) * np.diff(w)
    numerator = _general_product(np.conjugate(w), dw).imag.copy()
    speed2 = _general_product(np.conjugate(w), w).real.copy()
    denominator = _general_product(speed2, speed2)
    d_num = (numerator.size - 1) * np.diff(numerator)
    d_den = (denominator.size - 1) * np.diff(denominator)
    critical = _general_product(d_num, denominator) - _general_product(
        numerator, d_den
    )

    def kappa(t: float) -> float:
        return (
            2.0
            * float(de_casteljau(numerator, t))
            / (parameter_width * float(de_casteljau(denominator, t)))
        )

    candidates = [0.0, 1.0]
    magnitude = float(np.max(np.abs(critical)))
    if magnitude > 0.0:
        ambiguous = 1.0e-14 * magnitude
        stack: list[tuple[float, float, NDArray[np.float64], int]] = [
            (0.0, 1.0, critical, 0)
        ]
        boxes = 0
        while stack:
            lo, hi, coeffs, depth = stack.pop()
            boxes += 1
            has_pos = bool(np.any(coeffs > ambiguous))
            has_neg = bool(np.any(coeffs < -ambiguous))
            if not has_pos and not has_neg:
                candidates.append(0.5 * (lo + hi))  # locally flat E
                continue
            if not (has_pos and has_neg):
                continue  # one strict sign: no crossing inside
            if depth >= 30 or boxes > 4096:
                candidates.append(0.5 * (lo + hi))
                continue
            left, right = _split_half(coeffs)
            mid = 0.5 * (lo + hi)
            stack.append((mid, hi, right, depth + 1))
            stack.append((lo, mid, left, depth + 1))
    values = [kappa(t) for t in candidates]
    return (min(values), max(values))


def _split_half(values: NDArray[np.complex128]) -> tuple[NDArray, NDArray]:
    degree = values.size - 1
    triangle = [np.array(values, copy=True)]
    for _ in range(degree):
        triangle.append(0.5 * (triangle[-1][:-1] + triangle[-1][1:]))
    left = np.array([triangle[j][0] for j in range(degree + 1)])
    right = np.array([triangle[degree - j][j] for j in range(degree + 1)])
    return left, right


def _box_distance(values: NDArray[np.complex128]) -> float:
    min_x = float(np.min(values.real))
    max_x = float(np.max(values.real))
    min_y = float(np.min(values.imag))
    max_y = float(np.max(values.imag))
    dx = min_x if min_x > 0.0 else -max_x if max_x < 0.0 else 0.0
    dy = min_y if min_y > 0.0 else -max_y if max_y < 0.0 else 0.0
    return math.hypot(dx, dy)


def certify_nonzero(
    values: NDArray[np.complex128], max_depth: int
) -> tuple[float, float]:
    """Certify a lower preimage norm using recursive Bernstein boxes."""

    upper = float(np.max(np.abs(values)))
    if not upper > 0.0 or not math.isfinite(upper):
        return (0.0, upper)
    lower = math.inf
    stack: list[tuple[NDArray[np.complex128], int]] = [(values, 0)]
    while stack:
        controls, depth = stack.pop()
        bound = _box_distance(controls)
        if bound > 0.0:
            lower = min(lower, bound)
            continue
        if depth >= max_depth:
            return (0.0, upper)
        left, right = _split_half(controls)
        stack.append((left, depth + 1))
        stack.append((right, depth + 1))
    return (lower if math.isfinite(lower) else 0.0, upper)


def sampled_min_norm(values: NDArray[np.complex128]) -> float:
    """Cheap deterministic branch-ranking estimate; not a certificate."""

    degree = values.size - 1
    minimum = math.inf
    for u in np.linspace(0.0, 1.0, 2 * degree + 9):
        minimum = min(minimum, abs(de_casteljau(values, float(u))))
    return minimum


@dataclass(frozen=True, slots=True)
class PHBSplineSpan:
    """One verified immutable PH polynomial span in normalized coordinates."""

    span_id: int
    parameter_width: float
    preimage: NDArray[np.complex128]
    preimage_left_jet: NDArray[np.complex128]
    preimage_right_jet: NDArray[np.complex128]
    position: NDArray[np.complex128]
    speed: NDArray[np.float64]
    arc: NDArray[np.float64]
    reverse_arc: NDArray[np.float64]
    position_power: NDArray[np.complex128]
    speed_power: NDArray[np.float64]
    arc_power: NDArray[np.float64]
    reverse_arc_power: NDArray[np.float64]
    length: float
    regularity_lower: float
    regularity_upper: float
    kappa_min: float
    kappa_max: float
    lut_u: NDArray[np.float64]
    lut_s: NDArray[np.float64]

    @property
    def preimage_degree(self) -> int:
        return self.preimage.size - 1

    @property
    def degree(self) -> int:
        return self.position.size - 1

    def w(self, local_u: float) -> complex:
        return complex(de_casteljau(self.preimage, local_u))

    def w_derivative(self, local_u: float, order: int) -> complex:
        # High-order Bernstein endpoint differences can subtract nearly equal
        # controls.  Preserve the endpoint jets used to construct the span so
        # joins are evaluated from their well-conditioned elementary data.
        if local_u == 0.0 and order < self.preimage_left_jet.size:
            return complex(self.preimage_left_jet[order])
        if local_u == 1.0 and order < self.preimage_right_jet.size:
            return complex(self.preimage_right_jet[order])
        controls = derivative_controls(self.preimage, order)
        return complex(de_casteljau(controls, local_u))

    def point_normalized(self, local_u: float) -> complex:
        if self.degree <= 9 and 0.0 < local_u < 1.0:
            return complex(_horner(self.position_power, local_u))
        return complex(de_casteljau(self.position, local_u))

    def position_derivative_local(self, local_u: float, order: int) -> complex:
        if order > self.degree:
            return 0.0j
        controls = derivative_controls(self.position, order)
        return complex(de_casteljau(controls, local_u))

    def speed_local(self, local_u: float) -> float:
        w = self.w(local_u)
        return self.parameter_width * (w.real * w.real + w.imag * w.imag)

    def _fast_speed_local(self, local_u: float) -> float:
        result = self.parameter_width * float(_horner(self.speed_power, local_u))
        return (
            result
            if result > 0.0 and math.isfinite(result)
            else self.speed_local(local_u)
        )

    def tangent(self, local_u: float) -> tuple[float, float]:
        w = self.w(local_u)
        norm = math.hypot(w.real, w.imag)
        r = w.real / norm
        s = w.imag / norm
        tx = (r - s) * (r + s)
        ty = 2.0 * r * s
        norm2 = tx * tx + ty * ty
        if abs(norm2 - 1.0) > 64.0 * _EPS:
            correction = 1.0 / math.sqrt(norm2)
            tx *= correction
            ty *= correction
        return (tx, ty)

    def signed_curvature_normalized(self, local_u: float) -> float:
        w = self.w(local_u)
        dw_du = self.w_derivative(local_u, 1)
        dw_dt = dw_du / self.parameter_width
        scale = max(abs(w.real), abs(w.imag), abs(dw_dt.real), abs(dw_dt.imag))
        if scale == 0.0:
            return 0.0
        ws = w / scale
        ds = dw_dt / scale
        cross = ws.real * ds.imag - ws.imag * ds.real
        norm2 = ws.real * ws.real + ws.imag * ws.imag
        return 2.0 * cross / (scale * scale * norm2 * norm2)

    def arc_at(self, local_u: float) -> float:
        if local_u == 0.0:
            return 0.0
        if local_u == 1.0:
            return self.length
        if local_u <= 0.5:
            return float(de_casteljau(self.arc, local_u))
        remainder = float(de_casteljau(self.reverse_arc, 1.0 - local_u))
        return self.length - remainder

    def arc_remainder(self, local_u: float) -> float:
        if local_u == 0.0:
            return self.length
        if local_u == 1.0:
            return 0.0
        return float(de_casteljau(self.reverse_arc, 1.0 - local_u))

    def _fast_arc_value(self, local_u: float, reverse: bool) -> float:
        if reverse:
            return float(_horner(self.reverse_arc_power, 1.0 - local_u))
        if local_u <= 0.5:
            return float(_horner(self.arc_power, local_u))
        return self.length - float(_horner(self.reverse_arc_power, 1.0 - local_u))

    def invert_arc_length(self, target: float, policy: InversePolicy) -> float:
        """Invert the analytic monotone arc polynomial with a strict bracket."""

        if target == 0.0:
            return 0.0
        if target == self.length:
            return 1.0
        reverse = target > policy.endpoint_reverse_threshold * self.length
        if reverse:
            goal = self.length - target
            s_nodes = self.length - self.lut_s[::-1]
        else:
            goal = target
            s_nodes = self.lut_s
        index = int(np.searchsorted(s_nodes, goal, side="right") - 1)
        index = max(0, min(index, s_nodes.size - 2))
        lo = float(self.lut_u[index])
        hi = float(self.lut_u[index + 1])
        s_lo = float(s_nodes[index])
        s_hi = float(s_nodes[index + 1])
        if reverse:
            lo, hi = 1.0 - hi, 1.0 - lo
        fraction = 0.5 if s_hi == s_lo else (goal - s_lo) / (s_hi - s_lo)
        x = (1.0 - fraction) * lo + fraction * hi
        tolerance = (
            policy.fast_iterations * 0.0
            + 64.0 * _EPS * self.length
            + 4.0 * math.ulp(max(goal, np.nextafter(0.0, 1.0)))
        )
        # A few compiled Horner steps provide a cheap seed.  They never alter
        # the certified LUT bracket and are never used for acceptance; the
        # safeguarded phase below evaluates Bernstein data and applies the
        # production residual gate.
        if self.degree <= 17:
            for _ in range(policy.fast_iterations):
                residual = self._fast_arc_value(x, reverse) - goal
                speed = self._fast_speed_local(x)
                derivative = -speed if reverse else speed
                candidate = x - residual / derivative if derivative != 0.0 else math.nan
                if not (lo < candidate < hi) or not math.isfinite(candidate):
                    break
                x = candidate
        for _ in range(policy.max_iterations):
            if reverse:
                residual = self.arc_remainder(x) - goal
                derivative = -self.speed_local(x)
            else:
                residual = self.arc_at(x) - goal
                derivative = self.speed_local(x)
            if abs(residual) <= tolerance:
                return min(1.0, max(0.0, x))
            if residual > 0.0:
                if reverse:
                    lo = x
                else:
                    hi = x
            else:
                if reverse:
                    hi = x
                else:
                    lo = x
            candidate = x - residual / derivative if derivative != 0.0 else math.nan
            if not (lo < candidate < hi) or not math.isfinite(candidate):
                candidate = 0.5 * (lo + hi)
            x = candidate
        raise ArcLengthInversionError(
            "PH B-spline local arc-length inversion missed its residual bound",
            span_id=self.span_id,
            quantity="arc residual",
            value=abs(residual),
            bound=tolerance,
        )


def compile_span(
    *,
    span_id: int,
    parameter_width: float,
    preimage: NDArray[np.complex128],
    preimage_left_jet: NDArray[np.complex128],
    preimage_right_jet: NDArray[np.complex128],
    start: complex,
    regularity_lower: float,
    regularity_upper: float,
    inverse_policy: InversePolicy,
) -> PHBSplineSpan:
    """Construct every authoritative span polynomial from ``preimage``."""

    degree = preimage.size - 1
    curve_degree = 2 * degree + 1
    hodograph = parameter_width * bernstein_square(preimage)
    speed = parameter_width * bernstein_abs_square(preimage)
    position = np.empty(curve_degree + 1, dtype=np.complex128)
    position[0] = start
    for k in range(curve_degree):
        position[k + 1] = position[k] + hodograph[k] / curve_degree
    arc = np.empty(curve_degree + 1, dtype=np.float64)
    arc[0] = 0.0
    for k in range(curve_degree):
        arc[k + 1] = arc[k] + speed[k] / curve_degree
    length = float(arc[-1])
    if not length > 0.0 or not math.isfinite(length):
        raise ArithmeticError("PH B-spline span length is not positive and finite")
    reverse_arc = length - arc[::-1]
    if curve_degree <= 17:
        curve_power = _power_matrix(curve_degree) @ np.column_stack(
            (position, arc, reverse_arc)
        )
        position_power = curve_power[:, 0]
        arc_power = curve_power[:, 1].real
        reverse_arc_power = curve_power[:, 2].real
        speed_power = _power_matrix(curve_degree - 1) @ speed
    else:
        position_power = np.empty(0, dtype=np.complex128)
        speed_power = np.empty(0, dtype=np.float64)
        arc_power = np.empty(0, dtype=np.float64)
        reverse_arc_power = np.empty(0, dtype=np.float64)
    variation = max(1.0, regularity_upper / max(regularity_lower, 1.0e-300))
    requested = max(
        inverse_policy.lut_nodes_min,
        math.ceil(4.0 + 2.0 * math.sqrt(variation)),
    )
    if inverse_policy.lut_power_of_two:
        requested = 1 << max(3, (requested - 1).bit_length())
    node_count = min(inverse_policy.lut_nodes_max, requested)
    lut_u = np.linspace(0.0, 1.0, node_count + 1, dtype=np.float64)
    lut_s = np.array([de_casteljau(arc, float(u)) for u in lut_u])
    lut_s[0] = 0.0
    lut_s[-1] = length
    if not np.all(np.diff(lut_s) > 0.0):
        raise ArithmeticError("PH B-spline inverse LUT is not strictly monotone")
    kappa_min, kappa_max = curvature_extremes(preimage, parameter_width)
    return PHBSplineSpan(
        span_id=span_id,
        parameter_width=float(parameter_width),
        preimage=_readonly(preimage),
        preimage_left_jet=_readonly(preimage_left_jet),
        preimage_right_jet=_readonly(preimage_right_jet),
        position=_readonly(position),
        speed=_readonly(speed),
        arc=_readonly(arc),
        reverse_arc=_readonly(reverse_arc),
        position_power=_readonly(position_power),
        speed_power=_readonly(speed_power),
        arc_power=_readonly(arc_power),
        reverse_arc_power=_readonly(reverse_arc_power),
        length=length,
        regularity_lower=float(regularity_lower),
        regularity_upper=float(regularity_upper),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),
        lut_u=_readonly(lut_u),
        lut_s=_readonly(lut_s),
    )


__all__ = [
    "PHBSplineSpan",
    "certify_nonzero",
    "compile_span",
    "curvature_extremes",
    "de_casteljau",
    "derivative_controls",
    "sampled_min_norm",
]
