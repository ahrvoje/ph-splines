"""Deterministic construction and independent verification for ``PHBSpline``."""

from __future__ import annotations

import cmath
import math
import numbers
from dataclasses import dataclass, replace
from functools import cache
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from ph_spline.arclength import compensated_prefix_sums
from ph_spline.bspline_basis import (
    PreimageBasis,
    PreimageSolution,
    build_preimage_basis,
    guide_controls,
    solve_preimage,
)
from ph_spline.bspline_segment import (
    PHBSplineSpan,
    certify_nonzero,
    compile_span,
    derivative_controls,
    sampled_min_norm,
)
from ph_spline.bspline_types import (
    BuildDiagnostics,
    ConstructionPolicy,
    ContinuitySpec,
    InversePolicy,
    NumericalPolicy,
)
from ph_spline.exceptions import (
    ContinuitySpecificationError,
    ContinuityVerificationError,
    DegeneratePointDataError,
    InsufficientPointDataError,
    InterpolationVerificationError,
    InvalidPointDataError,
    LengthResolutionError,
    NonFiniteCoordinateError,
    NonRegularSplineError,
    NumericalPrecisionError,
    ResourceLimitError,
)

_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True, slots=True)
class PHBSplineBuildState:
    """Complete verified state prepared before publication or edit commit."""

    points: NDArray[np.float64]
    normalized_points: NDArray[np.complex128]
    origin: tuple[float, float]
    scale: float
    raw_widths: NDArray[np.float64]
    span_widths: NDArray[np.float64]
    parameter_total: float
    knots: NDArray[np.float64]
    span_knots: NDArray[np.float64]
    spans: tuple[PHBSplineSpan, ...]
    span_keys: tuple[tuple[int, int, int], ...]
    preimage_controls: NDArray[np.complex128]
    preimage_seam_sign: int
    prefix_normalized: NDArray[np.float64]
    prefix_user: NDArray[np.float64]
    total_length: float
    preimage_degree: int
    requested_continuity: ContinuitySpec
    verified_continuity: ContinuitySpec
    diagnostics: BuildDiagnostics


def _readonly(array: NDArray) -> NDArray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _safe_difference(a: float, b: float) -> float:
    direct = a - b
    if math.isfinite(direct):
        return direct
    scale = max(abs(a), abs(b))
    if scale == 0.0 or not math.isfinite(scale):
        return direct
    return scale * (a / scale - b / scale)


def _safe_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = _safe_difference(a[0], b[0])
    dy = _safe_difference(a[1], b[1])
    return math.hypot(dx, dy)


def validate_bspline_points(points: object, *, closed: bool) -> NDArray[np.float64]:
    """Validate general array-like planar data without unsafe coercion."""

    if isinstance(points, (str, bytes, dict)) or points is None:
        raise InvalidPointDataError("points must be a finite planar array-like")
    try:
        rows = list(points)  # type: ignore[arg-type]
    except TypeError as exc:
        raise InvalidPointDataError(
            "points must be an iterable of planar points"
        ) from exc
    minimum = 3 if closed else 2
    if len(rows) < minimum:
        raise InsufficientPointDataError(
            f"{'Closed' if closed else 'Open'} PHBSpline requires at least {minimum} points",
            quantity="point count",
            value=len(rows),
            bound=f">= {minimum}",
        )
    result = np.empty((len(rows), 2), dtype=np.float64)
    for i, row in enumerate(rows):
        if isinstance(row, (str, bytes, dict)):
            raise InvalidPointDataError(
                "Each point must contain two real scalars", index=i
            )
        try:
            values = list(row)  # type: ignore[arg-type]
        except TypeError as exc:
            raise InvalidPointDataError(
                "Each point must contain two real scalars", index=i
            ) from exc
        if len(values) != 2:
            raise InvalidPointDataError(
                "Each point must contain exactly two coordinates",
                index=i,
                quantity="coordinate count",
                value=len(values),
                bound="== 2",
            )
        for axis, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, numbers.Real
            ):
                raise InvalidPointDataError(
                    "Coordinates must be real non-Boolean scalars",
                    index=i,
                    quantity=f"coordinate[{axis}]",
                    value=value,
                )
            coordinate = float(value)
            if not math.isfinite(coordinate):
                raise NonFiniteCoordinateError(
                    "Coordinates must be finite",
                    index=i,
                    quantity="coordinate",
                    value=coordinate,
                )
            result[i, axis] = coordinate
    pairs = [(i, i + 1) for i in range(len(rows) - 1)]
    if closed:
        pairs.append((len(rows) - 1, 0))
    for left, right in pairs:
        distance = _safe_distance(tuple(result[left]), tuple(result[right]))
        if distance == 0.0:
            raise DegeneratePointDataError(
                "Consecutive interpolation points must be distinct",
                index=(left, right),
                quantity="chord length",
                value=distance,
                bound="> 0",
            )
        if not math.isfinite(distance):
            raise NumericalPrecisionError(
                "A chord displacement is not representable in binary64",
                index=(left, right),
                quantity="chord length",
                value=distance,
                bound="finite",
            )
    return result


def validate_continuity(
    g_order: object, c_order: object, curvature_order: object
) -> tuple[int, ContinuitySpec]:
    values: list[int | None] = []
    for name, value in (
        ("g_order", g_order),
        ("c_order", c_order),
        ("curvature_order", curvature_order),
    ):
        if value is None:
            values.append(None)
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ContinuitySpecificationError(
                f"{name} must be a nonnegative integer or None",
                quantity=name,
                value=value,
            )
        integer = int(value)
        if integer < 0:
            raise ContinuitySpecificationError(
                f"{name} must be nonnegative",
                quantity=name,
                value=integer,
                bound=">= 0",
            )
        values.append(integer)
    g, c, k = values
    if g is None and c is None and k is None:
        g = 2
    required = max(2, g or 0, c or 0, (k + 2) if k is not None else 0)
    return required, ContinuitySpec(g, c, k)


def _normalize_points(
    points: NDArray[np.float64],
    closed: bool,
    frame: tuple[tuple[float, float], float] | None = None,
) -> tuple[NDArray[np.complex128], tuple[float, float], float, NDArray[np.float64]]:
    pairs = [(i, i + 1) for i in range(points.shape[0] - 1)]
    if closed:
        pairs.append((points.shape[0] - 1, 0))
    lengths = np.array(
        [_safe_distance(tuple(points[a]), tuple(points[b])) for a, b in pairs],
        dtype=np.float64,
    )
    scale = float(np.max(lengths)) if frame is None else float(frame[1])
    if not scale > 0.0 or not math.isfinite(scale):
        raise NumericalPrecisionError(
            "Point normalization scale is not positive and finite",
            quantity="normalization scale",
            value=scale,
            bound="positive finite",
        )
    origin = (float(points[0, 0]), float(points[0, 1])) if frame is None else frame[0]
    normalized = np.empty(points.shape[0], dtype=np.complex128)
    for i, (x, y) in enumerate(points):
        nx = _safe_difference(float(x), origin[0]) / scale
        ny = _safe_difference(float(y), origin[1]) / scale
        if not (math.isfinite(nx) and math.isfinite(ny)):
            raise NumericalPrecisionError(
                "Normalized point coordinate is not finite",
                index=i,
                quantity="normalized coordinate",
                value=(nx, ny),
            )
        normalized[i] = complex(nx, ny)
    return normalized, origin, scale, lengths / scale


def _parameter_widths(
    normalized_lengths: NDArray[np.float64], policy: ConstructionPolicy
) -> NDArray[np.float64]:
    if policy.parameterization == "uniform":
        widths = np.ones_like(normalized_lengths)
    elif policy.parameterization == "chord":
        widths = np.array(normalized_lengths, copy=True)
    elif policy.parameterization == "centripetal":
        widths = np.sqrt(normalized_lengths)
    else:
        raise ValueError(
            "ConstructionPolicy.parameterization must be 'uniform', 'chord', or 'centripetal'"
        )
    if not np.all(np.isfinite(widths)) or not np.all(widths > 0.0):
        raise NumericalPrecisionError(
            "Parameter widths are not positive and finite",
            quantity="parameter widths",
            value=widths.tolist(),
            bound="positive finite",
        )
    return widths


def _guide_preimage(
    points: NDArray[np.complex128], widths: NDArray[np.float64], closed: bool
) -> tuple[NDArray[np.complex128], int]:
    count = points.size
    span_count = widths.size
    secants = np.empty(span_count, dtype=np.complex128)
    for i in range(span_count):
        secants[i] = (points[(i + 1) % count] - points[i]) / widths[i]
    velocity = np.empty(count, dtype=np.complex128)
    if closed:
        for i in range(count):
            prev = (i - 1) % count
            nxt = i
            velocity[i] = (
                widths[nxt] * secants[prev] + widths[prev] * secants[nxt]
            ) / (widths[prev] + widths[nxt])
            if abs(velocity[i]) <= 32.0 * _EPS * max(
                abs(secants[prev]), abs(secants[nxt])
            ):
                velocity[i] = (
                    secants[prev]
                    if abs(secants[prev]) >= abs(secants[nxt])
                    else secants[nxt]
                )
    else:
        velocity[0] = secants[0]
        velocity[-1] = secants[-1]
        for i in range(1, count - 1):
            velocity[i] = (widths[i] * secants[i - 1] + widths[i - 1] * secants[i]) / (
                widths[i - 1] + widths[i]
            )
            if abs(velocity[i]) <= 32.0 * _EPS * max(
                abs(secants[i - 1]), abs(secants[i])
            ):
                velocity[i] = (
                    secants[i - 1]
                    if abs(secants[i - 1]) >= abs(secants[i])
                    else secants[i]
                )
    roots = np.array([cmath.sqrt(value) for value in velocity])
    if not closed:
        for i in range(1, count):
            if abs(-roots[i] - roots[i - 1]) < abs(roots[i] - roots[i - 1]):
                roots[i] = -roots[i]
        return roots, 1
    # Optimize both possible lifts of the squared tangent around the cycle.
    # A regular simple closed curve has odd turning number, so its continuous
    # square root is antiperiodic even though w**2 is periodic.
    signs = (1.0, -1.0)
    costs = np.full((count, 2), math.inf)
    parent = np.zeros((count, 2), dtype=np.int8)
    costs[0, 0] = 0.0
    for i in range(1, count):
        for state, sign in enumerate(signs):
            value = sign * roots[i]
            options = [
                costs[i - 1, previous]
                + abs(value - signs[previous] * roots[i - 1]) ** 2
                for previous in range(2)
            ]
            parent[i, state] = int(np.argmin(options))
            costs[i, state] = options[parent[i, state]]
    endings = []
    for seam_sign in (1, -1):
        final_costs = [
            costs[-1, state]
            + abs(signs[state] * roots[-1] - seam_sign * roots[0]) ** 2
            for state in range(2)
        ]
        state = int(np.argmin(final_costs))
        endings.append((float(final_costs[state]), -seam_sign, seam_sign, state))
    _, _, seam_sign, state = min(endings)
    selected = np.empty(count, dtype=np.complex128)
    for i in range(count - 1, -1, -1):
        selected[i] = signs[state] * roots[i]
        if i:
            state = int(parent[i, state])
    return selected, seam_sign


def _closed_offset(widths: NDArray[np.float64], start: int, offset: int) -> float:
    count = widths.size
    if offset > 0:
        return float(sum(widths[(start + j) % count] for j in range(offset)))
    if offset < 0:
        return -float(sum(widths[(start - j - 1) % count] for j in range(-offset)))
    return 0.0


def _guide_jets(
    guide: NDArray[np.complex128],
    widths: NDArray[np.float64],
    required_order: int,
    closed: bool,
) -> NDArray[np.complex128]:
    count = guide.size
    jets = np.zeros((count, required_order), dtype=np.complex128)
    jets[:, 0] = guide
    degree = min(3, count - 1)
    if degree == 0:
        return jets
    for i in range(count):
        if closed:
            if degree == 1:
                offsets = [0, 1]
            elif degree == 2:
                offsets = [-1, 0, 1]
            else:
                offsets = [-1, 0, 1, 2]
            xs = np.array([_closed_offset(widths, i, off) for off in offsets])
            ys = np.array([guide[(i + off) % count] for off in offsets])
        else:
            window = degree + 1
            left = max(0, min(i - degree // 2, count - window))
            indices = np.arange(left, left + window)
            # Form offsets from nearby widths directly.  Subtracting two
            # large cumulative parameters would make a local edit perturb
            # otherwise independent downstream jets through roundoff.
            xs = np.array(
                [
                    -math.fsum(float(value) for value in widths[j:i])
                    if j < i
                    else math.fsum(float(value) for value in widths[i:j])
                    for j in indices
                ],
                dtype=np.float64,
            )
            ys = guide[indices]
        local_scale = float(np.max(np.abs(xs)))
        if not local_scale > 0.0:
            continue
        scaled = xs / local_scale
        matrix = np.vander(scaled, N=degree + 1, increasing=True)
        coefficients = np.linalg.solve(matrix, ys)
        for order in range(1, min(required_order, degree + 1)):
            jets[i, order] = (
                math.factorial(order) * coefficients[order] / local_scale**order
            )
    return jets


@cache
def _integral_weights(degree: int) -> NDArray[np.float64]:
    result = np.empty((degree + 1, degree + 1), dtype=np.float64)
    for a in range(degree + 1):
        for b in range(degree + 1):
            result[a, b] = (
                math.comb(degree, a)
                * math.comb(degree, b)
                / ((2 * degree + 1) * math.comb(2 * degree, a + b))
            )
    result.setflags(write=False)
    return result


def _falling(value: int, order: int) -> int:
    result = 1
    for j in range(order):
        result *= value - j
    return result


def _span_preimage_candidates(
    *,
    span: int,
    endpoint: int,
    chord: complex,
    width: float,
    jets: NDArray[np.complex128],
    required_order: int,
    degree: int,
    derivative_scale: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    controls = np.empty(degree + 1, dtype=np.complex128)
    left_differences = []
    right_differences = []
    for order in range(required_order):
        factor = derivative_scale if order else 1.0
        denominator = _falling(degree, order)
        left_differences.append(factor * width**order * jets[span, order] / denominator)
        right_differences.append(
            factor * width**order * jets[endpoint, order] / denominator
        )
    for k in range(required_order):
        controls[k] = sum(
            math.comb(k, order) * left_differences[order] for order in range(k + 1)
        )
        controls[degree - k] = sum(
            (-1) ** order * math.comb(k, order) * right_differences[order]
            for order in range(k + 1)
        )
    free = required_order
    known = [index for index in range(degree + 1) if index != free]
    weights = _integral_weights(degree)
    qa = weights[free, free]
    qb = 2.0 * sum(weights[free, j] * controls[j] for j in known)
    qc = (
        sum(weights[a, b] * controls[a] * controls[b] for a in known for b in known)
        - chord / width
    )
    discriminant = qb * qb - 4.0 * qa * qc
    root = cmath.sqrt(discriminant)
    first = np.array(controls, copy=True)
    second = np.array(controls, copy=True)
    first[free] = (-qb + root) / (2.0 * qa)
    second[free] = (-qb - root) / (2.0 * qa)
    return first, second


def _construct_spans(
    *,
    normalized: NDArray[np.complex128],
    widths: NDArray[np.float64],
    jets: NDArray[np.complex128],
    required_order: int,
    degree: int,
    closed: bool,
    numerics: NumericalPolicy,
    inverse: InversePolicy,
    reusable: tuple[PHBSplineSpan | None, ...] | None = None,
) -> tuple[tuple[PHBSplineSpan, ...], float, float, float]:
    span_count = widths.size
    threshold = math.sqrt(numerics.regularity_ratio_min)
    best_failure = 0.0
    for derivative_scale in (1.0, 0.5, 0.25, 0.0):
        spans: list[PHBSplineSpan] = []
        min_ratio = math.inf
        max_residual = 0.0
        failed = False
        for i in range(span_count):
            endpoint = (i + 1) % normalized.size
            chord = normalized[endpoint] - normalized[i]
            left_jet = np.array(
                [
                    jets[i, order]
                    if order == 0
                    else derivative_scale * widths[i] ** order * jets[i, order]
                    for order in range(required_order)
                ],
                dtype=np.complex128,
            )
            right_jet = np.array(
                [
                    jets[endpoint, order]
                    if order == 0
                    else derivative_scale * widths[i] ** order * jets[endpoint, order]
                    for order in range(required_order)
                ],
                dtype=np.complex128,
            )
            old = None if reusable is None else reusable[i]
            if (
                old is not None
                and old.preimage_degree == degree
                and old.parameter_width == float(widths[i])
                and np.array_equal(old.preimage_left_jet, left_jet)
                and np.array_equal(old.preimage_right_jet, right_jet)
            ):
                reused = old if old.span_id == i else replace(old, span_id=i)
                residual = abs(reused.position[-1] - normalized[endpoint])
                ratio = reused.regularity_lower / reused.regularity_upper
                max_residual = max(max_residual, residual)
                min_ratio = min(min_ratio, ratio * ratio)
                spans.append(reused)
                continue
            first, second = _span_preimage_candidates(
                span=i,
                endpoint=endpoint,
                chord=chord,
                width=float(widths[i]),
                jets=jets,
                required_order=required_order,
                degree=degree,
                derivative_scale=derivative_scale,
            )
            guide_midpoint = 0.5 * (jets[i, 0] + jets[endpoint, 0])
            candidates = []
            for controls in (first, second):
                lower, upper = certify_nonzero(
                    controls, numerics.max_regularization_subdivision_depth
                )
                ratio = lower / upper if upper > 0.0 else 0.0
                score = abs(controls[required_order] - guide_midpoint) / (
                    1.0 + abs(guide_midpoint)
                )
                score += 1.0e-3 / max(sampled_min_norm(controls), 1.0e-15)
                candidates.append(
                    (ratio >= threshold, score, -ratio, controls, lower, upper)
                )
            candidates.sort(key=lambda item: (not item[0], item[1], item[2]))
            accepted, _, negative_ratio, controls, lower, upper = candidates[0]
            ratio = -negative_ratio
            best_failure = max(best_failure, ratio)
            if not accepted:
                failed = True
                break
            try:
                compiled = compile_span(
                    span_id=i,
                    parameter_width=float(widths[i]),
                    preimage=controls,
                    preimage_left_jet=left_jet,
                    preimage_right_jet=right_jet,
                    start=normalized[i],
                    regularity_lower=lower,
                    regularity_upper=upper,
                    inverse_policy=inverse,
                )
            except ArithmeticError:
                failed = True
                break
            residual = abs(compiled.position[-1] - normalized[endpoint])
            max_residual = max(max_residual, residual)
            min_ratio = min(min_ratio, ratio * ratio)
            spans.append(compiled)
        if not failed:
            return tuple(spans), min_ratio, max_residual, derivative_scale
    raise NonRegularSplineError(
        "No deterministic PH preimage branch met the regularity margin",
        quantity="min |w| / max |w|",
        value=best_failure,
        bound=f"> {threshold:.3e}",
    )


def _construct_basis_spans(
    *,
    normalized: NDArray[np.complex128],
    basis: PreimageBasis,
    solution: PreimageSolution,
    required_order: int,
    closed: bool,
    numerics: NumericalPolicy,
    inverse: InversePolicy,
    point_ids: tuple[int, ...],
    reusable: dict[
        tuple[int, int, int], tuple[PHBSplineSpan, complex, complex]
    ]
    | None,
) -> tuple[
    tuple[PHBSplineSpan, ...],
    tuple[tuple[int, int, int], ...],
    float,
    float,
]:
    """Compile the Bezier extraction of one shared simple-knot preimage."""

    threshold = math.sqrt(numerics.regularity_ratio_min)
    spans: list[PHBSplineSpan] = []
    keys: list[tuple[int, int, int]] = []
    min_ratio = math.inf
    max_residual = 0.0
    left_jets = [
        np.asarray(
            [
                derivative_controls(preimage, order)[0]
                for order in range(required_order)
            ],
            dtype=np.complex128,
        )
        for preimage in solution.span_controls
    ]
    right_jets = [
        np.asarray(
            [
                derivative_controls(preimage, order)[-1]
                for order in range(required_order)
            ],
            dtype=np.complex128,
        )
        for preimage in solution.span_controls
    ]
    join_count = len(solution.span_controls) if closed else len(solution.span_controls) - 1
    for left_span in range(join_count):
        right_span = (left_span + 1) % len(solution.span_controls)
        join_sign = basis.seam_sign if closed and right_span == 0 else 1
        left_width = float(basis.span_widths[left_span])
        right_width = float(basis.span_widths[right_span])
        for order in range(required_order):
            left_value = right_jets[left_span][order] / left_width**order
            right_value = left_jets[right_span][order] / right_width**order
            magnitude = max(abs(left_value), abs(right_value))
            common = (
                0.0j
                if magnitude == 0.0
                else magnitude
                * 0.5
                * (left_value / magnitude + join_sign * right_value / magnitude)
            )
            # The two values are extractions of one mathematical B-spline
            # knot jet.  Store one canonical physical value on both spans so
            # endpoint evaluation does not inherit cancellation from two
            # differently scaled Bernstein difference ladders.
            right_jets[left_span][order] = common * left_width**order
            left_jets[right_span][order] = (
                join_sign * common * right_width**order
            )
    for span_id, preimage in enumerate(solution.span_controls):
        user_span = int(basis.span_to_user[span_id])
        endpoint = (user_span + 1) % normalized.size
        subspan = span_id - 2 * user_span
        key = (point_ids[user_span], point_ids[endpoint], subspan)
        keys.append(key)
        lower, upper = certify_nonzero(
            preimage, numerics.max_regularization_subdivision_depth
        )
        ratio = lower / upper if upper > 0.0 else 0.0
        if ratio < threshold:
            raise NonRegularSplineError(
                "The minimum-degree PH B-spline preimage is not regular",
                span_id=span_id,
                quantity="min |w| / max |w|",
                value=ratio,
                bound=f">= {threshold:.3e}",
            )
        min_ratio = min(min_ratio, ratio * ratio)
        left_jet = left_jets[span_id]
        right_jet = right_jets[span_id]
        start = (
            normalized[user_span]
            if subspan == 0
            else complex(spans[-1].position[-1])
        )
        old = None
        if reusable is not None:
            entry = reusable.get(key)
            if (
                entry is not None
                and entry[1] == normalized[user_span]
                and entry[2] == normalized[endpoint]
            ):
                old = entry[0]
        parameter_width = float(basis.span_widths[span_id])
        if (
            old is not None
            and old.preimage_degree == basis.degree
            and old.parameter_width == parameter_width
            and np.array_equal(old.preimage, preimage)
            and old.position[0] == start
        ):
            compiled = old if old.span_id == span_id else replace(old, span_id=span_id)
        else:
            try:
                compiled = compile_span(
                    span_id=span_id,
                    parameter_width=parameter_width,
                    preimage=preimage,
                    preimage_left_jet=left_jet,
                    preimage_right_jet=right_jet,
                    start=start,
                    regularity_lower=lower,
                    regularity_upper=upper,
                    inverse_policy=inverse,
                )
            except ArithmeticError as exc:
                raise NonRegularSplineError(
                    "A minimum-degree PH B-spline span could not be compiled",
                    span_id=span_id,
                    value=str(exc),
                ) from exc
        spans.append(compiled)
        if subspan == 1:
            max_residual = max(
                max_residual, abs(compiled.position[-1] - normalized[endpoint])
            )
    return tuple(spans), tuple(keys), min_ratio, max_residual


def _verify_continuity_pairs(
    pairs: list[tuple[PHBSplineSpan, PHBSplineSpan, int]],
    required_order: int,
) -> float:
    if not pairs:
        return 0.0
    maximum = 0.0
    for left, right, join_sign in pairs:
        for order in range(required_order):
            left_value = left.w_derivative(1.0, order) / left.parameter_width**order
            right_value = right.w_derivative(0.0, order) / right.parameter_width**order
            residual = abs(left_value - join_sign * right_value)
            scale = max(1.0, abs(left_value), abs(right_value))
            maximum = max(maximum, residual / scale)
    degree = pairs[0][0].preimage_degree
    # Endpoint derivatives are alternating Bernstein differences.  Their
    # forward error grows with both derivative order and the binomial sum;
    # 4**r is a conservative bound for the two one-sided difference ladders.
    tolerance = 4096.0 * _EPS * max(1, 4**required_order) * max(1, degree * degree)
    if maximum > tolerance:
        raise ContinuityVerificationError(
            "Independent preimage-jet verification failed",
            quantity="relative preimage jet residual",
            value=maximum,
            bound=f"<= {tolerance:.3e}",
        )
    return maximum


def _verify_continuity(
    spans: tuple[PHBSplineSpan, ...],
    required_order: int,
    closed: bool,
    seam_sign: int = 1,
) -> float:
    pairs = [(left, right, 1) for left, right in pairwise(spans)]
    if closed:
        pairs.append((spans[-1], spans[0], seam_sign))
    return _verify_continuity_pairs(pairs, required_order)


def _span_partition(
    widths: NDArray[np.float64], numerics: NumericalPolicy
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return the public and midpoint-refined parameter partitions."""

    parameter_total = math.fsum(float(value) for value in widths)
    if not parameter_total > 0.0 or not math.isfinite(parameter_total):
        raise NumericalPrecisionError(
            "Parameter total is not positive and finite",
            quantity="parameter total",
            value=parameter_total,
        )
    raw_prefix = np.asarray(
        compensated_prefix_sums([float(value) for value in widths]),
        dtype=np.float64,
    )
    knots = raw_prefix / raw_prefix[-1]
    knots[0] = 0.0
    knots[-1] = 1.0
    if not np.all(np.diff(knots) > 0.0):
        raise NumericalPrecisionError(
            "Global PH B-spline parameters are not strictly increasing",
            quantity="diff(knots)",
            bound="> 0",
        )
    span_knots = np.empty(2 * widths.size + 1, dtype=np.float64)
    span_knots[0::2] = knots
    span_knots[1::2] = 0.5 * (knots[:-1] + knots[1:])
    if not np.all(np.diff(span_knots) > 0.0):
        raise NumericalPrecisionError(
            "Compiled PH B-spline parameters are not strictly increasing",
            quantity="diff(span_knots)",
            bound="> 0",
        )
    return parameter_total, knots, span_knots


def _changed_intervals(
    old: PHBSplineBuildState,
    normalized: NDArray[np.complex128],
    widths: NDArray[np.float64],
    point_ids: tuple[int, ...],
    closed: bool,
) -> tuple[int, ...]:
    old_index = {
        key: span_id // 2
        for span_id, key in enumerate(old.span_keys)
        if key[2] == 0
    }
    changed: list[int] = []
    for interval in range(widths.size):
        endpoint = (interval + 1) % normalized.size
        key = (point_ids[interval], point_ids[endpoint], 0)
        previous = old_index.get(key)
        if (
            previous is None
            or normalized[interval] != old.normalized_points[previous]
            or normalized[endpoint]
            != old.normalized_points[(previous + 1) % old.normalized_points.size]
            or widths[interval] != old.raw_widths[previous]
        ):
            changed.append(interval)
    return tuple(changed)


def _local_patch_intervals(
    changed: tuple[int, ...], interval_count: int, minimum: int, closed: bool
) -> tuple[int, ...]:
    """Choose the shortest contiguous patch, then add symmetric guard spans."""

    if not changed:
        return ()
    target = min(interval_count, max(minimum, len(changed)))
    if not closed:
        first = min(changed)
        stop = max(changed) + 1
        while stop - first < target:
            if first > 0:
                first -= 1
            if stop - first < target and stop < interval_count:
                stop += 1
            if first == 0 and stop == interval_count:
                break
        return tuple(range(first, stop))

    ordered = sorted(changed)
    largest_distance = -1
    start = ordered[0]
    for index, current in enumerate(ordered):
        following = ordered[(index + 1) % len(ordered)]
        if index + 1 == len(ordered):
            following += interval_count
        distance = following - current
        if distance > largest_distance:
            largest_distance = distance
            start = following % interval_count
    count = interval_count - largest_distance + 1
    while count < target:
        start = (start - 1) % interval_count
        count += 1
        if count < target:
            count += 1
    return tuple((start + offset) % interval_count for offset in range(count))


def _physical_endpoint_jet(
    span: PHBSplineSpan, endpoint: int, required_order: int
) -> NDArray[np.complex128]:
    return np.asarray(
        [
            span.w_derivative(float(endpoint), order)
            / span.parameter_width**order
            for order in range(required_order)
        ],
        dtype=np.complex128,
    )


def _negate_span_preimage(span: PHBSplineSpan) -> PHBSplineSpan:
    """Change only the square-root gauge; PH geometry and metric are unchanged."""

    return replace(
        span,
        preimage=_readonly(-span.preimage),
        preimage_left_jet=_readonly(-span.preimage_left_jet),
        preimage_right_jet=_readonly(-span.preimage_right_jet),
    )


def build_ph_bspline_local_state(
    old: PHBSplineBuildState,
    points_input: object,
    *,
    closed: bool,
    g_order: object,
    c_order: object,
    curvature_order: object,
    construction: ConstructionPolicy,
    inverse: InversePolicy,
    numerics: NumericalPolicy,
    point_ids: tuple[int, ...],
) -> PHBSplineBuildState:
    """Reconstruct only a bounded simple-knot neighborhood of an edit.

    The old exterior spans are retained bitwise.  The patch is a clamped
    minimum-degree B-spline whose boundary preimage jets are fixed to the
    adjacent certified exterior; exact PH displacement constraints then
    interpolate every edited point inside the patch.
    """

    required_order, requested = validate_continuity(
        g_order, c_order, curvature_order
    )
    points = validate_bspline_points(points_input, closed=closed)
    normalized, origin, scale, normalized_lengths = _normalize_points(
        points, closed, (old.origin, old.scale)
    )
    widths = _parameter_widths(normalized_lengths, construction)
    if required_order > 1:
        allowed_ratio = (1.0e8) ** (1.0 / (required_order - 1))
        largest_width = float(np.max(widths))
        widths = np.maximum(widths, largest_width / allowed_ratio)
    changed = _changed_intervals(old, normalized, widths, point_ids, closed)
    if not changed and widths.size != old.raw_widths.size:
        # Removing an open endpoint deletes one whole interval but does not
        # alter any surviving edge key.  Force a boundary patch so the new
        # points, handles, partitions, and metric state are still published.
        old_first_id = old.span_keys[0][0]
        changed = (0,) if point_ids[0] != old_first_id else (widths.size - 1,)
    patch = _local_patch_intervals(
        changed, widths.size, required_order + 3, closed
    )
    if not patch:
        return old
    if closed and len(patch) == widths.size:
        raise ContinuityVerificationError(
            "The closed edit leaves no certified exterior for local repair",
            quantity="local patch interval count",
            value=len(patch),
            bound=f"< {widths.size}",
        )

    patch_point_indices = [patch[0]]
    for interval in patch:
        patch_point_indices.append((interval + 1) % normalized.size)
    patch_normalized = np.asarray(
        normalized[patch_point_indices], dtype=np.complex128
    )
    patch_widths = np.asarray(widths[list(patch)], dtype=np.float64)
    patch_ids = tuple(point_ids[index] for index in patch_point_indices)
    basis = build_preimage_basis(patch_widths, required_order, False)
    patch_guide, _ = _guide_preimage(patch_normalized, patch_widths, False)
    initial = guide_controls(
        basis, patch_guide
    )
    old_by_key = {
        key: span for key, span in zip(old.span_keys, old.spans, strict=True)
    }
    first_interval = patch[0]
    last_interval = patch[-1]
    wrap_at = next(
        (
            index
            for index in range(1, len(patch))
            if patch[index] < patch[index - 1]
        ),
        None,
    )
    patch_gauge = (
        old.preimage_seam_sign if closed and first_interval == 0 else 1
    )
    right_gauge = patch_gauge
    if wrap_at is not None:
        right_gauge *= old.preimage_seam_sign
    if closed and last_interval == widths.size - 1:
        right_gauge *= old.preimage_seam_sign
    left_jet = None
    right_jet = None
    if closed or first_interval > 0:
        before = (first_interval - 1) % widths.size
        before_endpoint = (before + 1) % normalized.size
        key = (point_ids[before], point_ids[before_endpoint], 1)
        left_jet = _physical_endpoint_jet(old_by_key[key], 1, required_order)
    if closed or last_interval + 1 < widths.size:
        after = (last_interval + 1) % widths.size
        after_endpoint = (after + 1) % normalized.size
        key = (point_ids[after], point_ids[after_endpoint], 0)
        right_jet = _physical_endpoint_jet(old_by_key[key], 0, required_order)
        right_jet *= right_gauge

    normalized_bound = (
        numerics.position_eps_factor
        * _EPS
        * max(1.0, float(np.max(np.abs(normalized))))
        * (required_order + 1) ** 2
    )
    chords = np.diff(patch_normalized)
    solution = solve_preimage(
        basis,
        chords,
        initial,
        max_iterations=construction.max_iterations,
        max_line_search_steps=construction.max_line_search_steps,
        tolerance=0.25 * normalized_bound,
        left_jet=left_jet,
        right_jet=right_jet,
    )
    if solution.max_displacement_residual > normalized_bound:
        raise InterpolationVerificationError(
            "Local minimum-degree PH B-spline solve did not converge",
            quantity="normalized displacement residual",
            value=solution.max_displacement_residual,
            bound=f"<= {normalized_bound:.3e}",
        )
    local_spans, _, patch_ratio, patch_residual = _construct_basis_spans(
        normalized=patch_normalized,
        basis=basis,
        solution=solution,
        required_order=required_order,
        closed=False,
        numerics=numerics,
        inverse=inverse,
        point_ids=patch_ids,
        reusable=None,
    )
    local_spans = list(local_spans)
    if left_jet is not None:
        local_spans[0] = replace(
            local_spans[0],
            preimage_left_jet=_readonly(
                np.asarray(
                    [
                        left_jet[order] * basis.span_widths[0] ** order
                        for order in range(required_order)
                    ],
                    dtype=np.complex128,
                )
            ),
        )
    if right_jet is not None:
        local_spans[-1] = replace(
            local_spans[-1],
            preimage_right_jet=_readonly(
                np.asarray(
                    [
                        right_jet[order] * basis.span_widths[-1] ** order
                        for order in range(required_order)
                    ],
                    dtype=np.complex128,
                )
            ),
        )
    gauge = patch_gauge
    for local_interval in range(len(patch)):
        if wrap_at == local_interval:
            gauge *= old.preimage_seam_sign
        if gauge == -1:
            for subspan in (0, 1):
                span_index = 2 * local_interval + subspan
                local_spans[span_index] = _negate_span_preimage(
                    local_spans[span_index]
                )

    patch_lookup: dict[tuple[int, int, int], PHBSplineSpan] = {}
    for local_interval, interval in enumerate(patch):
        endpoint = (interval + 1) % normalized.size
        for subspan in (0, 1):
            patch_lookup[(point_ids[interval], point_ids[endpoint], subspan)] = (
                local_spans[2 * local_interval + subspan]
            )
    assembled: list[PHBSplineSpan] = []
    span_keys: list[tuple[int, int, int]] = []
    patch_set = set(patch)
    for interval in range(widths.size):
        endpoint = (interval + 1) % normalized.size
        for subspan in (0, 1):
            key = (point_ids[interval], point_ids[endpoint], subspan)
            source = patch_lookup[key] if interval in patch_set else old_by_key[key]
            span_id = 2 * interval + subspan
            assembled.append(
                source if source.span_id == span_id else replace(source, span_id=span_id)
            )
            span_keys.append(key)
    spans = tuple(assembled)

    rebuilt_span_ids = {
        2 * interval + subspan for interval in patch for subspan in (0, 1)
    }
    join_ids: set[int] = set()
    for span_id in rebuilt_span_ids:
        if span_id > 0:
            join_ids.add(span_id - 1)
        elif closed:
            join_ids.add(len(spans) - 1)
        if span_id + 1 < len(spans) or closed:
            join_ids.add(span_id)
    continuity_pairs = [
        (
            spans[left],
            spans[(left + 1) % len(spans)],
            (
                old.preimage_seam_sign
                if closed and left == len(spans) - 1
                else 1
            ),
        )
        for left in sorted(join_ids)
    ]
    continuity_residual = _verify_continuity_pairs(
        continuity_pairs, required_order
    )
    max_residual = max(patch_residual, solution.max_displacement_residual)
    if max_residual > normalized_bound:
        raise InterpolationVerificationError(
            "Compiled local PH spans miss interpolation points",
            quantity="normalized endpoint residual",
            value=max_residual,
            bound=f"<= {normalized_bound:.3e}",
        )

    prefix_normalized = np.asarray(
        compensated_prefix_sums([span.length for span in spans]), dtype=np.float64
    )
    if not np.all(np.diff(prefix_normalized) > 0.0):
        raise LengthResolutionError(
            "Normalized PH B-spline prefix lengths are not strictly increasing",
            quantity="diff(prefix_normalized)",
            bound="> 0",
        )
    prefix_user = scale * prefix_normalized
    if not np.all(np.isfinite(prefix_user)):
        raise NumericalPrecisionError(
            "PH B-spline length overflows user coordinates",
            quantity="scale * prefix length",
            bound="finite",
        )
    if numerics.reject_unresolved_global_lengths and not np.all(
        np.diff(prefix_user) > 0.0
    ):
        raise LengthResolutionError(
            "User-space PH B-spline prefix lengths are not resolvable",
            quantity="diff(prefix_user)",
            bound="> 0",
        )
    parameter_total, knots, span_knots = _span_partition(widths, numerics)
    verified = ContinuitySpec(
        g_order=required_order,
        c_order=required_order,
        curvature_order=max(0, required_order - 2),
    )
    continuity_bound = (
        4096.0
        * _EPS
        * max(1, 4**required_order)
        * max(1, required_order * required_order)
    )
    diagnostics = BuildDiagnostics(
        point_count=points.shape[0],
        span_count=len(spans),
        hidden_span_count=widths.size,
        preimage_degree=required_order,
        iterations=solution.iterations,
        refinement_rounds=0,
        max_interpolation_residual=max_residual * scale,
        interpolation_bound=normalized_bound * scale,
        max_continuity_residual=max(
            continuity_residual, old.diagnostics.max_continuity_residual
        ),
        continuity_bound=continuity_bound,
        min_regularity_ratio=min(patch_ratio, old.diagnostics.min_regularity_ratio),
        max_inverse_residual_ratio=0.0,
        max_lut_nodes=max(span.lut_u.size for span in spans),
        longdouble_verification_used=False,
    )
    return PHBSplineBuildState(
        points=_readonly(points),
        normalized_points=_readonly(normalized),
        origin=origin,
        scale=scale,
        raw_widths=_readonly(widths),
        span_widths=_readonly(np.repeat(widths * 0.5, 2)),
        parameter_total=parameter_total,
        knots=_readonly(knots),
        span_knots=_readonly(span_knots),
        spans=spans,
        span_keys=tuple(span_keys),
        preimage_controls=solution.controls,
        preimage_seam_sign=old.preimage_seam_sign,
        prefix_normalized=_readonly(prefix_normalized),
        prefix_user=_readonly(prefix_user),
        total_length=float(prefix_user[-1]),
        preimage_degree=required_order,
        requested_continuity=requested,
        verified_continuity=verified,
        diagnostics=diagnostics,
    )


def build_ph_bspline_state(
    points_input: object,
    *,
    closed: bool,
    g_order: object,
    c_order: object,
    curvature_order: object,
    construction: ConstructionPolicy,
    inverse: InversePolicy,
    numerics: NumericalPolicy,
    point_ids: tuple[int, ...] | None = None,
    reusable: dict[
        tuple[int, int, int], tuple[PHBSplineSpan, complex, complex]
    ]
    | None = None,
    normalization_frame: tuple[tuple[float, float], float] | None = None,
) -> PHBSplineBuildState:
    """Build, independently verify, and return unpublished spline state."""

    if not isinstance(closed, (bool, np.bool_)):
        raise TypeError("closed must be a Boolean")
    required_order, requested = validate_continuity(g_order, c_order, curvature_order)
    actual_degree = required_order
    if actual_degree > numerics.max_preimage_degree:
        raise ResourceLimitError(
            "Requested continuity requires a preimage degree above policy",
            quantity="preimage degree",
            value=actual_degree,
            bound=f"<= {numerics.max_preimage_degree}",
        )
    points = validate_bspline_points(points_input, closed=bool(closed))
    normalized, origin, scale, normalized_lengths = _normalize_points(
        points, bool(closed), normalization_frame
    )
    widths = _parameter_widths(normalized_lengths, construction)
    # Very high derivative orders amplify adjacent parameter-width ratios as
    # h**(-r).  Clamp only the numerically unresolvable tail of the requested
    # parameterization; geometry and interpolation remain unchanged, while
    # the public parameter stays strictly increasing and C^r jets remain
    # representable in binary64.
    if required_order > 1:
        allowed_ratio = (1.0e8) ** (1.0 / (required_order - 1))
        largest_width = float(np.max(widths))
        minimum_width = largest_width / allowed_ratio
        if float(np.min(widths)) < minimum_width:
            widths = np.maximum(widths, minimum_width)
    parameter_total = math.fsum(float(value) for value in widths)
    if not parameter_total > 0.0 or not math.isfinite(parameter_total):
        raise NumericalPrecisionError(
            "Parameter total is not positive and finite",
            quantity="parameter total",
            value=parameter_total,
        )
    normalized_bound = (
        numerics.position_eps_factor
        * _EPS
        * max(1.0, float(np.max(np.abs(normalized))))
        * (actual_degree + 1) ** 2
    )
    guide, seam_sign = _guide_preimage(normalized, widths, bool(closed))
    basis = build_preimage_basis(
        widths,
        actual_degree,
        bool(closed),
        seam_sign=seam_sign,
    )
    initial = guide_controls(basis, guide)
    chords = np.asarray(
        [
            normalized[(index + 1) % normalized.size] - normalized[index]
            for index in range(widths.size)
        ],
        dtype=np.complex128,
    )
    solution = solve_preimage(
        basis,
        chords,
        initial,
        max_iterations=construction.max_iterations,
        max_line_search_steps=construction.max_line_search_steps,
        tolerance=0.25 * normalized_bound,
    )
    if solution.max_displacement_residual > normalized_bound:
        raise InterpolationVerificationError(
            "Minimum-degree PH B-spline displacement solve did not converge",
            quantity="normalized displacement residual",
            value=solution.max_displacement_residual,
            bound=f"<= {normalized_bound:.3e}",
        )
    if point_ids is None:
        point_ids = tuple(range(normalized.size))
    spans, span_keys, min_ratio, max_residual = _construct_basis_spans(
        normalized=normalized,
        basis=basis,
        solution=solution,
        required_order=required_order,
        closed=bool(closed),
        numerics=numerics,
        inverse=inverse,
        point_ids=point_ids,
        reusable=reusable,
    )
    max_residual = max(max_residual, solution.max_displacement_residual)
    continuity_residual = _verify_continuity(
        spans,
        required_order,
        bool(closed),
        seam_sign,
    )
    if max_residual > normalized_bound:
        raise InterpolationVerificationError(
            "Compiled PH spans miss interpolation points",
            quantity="normalized endpoint residual",
            value=max_residual,
            bound=f"<= {normalized_bound:.3e}",
        )
    prefix_normalized = np.asarray(
        compensated_prefix_sums([span.length for span in spans]), dtype=np.float64
    )
    if not np.all(np.diff(prefix_normalized) > 0.0):
        raise LengthResolutionError(
            "Normalized PH B-spline prefix lengths are not strictly increasing",
            quantity="diff(prefix_normalized)",
            bound="> 0",
        )
    prefix_user = scale * prefix_normalized
    if not np.all(np.isfinite(prefix_user)):
        raise NumericalPrecisionError(
            "PH B-spline length overflows user coordinates",
            quantity="scale * prefix length",
            bound="finite",
        )
    if numerics.reject_unresolved_global_lengths and not np.all(
        np.diff(prefix_user) > 0.0
    ):
        raise LengthResolutionError(
            "User-space PH B-spline prefix lengths are not resolvable",
            quantity="diff(prefix_user)",
            bound="> 0",
        )
    raw_prefix = np.asarray(
        compensated_prefix_sums([float(value) for value in widths]),
        dtype=np.float64,
    )
    knots = raw_prefix / raw_prefix[-1]
    knots[0] = 0.0
    knots[-1] = 1.0
    if not np.all(np.diff(knots) > 0.0):
        raise NumericalPrecisionError(
            "Global PH B-spline parameters are not strictly increasing",
            quantity="diff(knots)",
            bound="> 0",
        )
    span_knots = np.empty(2 * widths.size + 1, dtype=np.float64)
    span_knots[0::2] = knots
    span_knots[1::2] = 0.5 * (knots[:-1] + knots[1:])
    if not np.all(np.diff(span_knots) > 0.0):
        raise NumericalPrecisionError(
            "Compiled PH B-spline parameters are not strictly increasing",
            quantity="diff(span_knots)",
            bound="> 0",
        )
    verified = ContinuitySpec(
        g_order=required_order,
        c_order=required_order,
        curvature_order=max(0, required_order - 2),
    )
    diagnostics = BuildDiagnostics(
        point_count=points.shape[0],
        span_count=len(spans),
        hidden_span_count=len(spans) - widths.size,
        preimage_degree=actual_degree,
        iterations=solution.iterations,
        refinement_rounds=0,
        max_interpolation_residual=max_residual * scale,
        interpolation_bound=normalized_bound * scale,
        max_continuity_residual=continuity_residual,
        continuity_bound=(
            4096.0
            * _EPS
            * max(1, 4**required_order)
            * max(1, actual_degree * actual_degree)
        ),
        min_regularity_ratio=min_ratio,
        max_inverse_residual_ratio=0.0,
        max_lut_nodes=max(span.lut_u.size for span in spans),
        longdouble_verification_used=False,
    )
    return PHBSplineBuildState(
        points=_readonly(points),
        normalized_points=_readonly(normalized),
        origin=origin,
        scale=scale,
        raw_widths=_readonly(widths),
        span_widths=_readonly(basis.span_widths),
        parameter_total=parameter_total,
        knots=_readonly(knots),
        span_knots=_readonly(span_knots),
        spans=spans,
        span_keys=span_keys,
        preimage_controls=solution.controls,
        preimage_seam_sign=seam_sign,
        prefix_normalized=_readonly(prefix_normalized),
        prefix_user=_readonly(prefix_user),
        total_length=float(prefix_user[-1]),
        preimage_degree=actual_degree,
        requested_continuity=requested,
        verified_continuity=verified,
        diagnostics=diagnostics,
    )


__all__ = [
    "PHBSplineBuildState",
    "build_ph_bspline_local_state",
    "build_ph_bspline_state",
    "validate_bspline_points",
    "validate_continuity",
]
