"""Simple-knot complex B-spline basis and constrained PH interpolation.

The interpolation unknown is the complex preimage B-spline ``w``.  Every
user interval is split once at its parameter midpoint.  The additional
simple knot supplies local shape freedom without raising the preimage degree;
the PH displacement over the two extracted spans is still constrained to the
corresponding input chord exactly.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from ph_spline.bspline_segment import derivative_controls


@dataclass(frozen=True, slots=True)
class PreimageBasis:
    """Bezier extraction data for one degree and parameter partition."""

    degree: int
    closed: bool
    user_widths: NDArray[np.float64]
    span_widths: NDArray[np.float64]
    user_breaks: NDArray[np.float64]
    span_breaks: NDArray[np.float64]
    spline_knots: NDArray[np.float64]
    control_count: int
    control_greville: NDArray[np.float64]
    extraction_indices: tuple[NDArray[np.int64], ...]
    extraction_matrices: tuple[NDArray[np.float64], ...]
    span_to_user: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PreimageSolution:
    """A converged constrained preimage and its extracted span controls."""

    controls: NDArray[np.complex128]
    span_controls: tuple[NDArray[np.complex128], ...]
    iterations: int
    max_displacement_residual: float


def _readonly(values: NDArray) -> NDArray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _basis_derivatives_at_span_start(
    knots: NDArray[np.float64], degree: int, span: int
) -> NDArray[np.float64]:
    """Evaluate every active basis derivative at a span's left endpoint.

    This is Algorithm A2.3 from *The NURBS Book*, specialized to all
    derivatives through the degree.  Direct endpoint jets avoid the badly
    conditioned Bernstein collocation inverse that otherwise amplifies a few
    ulps into visible G8 join residuals.
    """

    value = float(knots[span])
    ndu = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    left = np.zeros(degree + 1, dtype=np.float64)
    right = np.zeros(degree + 1, dtype=np.float64)
    ndu[0, 0] = 1.0
    for j in range(1, degree + 1):
        left[j] = value - float(knots[span + 1 - j])
        right[j] = float(knots[span + j]) - value
        saved = 0.0
        for r in range(j):
            denominator = right[r + 1] + left[j - r]
            ndu[j, r] = denominator
            temporary = 0.0 if denominator == 0.0 else ndu[r, j - 1] / denominator
            ndu[r, j] = saved + right[r + 1] * temporary
            saved = left[j - r] * temporary
        ndu[j, j] = saved

    derivatives = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    derivatives[0, :] = ndu[:, degree]
    work = np.zeros((2, degree + 1), dtype=np.float64)
    for r in range(degree + 1):
        first = 0
        second = 1
        work[0, 0] = 1.0
        for order in range(1, degree + 1):
            derivative = 0.0
            rk = r - order
            pk = degree - order
            if r >= order:
                work[second, 0] = work[first, 0] / ndu[pk + 1, rk]
                derivative = work[second, 0] * ndu[rk, pk]
            lower = 1 if rk >= -1 else -rk
            upper = order - 1 if r - 1 <= pk else degree - r
            for j in range(lower, upper + 1):
                work[second, j] = (
                    work[first, j] - work[first, j - 1]
                ) / ndu[pk + 1, rk + j]
                derivative += work[second, j] * ndu[rk + j, pk]
            if r <= pk:
                work[second, order] = -work[first, order - 1] / ndu[
                    pk + 1, r
                ]
                derivative += work[second, order] * ndu[r, pk]
            derivatives[order, r] = derivative
            first, second = second, first

    factor = degree
    for order in range(1, degree + 1):
        derivatives[order, :] *= factor
        factor *= degree - order
    return derivatives


def _bezier_extraction_at_span(
    knots: NDArray[np.float64], degree: int, span: int, width: float
) -> NDArray[np.float64]:
    """Return the active B-spline-to-Bernstein extraction matrix."""

    derivatives = _basis_derivatives_at_span_start(knots, degree, span)
    power = np.empty_like(derivatives)
    scale = 1.0
    for order in range(degree + 1):
        if order:
            scale *= width / order
        power[order, :] = derivatives[order, :] * scale
    extraction = np.zeros_like(power)
    for bernstein_index in range(degree + 1):
        for order in range(bernstein_index + 1):
            extraction[bernstein_index, :] += (
                math.comb(bernstein_index, order)
                / math.comb(degree, order)
                * power[order, :]
            )
    return extraction


@cache
def _integral_gram(degree: int) -> NDArray[np.float64]:
    result = np.empty((degree + 1, degree + 1), dtype=np.float64)
    for left in range(degree + 1):
        for right in range(degree + 1):
            result[left, right] = (
                math.comb(degree, left)
                * math.comb(degree, right)
                / ((2 * degree + 1) * math.comb(2 * degree, left + right))
            )
    result.setflags(write=False)
    return result


def _periodic_break(
    widths: NDArray[np.float64], base_breaks: NDArray[np.float64], index: int
) -> float:
    count = widths.size
    if index < 0:
        return -math.fsum(float(widths[j % count]) for j in range(index, 0))
    if index <= count:
        return float(base_breaks[index])
    return float(base_breaks[-1]) + math.fsum(
        float(widths[j % count]) for j in range(count, index)
    )


def build_preimage_basis(
    user_widths: NDArray[np.float64], degree: int, closed: bool
) -> PreimageBasis:
    """Build a degree-``degree`` simple-knot basis and Bezier extraction."""

    # One midpoint knot per interpolation interval provides enough local
    # freedom for robust interpolation and subsequent bounded patch repair,
    # while retaining the minimum requested preimage degree.
    span_widths = np.repeat(np.asarray(user_widths, dtype=np.float64) * 0.5, 2)
    span_breaks = np.asarray(
        [0.0, *np.cumsum(span_widths, dtype=np.float64)], dtype=np.float64
    )
    user_breaks = np.asarray(
        [0.0, *np.cumsum(user_widths, dtype=np.float64)], dtype=np.float64
    )
    span_count = span_widths.size
    if closed:
        spline_knots = np.asarray(
            [
                _periodic_break(span_widths, span_breaks, index)
                for index in range(-degree, span_count + degree + 1)
            ],
            dtype=np.float64,
        )
        control_count = span_count
    else:
        spline_knots = np.concatenate(
            (
                np.repeat(span_breaks[0], degree + 1),
                span_breaks[1:-1],
                np.repeat(span_breaks[-1], degree + 1),
            )
        )
        control_count = span_count + degree

    extraction_indices: list[NDArray[np.int64]] = []
    extraction_matrices: list[NDArray[np.float64]] = []
    width = degree + 1
    for span in range(span_count):
        knot_span = degree + span
        local = _bezier_extraction_at_span(
            spline_knots, degree, knot_span, float(span_widths[span])
        )
        if closed:
            raw_indices = np.arange(span, span + width, dtype=np.int64) % control_count
            unique, inverse = np.unique(raw_indices, return_inverse=True)
            collapsed = np.zeros((width, unique.size), dtype=np.float64)
            for column, target in enumerate(inverse):
                collapsed[:, target] += local[:, column]
            indices = unique
            local = collapsed
        else:
            indices = np.arange(span, span + width, dtype=np.int64)
        extraction_indices.append(_readonly(indices))
        extraction_matrices.append(_readonly(local))

    extended_control_count = spline_knots.size - degree - 1
    greville = np.asarray(
        [
            math.fsum(float(value) for value in spline_knots[j + 1 : j + degree + 1])
            / degree
            for j in range(extended_control_count)
        ],
        dtype=np.float64,
    )
    if closed:
        greville = np.mod(greville[:control_count], span_breaks[-1])

    return PreimageBasis(
        degree=degree,
        closed=closed,
        user_widths=_readonly(user_widths),
        span_widths=_readonly(span_widths),
        user_breaks=_readonly(user_breaks),
        span_breaks=_readonly(span_breaks),
        spline_knots=_readonly(spline_knots),
        control_count=control_count,
        control_greville=_readonly(greville),
        extraction_indices=tuple(extraction_indices),
        extraction_matrices=tuple(extraction_matrices),
        span_to_user=_readonly(np.arange(span_count, dtype=np.int64) // 2),
    )


def guide_controls(
    basis: PreimageBasis, guide: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Return a deterministic Greville seed interpolated from guide roots."""

    x = basis.control_greville
    if basis.closed:
        period = float(basis.user_breaks[-1])
        sample_x = np.concatenate((basis.user_breaks[:-1], [period]))
        sample_y = np.concatenate((guide, guide[:1]))
        x = np.mod(x, period)
    else:
        sample_x = basis.user_breaks
        sample_y = guide
    real = np.interp(x, sample_x, sample_y.real)
    imag = np.interp(x, sample_x, sample_y.imag)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def _constraints_and_jacobian(
    controls: NDArray[np.complex128],
    basis: PreimageBasis,
    chords: NDArray[np.complex128],
    left_jet: NDArray[np.complex128] | None = None,
    right_jet: NDArray[np.complex128] | None = None,
) -> tuple[NDArray[np.complex128], csr_matrix]:
    user_count = chords.size
    control_count = basis.control_count
    jet_count = (0 if left_jet is None else left_jet.size) + (
        0 if right_jet is None else right_jet.size
    )
    values = np.zeros(user_count + jet_count, dtype=np.complex128)
    gram = _integral_gram(basis.degree)
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    for span, (indices, extraction) in enumerate(
        zip(basis.extraction_indices, basis.extraction_matrices)
    ):
        user = int(basis.span_to_user[span])
        local_controls = controls[indices]
        bezier = extraction @ local_controls
        gram_bezier = gram @ bezier
        span_width = float(basis.span_widths[span])
        values[user] += span_width * (bezier @ gram_bezier)
        gradient = 2.0 * span_width * (extraction.T @ gram_bezier)
        for index, derivative in zip(indices, gradient):
            column = 2 * int(index)
            row = 2 * user
            real = float(derivative.real)
            imag = float(derivative.imag)
            rows.extend((row, row, row + 1, row + 1))
            columns.extend((column, column + 1, column, column + 1))
            entries.extend((real, -imag, imag, real))
    values[:user_count] -= chords
    constraint = user_count
    for endpoint, target in ((0, left_jet), (-1, right_jet)):
        if target is None:
            continue
        span = 0 if endpoint == 0 else len(basis.span_widths) - 1
        indices = basis.extraction_indices[span]
        extraction = basis.extraction_matrices[span]
        width = float(basis.span_widths[span])
        for order, wanted in enumerate(target):
            # Constrain dimensionless local-parameter jets.  Scaling the
            # physical t-derivative by h**order is algebraically identical,
            # but avoids a condition number that grows as h**(-order) for
            # G8 patches with small parameter intervals.
            local = derivative_controls(extraction, order)[endpoint]
            values[constraint] = (
                local @ controls[indices] - wanted * width**order
            )
            for index, coefficient in zip(indices, local):
                column = 2 * int(index)
                row = 2 * constraint
                real = float(coefficient)
                rows.extend((row, row + 1))
                columns.extend((column, column + 1))
                entries.extend((real, real))
            constraint += 1
    jacobian = csr_matrix(
        (entries, (rows, columns)),
        shape=(2 * values.size, 2 * control_count),
        dtype=np.float64,
    )
    return values, jacobian


def _real_vector(values: NDArray[np.complex128]) -> NDArray[np.float64]:
    result = np.empty(2 * values.size, dtype=np.float64)
    result[0::2] = values.real
    result[1::2] = values.imag
    return result


def _falling(value: int, order: int) -> int:
    result = 1
    for offset in range(order):
        result *= value - offset
    return result


def _fix_endpoint_controls(
    controls: NDArray[np.complex128],
    basis: PreimageBasis,
    target: NDArray[np.complex128],
    endpoint: int,
) -> NDArray[np.int64]:
    """Impose one clamped endpoint jet and return the fixed control ids."""

    degree = basis.degree
    span = 0 if endpoint == 0 else len(basis.span_widths) - 1
    indices = basis.extraction_indices[span]
    extraction = basis.extraction_matrices[span]
    width = float(basis.span_widths[span])
    bezier = np.empty(degree + 1, dtype=np.complex128)
    if endpoint == 0:
        for index in range(degree):
            bezier[index] = sum(
                math.comb(index, order)
                * width**order
                * target[order]
                / _falling(degree, order)
                for order in range(index + 1)
            )
        fixed = indices[:degree]
        controls[fixed] = np.linalg.solve(
            extraction[:degree, :degree], bezier[:degree]
        )
    else:
        for offset in range(degree):
            index = degree - offset
            bezier[index] = sum(
                (-1) ** order
                * math.comb(offset, order)
                * width**order
                * target[order]
                / _falling(degree, order)
                for order in range(offset + 1)
            )
        fixed = indices[1:]
        controls[fixed] = np.linalg.solve(
            extraction[1:, 1:], bezier[1:]
        )
    return np.asarray(fixed, dtype=np.int64)


def solve_preimage(
    basis: PreimageBasis,
    chords: NDArray[np.complex128],
    initial: NDArray[np.complex128],
    *,
    max_iterations: int,
    max_line_search_steps: int,
    tolerance: float,
    left_jet: NDArray[np.complex128] | None = None,
    right_jet: NDArray[np.complex128] | None = None,
) -> PreimageSolution:
    """Project a guide seed onto all exact PH displacement constraints."""

    controls = np.asarray(initial, dtype=np.complex128).copy()
    if controls.shape != (basis.control_count,):
        raise ValueError("initial preimage controls have the wrong shape")
    fixed_parts: list[NDArray[np.int64]] = []
    if left_jet is not None:
        fixed_parts.append(_fix_endpoint_controls(controls, basis, left_jet, 0))
    if right_jet is not None:
        fixed_parts.append(_fix_endpoint_controls(controls, basis, right_jet, -1))
    fixed = (
        np.unique(np.concatenate(fixed_parts))
        if fixed_parts
        else np.empty(0, dtype=np.int64)
    )
    free_complex = np.setdiff1d(
        np.arange(basis.control_count, dtype=np.int64), fixed, assume_unique=True
    )
    free_real = np.empty(2 * free_complex.size, dtype=np.int64)
    free_real[0::2] = 2 * free_complex
    free_real[1::2] = 2 * free_complex + 1
    iterations = 0
    residual = math.inf
    for iteration in range(max_iterations + 1):
        # Endpoint controls are eliminated from the nonlinear solve, so the
        # boundary jets remain fixed to roundoff instead of competing with
        # displacement rows in an ill-conditioned weighted least-squares
        # system.
        constraints, jacobian = _constraints_and_jacobian(controls, basis, chords)
        residual = float(np.max(np.abs(constraints)))
        if residual <= tolerance:
            iterations = iteration
            break
        if iteration == max_iterations:
            break
        vector = _real_vector(constraints)
        step_vector: NDArray[np.float64] | None = None
        if fixed_parts:
            # Local patches are deliberately small.  Solving J delta = -F
            # directly with rank-revealing SVD avoids squaring the condition
            # number of high-order endpoint-jet rows in J J^T.
            try:
                candidate_free = np.linalg.lstsq(
                    jacobian[:, free_real].toarray(), -vector, rcond=None
                )[0]
            except np.linalg.LinAlgError:
                candidate_free = np.empty(0, dtype=np.float64)
            if candidate_free.shape == (free_real.size,) and np.all(
                np.isfinite(candidate_free)
            ):
                candidate = np.zeros(2 * basis.control_count, dtype=np.float64)
                candidate[free_real] = candidate_free
                step_vector = candidate
        else:
            normal = (jacobian @ jacobian.T).tocsr()
            normal_scale = max(1.0, float(abs(normal).max()))
            for relative_damping in (0.0, 1.0e-15, 1.0e-13, 1.0e-11, 1.0e-9):
                system = normal
                if relative_damping:
                    system = normal + relative_damping * normal_scale * eye(
                        normal.shape[0], format="csr"
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    try:
                        multipliers = spsolve(system, -vector)
                    except (MatrixRankWarning, RuntimeError, ValueError):
                        continue
                candidate = np.asarray(jacobian.T @ multipliers, dtype=np.float64)
                if np.all(np.isfinite(candidate)):
                    step_vector = candidate
                    break
        if step_vector is None:
            break
        step = step_vector[0::2] + 1j * step_vector[1::2]
        baseline = float(np.linalg.norm(vector))
        accepted = False
        fraction = 1.0
        for _ in range(max_line_search_steps + 1):
            trial = controls + fraction * step
            trial_constraints, _ = _constraints_and_jacobian(trial, basis, chords)
            trial_norm = float(np.linalg.norm(_real_vector(trial_constraints)))
            if trial_norm < baseline and math.isfinite(trial_norm):
                controls = trial
                accepted = True
                break
            fraction *= 0.5
        if not accepted:
            break
        iterations = iteration + 1

    span_controls = tuple(
        _readonly(extraction @ controls[indices])
        for indices, extraction in zip(
            basis.extraction_indices, basis.extraction_matrices
        )
    )
    return PreimageSolution(
        controls=_readonly(controls),
        span_controls=span_controls,
        iterations=iterations,
        max_displacement_residual=residual,
    )


__all__ = [
    "PreimageBasis",
    "PreimageSolution",
    "build_preimage_basis",
    "guide_controls",
    "solve_preimage",
]
