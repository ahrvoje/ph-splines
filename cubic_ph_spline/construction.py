"""Input classification, endpoint tangents and PH edge-length construction.

Everything here operates in normalized coordinates (specification
section 5): the origin is the first input point, the scale is the longest
chord, all norms are hypot-based.
"""

from __future__ import annotations

import cmath
import math
import numbers
from dataclasses import dataclass

import numpy as np

from cubic_ph_spline._constants import (
    CHORD_RATIO_MIN,
    COLLINEAR_EPS,
    DELTA_THETA,
    EPS,
    RECON_TOL,
    THETA_UNIQUE,
    TINY,
    X_MIN,
)
from cubic_ph_spline.exceptions import (
    DegeneratePointDataError,
    InsufficientPointDataError,
    InterpolationDomainError,
    InvalidPointDataError,
    NonFiniteCoordinateError,
    NonSimplePointDataError,
    NumericalPrecisionError,
    ReversalError,
    SplineConvergenceError,
)
from cubic_ph_spline.predicates import circumcenter, polyline_self_intersection
from cubic_ph_spline.segment import PHSegment

__all__ = [
    "BoundaryAngles",
    "Geometry",
    "InflectionData",
    "InputGeometry",
    "SplineBlock",
    "SplinePlan",
    "analyze_geometry",
    "block_geometry",
    "boundary_angles",
    "build_curved_segments",
    "build_straight_segments",
    "edge_quantities",
    "initial_internal_fractions",
    "plan_spline",
    "segment_alpha_beta",
    "sinc",
    "validate_points",
]


# ---------------------------------------------------------------------------
# Input validation (spec sections 2.1 and 6.1)
# ---------------------------------------------------------------------------


def _is_real_scalar(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, numbers.Real)


def validate_points(p: object) -> np.ndarray:
    """Validate the constructor input and copy it into a float64 array."""
    if not isinstance(p, list):
        raise InvalidPointDataError(
            "Input must be a list of points",
            quantity="type(p)",
            value=type(p).__name__,
        )
    n = len(p)
    if n < 2:
        raise InsufficientPointDataError(
            "At least two input points are required",
            quantity="len(p)",
            value=n,
            bound=">= 2",
        )
    coords = np.empty((n, 2), dtype=np.float64)
    for i, item in enumerate(p):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise InvalidPointDataError(
                "Each point must be a list or tuple of exactly two real coordinates",
                index=i,
                quantity="point",
                value=item,
            )
        for k in (0, 1):
            c = item[k]
            if not _is_real_scalar(c):
                raise InvalidPointDataError(
                    "Coordinates must be real scalars (Booleans are rejected)",
                    index=i,
                    quantity="coordinate",
                    value=c,
                )
            cf = float(c)
            if not math.isfinite(cf):
                raise NonFiniteCoordinateError(
                    "Coordinates must be finite",
                    index=i,
                    quantity="coordinate",
                    value=cf,
                )
            coords[i, k] = cf
    _check_duplicates(coords)
    return coords


def _check_duplicates(coords: np.ndarray) -> None:
    """Reject consecutive duplicates; block-local duplicates are checked later.

    General data may revisit a point after crossing an inflection or a
    straight sub-run.  Section 6.1 therefore makes nonconsecutive duplicate
    detection a property of each convex sub-problem, not of the input as a
    whole.
    """
    for i in range(1, coords.shape[0]):
        if coords[i, 0] == coords[i - 1, 0] and coords[i, 1] == coords[i - 1, 1]:
            raise DegeneratePointDataError(
                "Consecutive input points coincide",
                index=(i - 1, i),
                quantity="chord length",
                value=0.0,
                bound="> 0",
            )


# ---------------------------------------------------------------------------
# Normalization and classification (spec sections 5 and 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InputGeometry:
    """Normalized geometry and turn classification of the complete input."""

    points: np.ndarray  # original user points, shape (m + 1, 2)
    origin: np.ndarray  # O = P0
    scale: float  # H = max chord length
    phat: np.ndarray  # normalized points, shape (m + 1, 2)
    lhat: np.ndarray  # normalized chord lengths, shape (m,)
    unit: np.ndarray  # unit chord directions, shape (m, 2)
    cross: np.ndarray  # normalized signed turns, shape (m - 1,)
    dot: np.ndarray  # consecutive unit-chord dot products, shape (m - 1,)
    turn_signs: np.ndarray  # -1, 0, +1 after numerical classification
    turn_angles: np.ndarray  # unsigned angles in [0, pi)
    is_straight: bool


@dataclass(frozen=True, slots=True)
class Geometry:
    """One normalized convex block supplied to the unchanged G2 solver."""

    points: np.ndarray
    origin: np.ndarray
    scale: float
    phat: np.ndarray
    lhat: np.ndarray
    unit: np.ndarray
    is_straight: bool
    tau: int
    psi: np.ndarray
    phi: np.ndarray


def analyze_geometry(points: np.ndarray) -> InputGeometry:
    """Normalize once and classify all turns without rejecting inflections."""
    m = points.shape[0] - 1
    with np.errstate(over="ignore", invalid="ignore"):
        deltas = np.diff(points, axis=0)
    if not np.all(np.isfinite(deltas)):
        raise NumericalPrecisionError(
            "Chord vector overflowed during normalization",
            quantity="P[i+1] - P[i]",
        )
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    if not np.all(np.isfinite(lengths)):
        raise NumericalPrecisionError(
            "Chord length overflowed during normalization",
            quantity="|P[i+1] - P[i]|",
        )
    # Exact duplicates were rejected already; a zero here would be a
    # duplicate escaping representation, so guard anyway.
    if np.any(lengths == 0.0):
        i = int(np.argmin(lengths))
        raise DegeneratePointDataError(
            "Consecutive input points coincide",
            index=(i, i + 1),
            quantity="chord length",
            value=0.0,
            bound="> 0",
        )
    H = float(np.max(lengths))
    ratio = float(np.min(lengths)) / H
    if not ratio > CHORD_RATIO_MIN:
        raise NumericalPrecisionError(
            "Shortest chord is not distinguishable from the longest chord "
            "in binary64 arithmetic",
            index=int(np.argmin(lengths)),
            quantity="min|dP| / max|dP|",
            value=ratio,
            bound=f"> {CHORD_RATIO_MIN:.3e}",
        )
    origin = points[0].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        phat = (points - origin) / H
    if not np.all(np.isfinite(phat)):
        raise NumericalPrecisionError(
            "Normalized coordinates overflowed",
            quantity="(P - O) / H",
        )
    lhat = lengths / H
    unit = deltas / lengths[:, None]

    if m == 1:
        # A single chord is a degenerate straight spline by definition.
        return InputGeometry(
            points=points,
            origin=origin,
            scale=H,
            phat=phat,
            lhat=lhat,
            unit=unit,
            cross=np.empty(0),
            dot=np.empty(0),
            turn_signs=np.empty(0, dtype=np.int8),
            turn_angles=np.empty(0),
            is_straight=True,
        )

    cross = unit[:-1, 0] * unit[1:, 1] - unit[:-1, 1] * unit[1:, 0]
    dot = unit[:-1, 0] * unit[1:, 0] + unit[:-1, 1] * unit[1:, 1]
    tiny_turn = np.abs(cross) <= COLLINEAR_EPS

    # A numerically collinear turn pointing backward is a reversal even when
    # other turns in the sequence are material.
    reversal = tiny_turn & (dot < 0.0)
    if np.any(reversal):
        i = int(np.argmax(reversal))
        raise ReversalError(
            "Turn angle of approximately pi at an interior point",
            index=i + 1,
            quantity="normalized turn cross product",
            value=float(cross[i]),
            bound=f"|c| > {COLLINEAR_EPS:.3e} or forward dot > 0",
        )

    signs = np.zeros(cross.shape, dtype=np.int8)
    signs[cross > COLLINEAR_EPS] = 1
    signs[cross < -COLLINEAR_EPS] = -1
    angles = np.zeros(cross.shape, dtype=np.float64)
    material = signs != 0
    angles[material] = np.arctan2(np.abs(cross[material]), dot[material])

    if np.all(tiny_turn):
        # Collinear candidate: every chord must project strictly positively
        # onto the first chord direction (monotone, no backtracking).
        proj = deltas @ unit[0]
        if np.any(proj <= 0.0):
            i = int(np.argmax(proj <= 0.0))
            raise ReversalError(
                "Collinear data backtracks along its line",
                index=i + 1,
                quantity="projected chord length",
                value=float(proj[i]),
                bound="> 0",
            )
        return InputGeometry(
            points=points,
            origin=origin,
            scale=H,
            phat=phat,
            lhat=lhat,
            unit=unit,
            cross=cross,
            dot=dot,
            turn_signs=signs,
            turn_angles=angles,
            is_straight=True,
        )

    return InputGeometry(
        points=points,
        origin=origin,
        scale=H,
        phat=phat,
        lhat=lhat,
        unit=unit,
        cross=cross,
        dot=dot,
        turn_signs=signs,
        turn_angles=angles,
        is_straight=False,
    )


# ---------------------------------------------------------------------------
# General-data preprocessing (spec section 22)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InflectionData:
    """Deterministic auxiliary-point record required by section 22.7."""

    span_index: int
    rho: float
    delta: float
    fallback: bool
    point_hat: tuple[float, float]
    point: tuple[float, float]
    tangent: tuple[float, float]


@dataclass(frozen=True, slots=True)
class SplineBlock:
    """One maximal curved-convex or straight construction block."""

    kind: str  # "curved" or "straight"
    tau: int
    phat: np.ndarray
    points: np.ndarray
    knots: np.ndarray
    start_tangent: tuple[float, float] | None
    end_tangent: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class SplinePlan:
    """Complete deterministic partition of arbitrary admissible input."""

    geometry: InputGeometry
    blocks: tuple[SplineBlock, ...]
    inflections: tuple[InflectionData, ...]


@dataclass(frozen=True, slots=True)
class _AtomicEdge:
    kind: str
    tau: int
    p0_hat: tuple[float, float]
    p1_hat: tuple[float, float]
    p0: tuple[float, float]
    p1: tuple[float, float]
    u0: float
    u1: float
    span_index: int
    # The shared node after this edge: ("input", point index) or
    # ("inflection", input-span index).
    end_marker: tuple[str, int]


def _cross2(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _fallback_inflection(
    geometry: InputGeometry, span_index: int, tau_left: int
) -> InflectionData:
    """The midpoint-and-tilt fallback fixed by section 22.2."""
    i = span_index
    phi_left = float(geometry.turn_angles[i - 1])
    phi_right = float(geometry.turn_angles[i])
    delta = 0.5 * min(phi_left, phi_right, 0.5 * math.pi)
    if not delta > 0.0:
        raise InterpolationDomainError(
            "Inflection fallback produced a nonpositive tangent tilt",
            index=i,
            quantity="delta'",
            value=delta,
            bound="> 0",
        )
    ux = float(geometry.unit[i, 0])
    uy = float(geometry.unit[i, 1])
    angle = tau_left * delta
    ca = math.cos(angle)
    sa = math.sin(angle)
    tx = ca * ux - sa * uy
    ty = sa * ux + ca * uy
    rho = 0.5
    hx = float(geometry.phat[i, 0]) + rho * (
        float(geometry.phat[i + 1, 0]) - float(geometry.phat[i, 0])
    )
    hy = float(geometry.phat[i, 1]) + rho * (
        float(geometry.phat[i + 1, 1]) - float(geometry.phat[i, 1])
    )
    px = float(geometry.points[i, 0]) + rho * (
        float(geometry.points[i + 1, 0]) - float(geometry.points[i, 0])
    )
    py = float(geometry.points[i, 1]) + rho * (
        float(geometry.points[i + 1, 1]) - float(geometry.points[i, 1])
    )
    return InflectionData(
        span_index=i,
        rho=rho,
        delta=delta,
        fallback=True,
        point_hat=(hx, hy),
        point=(px, py),
        tangent=(tx, ty),
    )


def _inflection_data(geometry: InputGeometry, span_index: int) -> InflectionData:
    """Closed-form four-point cubic/chord construction of section 22.2."""
    i = span_index
    tau_left = int(geometry.turn_signs[i - 1])
    fallback = lambda: _fallback_inflection(geometry, i, tau_left)

    # Chord-length knots for P[i-1]..P[i+2], translated so the first is 0.
    h0 = 0.0
    h1 = float(geometry.lhat[i - 1])
    h2 = h1 + float(geometry.lhat[i])
    h3 = h2 + float(geometry.lhat[i + 1])
    xs = (h0, h1, h2, h3)
    zs = tuple(
        complex(float(geometry.phat[j, 0]), float(geometry.phat[j, 1]))
        for j in range(i - 1, i + 3)
    )
    chord = zs[2] - zs[1]
    dx, dy = chord.real, chord.imag

    # w(t)=(t-h1)(t-h2)(a*t+b).  Only the two outer Lagrange
    # basis functions survive because the two span endpoints are known roots.
    y0 = _cross2((zs[0] - zs[1]).real, (zs[0] - zs[1]).imag, dx, dy)
    y3 = _cross2((zs[3] - zs[1]).real, (zs[3] - zs[1]).imag, dx, dy)
    den0 = (h0 - h1) * (h0 - h2) * (h0 - h3)
    den3 = (h3 - h0) * (h3 - h1) * (h3 - h2)
    term0 = y0 / den0
    term3 = y3 / den3
    a_lin = term0 + term3
    scale_lin = abs(term0) + abs(term3)
    if (
        not math.isfinite(a_lin)
        or abs(a_lin) <= 64.0 * EPS * scale_lin
        or scale_lin == 0.0
    ):
        return fallback()
    t_star = (term0 * h3 + term3 * h0) / a_lin
    if not math.isfinite(t_star) or not h1 < t_star < h2:
        return fallback()

    # Newton divided differences for C and C'.  This avoids a power-basis
    # Vandermonde solve and evaluates both quantities with the same model.
    d01 = (zs[1] - zs[0]) / (xs[1] - xs[0])
    d12 = (zs[2] - zs[1]) / (xs[2] - xs[1])
    d23 = (zs[3] - zs[2]) / (xs[3] - xs[2])
    d012 = (d12 - d01) / (xs[2] - xs[0])
    d123 = (d23 - d12) / (xs[3] - xs[1])
    d0123 = (d123 - d012) / (xs[3] - xs[0])
    r0 = t_star - xs[0]
    r1 = t_star - xs[1]
    r2 = t_star - xs[2]
    c_star = zs[0] + d01 * r0 + d012 * r0 * r1 + d0123 * r0 * r1 * r2
    dc_star = d01 + d012 * (r0 + r1) + d0123 * (r1 * r2 + r0 * r2 + r0 * r1)
    dn = math.hypot(dc_star.real, dc_star.imag)
    if not math.isfinite(dn) or dn <= 1024.0 * EPS:
        return fallback()
    tx = dc_star.real / dn
    ty = dc_star.imag / dn
    forward = tx * dx + ty * dy
    oriented_cross = tau_left * _cross2(dx, dy, tx, ty)
    delta = math.atan2(oriented_cross, forward)

    phi_left = float(geometry.turn_angles[i - 1])
    phi_right = float(geometry.turn_angles[i])
    valid = (
        forward > 0.0
        and oriented_cross > 0.0
        and delta >= DELTA_THETA
        and delta + phi_left < THETA_UNIQUE - DELTA_THETA
        and delta + phi_right < THETA_UNIQUE - DELTA_THETA
    )
    if not valid:
        return fallback()

    chord2 = dx * dx + dy * dy
    rho_raw = (
        (c_star.real - zs[1].real) * dx + (c_star.imag - zs[1].imag) * dy
    ) / chord2
    if not math.isfinite(rho_raw):
        return fallback()
    rho = min(max(float(rho_raw), 1.0 / 16.0), 15.0 / 16.0)
    hx = zs[1].real + rho * dx
    hy = zs[1].imag + rho * dy
    px = float(geometry.points[i, 0]) + rho * (
        float(geometry.points[i + 1, 0]) - float(geometry.points[i, 0])
    )
    py = float(geometry.points[i, 1]) + rho * (
        float(geometry.points[i + 1, 1]) - float(geometry.points[i, 1])
    )
    if not all(math.isfinite(v) for v in (hx, hy, px, py, tx, ty)):
        return fallback()
    return InflectionData(
        span_index=i,
        rho=rho,
        delta=delta,
        fallback=False,
        point_hat=(hx, hy),
        point=(px, py),
        tangent=(tx, ty),
    )


def _check_block_simple(block: SplineBlock) -> None:
    """Apply duplicate and proper-intersection checks per convex block."""
    seen: dict[tuple[float, float], int] = {}
    for i, p in enumerate(block.points):
        key = (float(p[0]) + 0.0, float(p[1]) + 0.0)
        previous = seen.get(key)
        if previous is not None:
            raise NonSimplePointDataError(
                "Nonconsecutive points coincide within one convex sub-polyline",
                index=(previous, i),
                quantity="point",
                value=key,
            )
        seen[key] = i
    crossing = polyline_self_intersection(block.phat)
    if crossing is not None:
        raise NonSimplePointDataError(
            "Chords properly intersect within one convex sub-polyline",
            index=crossing,
            quantity="chord pair",
        )


def plan_spline(geometry: InputGeometry) -> SplinePlan:
    """Insert section-22 points and partition the data into solver blocks."""
    m = geometry.points.shape[0] - 1
    input_knots = np.arange(m + 1, dtype=np.float64) / m
    if geometry.is_straight:
        block = SplineBlock(
            kind="straight",
            tau=0,
            phat=np.array(geometry.phat, copy=True),
            points=np.array(geometry.points, copy=True),
            knots=input_knots,
            start_tangent=None,
            end_tangent=None,
        )
        _check_block_simple(block)
        return SplinePlan(geometry=geometry, blocks=(block,), inflections=())

    signs = geometry.turn_signs
    inflection_by_span: dict[int, InflectionData] = {}
    for i in range(1, m - 1):
        if int(signs[i - 1]) * int(signs[i]) < 0:
            inflection_by_span[i] = _inflection_data(geometry, i)

    # A span belongs to a straight sub-run when it touches a zero turn.
    straight_span = np.zeros(m, dtype=bool)
    for k in np.flatnonzero(signs == 0):
        straight_span[k] = True
        straight_span[k + 1] = True

    edges: list[_AtomicEdge] = []

    def endpoint(j: int) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (float(geometry.phat[j, 0]), float(geometry.phat[j, 1])),
            (float(geometry.points[j, 0]), float(geometry.points[j, 1])),
        )

    for i in range(m):
        h0, p0 = endpoint(i)
        h1, p1 = endpoint(i + 1)
        u0, u1 = float(input_knots[i]), float(input_knots[i + 1])
        if straight_span[i]:
            edges.append(
                _AtomicEdge(
                    "straight",
                    0,
                    h0,
                    h1,
                    p0,
                    p1,
                    u0,
                    u1,
                    i,
                    ("input", i + 1),
                )
            )
            continue
        info = inflection_by_span.get(i)
        if info is not None:
            ui = (i + info.rho) / m
            tau_left = int(signs[i - 1])
            tau_right = int(signs[i])
            edges.append(
                _AtomicEdge(
                    "curved",
                    tau_left,
                    h0,
                    info.point_hat,
                    p0,
                    info.point,
                    u0,
                    ui,
                    i,
                    ("inflection", i),
                )
            )
            edges.append(
                _AtomicEdge(
                    "curved",
                    tau_right,
                    info.point_hat,
                    h1,
                    info.point,
                    p1,
                    ui,
                    u1,
                    i,
                    ("input", i + 1),
                )
            )
            continue
        left = int(signs[i - 1]) if i > 0 else 0
        right = int(signs[i]) if i < m - 1 else 0
        tau = left if left != 0 else right
        if tau == 0:  # Defensive: every non-straight span has a material side.
            raise InterpolationDomainError(
                "A span could not be assigned to a curved or straight block",
                index=i,
                quantity="span classification",
            )
        edges.append(
            _AtomicEdge(
                "curved",
                tau,
                h0,
                h1,
                p0,
                p1,
                u0,
                u1,
                i,
                ("input", i + 1),
            )
        )

    # Group maximal blocks.  Straight edges merge only across a classified
    # zero turn; this exposes the impossible straight/straight corner pattern
    # instead of silently returning a G0 joint.
    ranges: list[tuple[int, int]] = []
    start = 0
    for j in range(1, len(edges)):
        left, right = edges[j - 1], edges[j]
        same = left.kind == right.kind and left.tau == right.tau
        if same and left.kind == "straight":
            marker_kind, marker_index = left.end_marker
            same = (
                marker_kind == "input"
                and 1 <= marker_index <= m - 1
                and int(signs[marker_index - 1]) == 0
            )
        if not same:
            ranges.append((start, j))
            start = j
    ranges.append((start, len(edges)))

    mutable: list[dict[str, object]] = []
    for a, b in ranges:
        group = edges[a:b]
        mutable.append(
            {
                "kind": group[0].kind,
                "tau": group[0].tau,
                "phat": np.array([group[0].p0_hat] + [e.p1_hat for e in group]),
                "points": np.array([group[0].p0] + [e.p1 for e in group]),
                "knots": np.array([group[0].u0] + [e.u1 for e in group]),
                "start_tangent": None,
                "end_tangent": None,
                "edge_start": a,
                "edge_end": b,
            }
        )

    for j in range(len(mutable) - 1):
        left = mutable[j]
        right = mutable[j + 1]
        lk, rk = str(left["kind"]), str(right["kind"])
        left_edge = edges[int(left["edge_end"]) - 1]
        right_edge = edges[int(right["edge_start"])]
        if lk == "straight" and rk == "straight":
            marker = left_edge.end_marker
            raise InterpolationDomainError(
                "Two straight sub-runs meet at a material corner; no regular "
                "G1 cubic-PH interpolation exists without an extra point",
                index=marker[1],
                quantity="straight-run tangent discontinuity",
                bound="a curved span between straight sub-runs",
            )
        if lk == "curved" and rk == "curved":
            marker = left_edge.end_marker
            if marker[0] != "inflection":
                raise InterpolationDomainError(
                    "Oppositely oriented convex blocks lack an auxiliary joint",
                    index=marker[1],
                    quantity="block boundary",
                )
            tangent = inflection_by_span[marker[1]].tangent
        else:
            straight_edge = left_edge if lk == "straight" else right_edge
            vx = straight_edge.p1_hat[0] - straight_edge.p0_hat[0]
            vy = straight_edge.p1_hat[1] - straight_edge.p0_hat[1]
            vn = math.hypot(vx, vy)
            tangent = (vx / vn, vy / vn)
        left["end_tangent"] = tangent
        right["start_tangent"] = tangent

    blocks: list[SplineBlock] = []
    for item in mutable:
        block = SplineBlock(
            kind=str(item["kind"]),
            tau=int(item["tau"]),
            phat=np.asarray(item["phat"], dtype=np.float64),
            points=np.asarray(item["points"], dtype=np.float64),
            knots=np.asarray(item["knots"], dtype=np.float64),
            start_tangent=item["start_tangent"],  # type: ignore[arg-type]
            end_tangent=item["end_tangent"],  # type: ignore[arg-type]
        )
        _check_block_simple(block)
        blocks.append(block)
    return SplinePlan(
        geometry=geometry,
        blocks=tuple(blocks),
        inflections=tuple(inflection_by_span[i] for i in sorted(inflection_by_span)),
    )


def block_geometry(plan: SplinePlan, block: SplineBlock) -> Geometry:
    """Create the solver's convex geometry view of one planned block."""
    phat = block.phat
    deltas = np.diff(phat, axis=0)
    lhat = np.hypot(deltas[:, 0], deltas[:, 1])
    unit = deltas / lhat[:, None]
    if block.kind == "straight":
        return Geometry(
            points=block.points,
            origin=plan.geometry.origin,
            scale=plan.geometry.scale,
            phat=phat,
            lhat=lhat,
            unit=unit,
            is_straight=True,
            tau=0,
            psi=np.empty(0),
            phi=np.empty(0),
        )
    tau = block.tau
    if lhat.shape[0] > 1:
        cross = unit[:-1, 0] * unit[1:, 1] - unit[:-1, 1] * unit[1:, 0]
        dot = unit[:-1, 0] * unit[1:, 0] + unit[:-1, 1] * unit[1:, 1]
        phi = np.arctan2(tau * cross, dot)
        if np.any(phi <= 0.0):
            i = int(np.argmax(phi <= 0.0))
            raise InterpolationDomainError(
                "Convex block contains a nonpositive oriented turn",
                index=i + 1,
                quantity="phi",
                value=float(phi[i]),
                bound="> 0",
            )
    else:
        phi = np.empty(0)
    psi = np.empty(lhat.shape[0])
    psi[0] = math.atan2(unit[0, 1], unit[0, 0])
    if phi.size:
        psi[1:] = psi[0] + tau * np.cumsum(phi)
    for i in range(1, phi.size):
        pair = float(phi[i - 1] + phi[i])
        if not pair < THETA_UNIQUE:
            raise InterpolationDomainError(
                f"Interior turn-angle condition failed in a convex block: "
                f"phi[{i}] + phi[{i + 1}] = {pair:.6g} rad",
                index=i + 1,
                quantity="phi[i] + phi[i+1]",
                value=pair,
                bound=f"< {THETA_UNIQUE:.10f} rad",
            )
    return Geometry(
        points=block.points,
        origin=plan.geometry.origin,
        scale=plan.geometry.scale,
        phat=phat,
        lhat=lhat,
        unit=unit,
        is_straight=False,
        tau=tau,
        psi=psi,
        phi=phi,
    )


# ---------------------------------------------------------------------------
# Boundary tangent policy (spec section 7)
# ---------------------------------------------------------------------------


def _wrap_pi(x: float) -> float:
    """Wrap an angle to ``[-pi, pi]``."""
    return math.remainder(x, math.tau)


@dataclass(frozen=True, slots=True)
class BoundaryAngles:
    phi0: float
    phim: float
    theta0: float
    thetam: float
    clamped_start: bool
    clamped_end: bool


def _circumcircle_deviation(
    geometry: Geometry,
    at_start: bool,
    support: np.ndarray | None = None,
) -> float:
    """Raw boundary turn from the three original supporting points."""
    phat = geometry.phat if support is None else support
    tau = geometry.tau
    if at_start:
        a, b, c = phat[0], phat[1], phat[2]
        anchor = geometry.phat[0]
        chord_dir = geometry.unit[0]
        psi_ref = float(geometry.psi[0])
        which = "start"
    else:
        a, b, c = phat[-3], phat[-2], phat[-1]
        anchor = geometry.phat[-1]
        chord_dir = geometry.unit[-1]
        psi_ref = float(geometry.psi[-1])
        which = "end"
    center = circumcenter(
        (float(a[0]), float(a[1])),
        (float(b[0]), float(b[1])),
        (float(c[0]), float(c[1])),
    )
    if center is None:
        raise NumericalPrecisionError(
            f"Circumcircle determinant at the {which} boundary is too small "
            "for reliable computation",
            quantity="circumcircle determinant",
        )
    rx = float(anchor[0]) - center[0]
    ry = float(anchor[1]) - center[1]
    # d_raw = tau * J(P - O_circ), then orient it along traversal exactly as
    # required by section 7.1.
    dx = -tau * ry
    dy = tau * rx
    forward = dx * float(chord_dir[0]) + dy * float(chord_dir[1])
    if forward < 0.0:
        dx = -dx
        dy = -dy
    theta_raw = math.atan2(dy, dx)
    if at_start:
        dev = tau * _wrap_pi(psi_ref - theta_raw)
    else:
        dev = tau * _wrap_pi(theta_raw - psi_ref)
    if not 0.0 < dev < math.pi:
        raise NumericalPrecisionError(
            f"Circumcircle boundary tangent at the {which} is not reliably "
            "oriented into the convex side",
            quantity="raw boundary turn",
            value=dev,
            bound="in (0, pi)",
        )
    return dev


def _prescribed_deviation(
    tangent: tuple[float, float],
    psi: float,
    tau: int,
    *,
    at_start: bool,
) -> float:
    theta = math.atan2(tangent[1], tangent[0])
    dev = tau * _wrap_pi(psi - theta if at_start else theta - psi)
    if not 0.0 < dev < math.pi:
        which = "start" if at_start else "end"
        raise InterpolationDomainError(
            f"Prescribed tangent at the {which} boundary is outside the "
            "strict admissible wedge",
            quantity="prescribed boundary deviation",
            value=dev,
            bound="in (0, pi)",
        )
    return dev


def boundary_angles(
    geometry: Geometry,
    *,
    start_tangent: tuple[float, float] | None = None,
    end_tangent: tuple[float, float] | None = None,
    free_start_support: np.ndarray | None = None,
    free_end_support: np.ndarray | None = None,
) -> BoundaryAngles:
    """Combine free circumcircle and prescribed section-22 boundaries."""
    phi = geometry.phi
    psi = geometry.psi
    tau = geometry.tau

    if start_tangent is None:
        phi0_raw = _circumcircle_deviation(
            geometry, at_start=True, support=free_start_support
        )
    else:
        phi0_raw = _prescribed_deviation(
            start_tangent, float(psi[0]), tau, at_start=True
        )
    if end_tangent is None:
        phim_raw = _circumcircle_deviation(
            geometry, at_start=False, support=free_end_support
        )
    else:
        phim_raw = _prescribed_deviation(
            end_tangent, float(psi[-1]), tau, at_start=False
        )

    strict_limit = THETA_UNIQUE - DELTA_THETA
    if phi.size:
        adjacent0 = float(phi[0])
        adjacentm = float(phi[-1])
        if start_tangent is not None and not phi0_raw + adjacent0 < strict_limit:
            raise InterpolationDomainError(
                "Prescribed start tangent violates the uniqueness condition",
                quantity="phi0 + adjacent turn",
                value=phi0_raw + adjacent0,
                bound=f"< {strict_limit:.10f} rad",
            )
        if end_tangent is not None and not phim_raw + adjacentm < strict_limit:
            raise InterpolationDomainError(
                "Prescribed end tangent violates the uniqueness condition",
                quantity="phim + adjacent turn",
                value=phim_raw + adjacentm,
                bound=f"< {strict_limit:.10f} rad",
            )
        limit0 = strict_limit - adjacent0
        limitm = strict_limit - adjacentm
        phi0 = phi0_raw if start_tangent is not None else min(phi0_raw, limit0)
        phim = phim_raw if end_tangent is not None else min(phim_raw, limitm)
    else:
        # A one-segment convex block has the opposite boundary deviation as
        # its uniqueness neighbour.  Clamp a free boundary first; a fixed
        # prescribed boundary must not be rejected merely because the raw
        # free estimate (which policy permits us to change) was too large.
        phi0 = (
            phi0_raw
            if start_tangent is not None
            else min(phi0_raw, strict_limit - phim_raw)
        )
        phim = (
            phim_raw
            if end_tangent is not None
            else min(phim_raw, strict_limit - phi0_raw)
        )
        pair = phi0 + phim
        has_prescribed = start_tangent is not None or end_tangent is not None
        has_free = start_tangent is None or end_tangent is None
        pair_valid = pair <= strict_limit if has_free else pair < strict_limit
        if has_prescribed and not pair_valid:
            which = "start" if start_tangent is not None else "end"
            raise InterpolationDomainError(
                f"Prescribed {which} tangent violates the uniqueness condition",
                quantity="phi0 + phim",
                value=pair,
                bound=f"< {strict_limit:.10f} rad",
            )
    if not (phi0 > 0.0 and phim > 0.0):
        raise NumericalPrecisionError(
            "Boundary turn clamp produced a nonpositive deviation",
            quantity="clamped boundary turn",
            value=min(phi0, phim),
            bound="> 0",
        )
    theta0 = float(psi[0]) - tau * phi0
    thetam = float(psi[-1]) + tau * phim
    return BoundaryAngles(
        phi0=phi0,
        phim=phim,
        theta0=theta0,
        thetam=thetam,
        clamped_start=start_tangent is None and phi0 < phi0_raw,
        clamped_end=end_tangent is None and phim < phim_raw,
    )


def initial_internal_fractions(geometry: Geometry) -> np.ndarray:
    """Centered-secant initial tangent fractions, projected into the wedge."""
    phat = geometry.phat
    psi = geometry.psi
    phi = geometry.phi
    tau = geometry.tau
    m = psi.shape[0]
    x0 = np.empty(m - 1)
    for i in range(1, m):
        vx = float(phat[i + 1, 0] - phat[i - 1, 0])
        vy = float(phat[i + 1, 1] - phat[i - 1, 1])
        theta = math.atan2(vy, vx)
        rel = tau * _wrap_pi(theta - float(psi[i - 1]))
        x = rel / float(phi[i - 1])
        x0[i - 1] = min(max(x, X_MIN), 1.0 - X_MIN)
    return x0


def chord_weighted_fractions(geometry: Geometry) -> np.ndarray:
    """Fallback initializer: chord-length-weighted tangent fractions.

    For strongly unequal adjacent chords the G2 solution places the joint
    tangent close to the shorter chord's direction, i.e.
    ``x_i ~ l_{i-1} / (l_{i-1} + l_i)``; the centered secant points along
    the longer chord instead and can leave the trust-region solver in a
    log-stiff valley.  This deterministic alternative start is the damped
    initializer permitted by spec section 9.5; the acceptance gate is
    unchanged.
    """
    lhat = geometry.lhat
    x0 = lhat[:-1] / (lhat[:-1] + lhat[1:])
    return np.clip(x0, X_MIN, 1.0 - X_MIN)


# ---------------------------------------------------------------------------
# PH edge lengths from tangent deviations (spec section 8)
# ---------------------------------------------------------------------------


def sinc(x: np.ndarray) -> np.ndarray:
    """``sin(x) / x`` with a series branch near zero; complex-step safe."""
    x = np.asarray(x)
    magnitude = np.abs(x.real) if np.iscomplexobj(x) else np.abs(x)
    small = magnitude < 1e-4
    xs = np.where(small, 1.0, x)
    ratio = np.sin(xs) / xs
    x2 = x * x
    series = 1.0 - (x2 / 6.0) * (1.0 - (x2 / 20.0) * (1.0 - x2 / 42.0))
    return np.where(small, series, ratio)


def segment_alpha_beta(
    x: np.ndarray, phi: np.ndarray, phi0: float, phim: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment oriented endpoint deviations ``alpha_i`` and ``beta_i``.

    ``x`` holds the internal tangent fractions ``x_1..x_{m-1}`` and ``phi``
    the interior turns ``phi_1..phi_{m-1}``; the boundary deviations are the
    fixed clamped boundary turns.
    """
    n = x.shape[0]
    dtype = np.result_type(x, np.float64)
    alpha = np.empty(n + 1, dtype=dtype)
    beta = np.empty(n + 1, dtype=dtype)
    alpha[0] = phi0
    alpha[1:] = (1.0 - x) * phi
    beta[:-1] = x * phi
    beta[-1] = phim
    return alpha, beta


def edge_quantities(
    alpha: np.ndarray,
    beta: np.ndarray,
    lhat: np.ndarray,
    *,
    guarded: bool,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
]:
    """PH edge lengths for every segment.

    Returns ``(xi0, xi1, lam0, lam1, beta1, xi1_err)``.

    With ``guarded=True`` (solver mode) the computation is branch-free,
    complex-step differentiable and clipped so that it always produces
    finite values; clipping only activates outside the admissible region,
    and ``xi1_err`` is ``None``.
    With ``guarded=False`` (strict mode) any materially invalid quantity
    raises :class:`SplineConvergenceError` with diagnostics, and
    ``xi1_err`` is a first-order bound on the absolute rounding error of
    ``xi1``, dominated by the ``sqrt`` of the discriminant: near a zero
    discriminant the root legitimately carries ``sqrt(eps)``-level error,
    and downstream acceptance tolerances must scale with it.
    """
    beta0 = 0.5 * (beta - alpha)
    beta1 = 0.5 * (beta + alpha)
    # sinc-based ratio sin(beta0)/sin(beta1) (spec section 8).
    xi0 = 0.5 * (beta0 / beta1) * (sinc(beta0) / sinc(beta1))
    q = np.cos(2.0 * beta1)
    D = 2.0 * np.cos(beta0) * np.cos(beta1)
    Aq = 1.0 + 2.0 * q
    Cq = 1.0 - (1.0 - 2.0 * q) * xi0 * xi0
    Dq = D * D - Aq * Cq

    if guarded:
        if np.iscomplexobj(Dq):
            # Clip the real part at zero; where clipped, the derivative is
            # deliberately killed (the point is outside the admissible set).
            Dq_safe = np.where(Dq.real > 0.0, Dq, 0.0)
        else:
            Dq_safe = np.maximum(Dq, 0.0)
        sq = np.sqrt(Dq_safe)
    else:
        floor = -64.0 * EPS * (D.real * D.real + np.abs(Aq * Cq))
        bad = Dq.real < floor
        if np.any(bad):
            i = int(np.argmax(bad))
            raise SplineConvergenceError(
                "Segment discriminant is materially negative",
                index=i,
                quantity="Delta_q",
                value=float(Dq.real[i]),
                bound=f">= {float(floor[i]) if np.ndim(floor) else float(floor):.3e}",
            )
        sq = np.sqrt(np.maximum(Dq, 0.0))

    # Cancellation-resistant admissible root.  For D >= 0 use the
    # rationalized form C_q / (D + sqrt); for D < 0 (only possible when
    # beta1 > pi/2, where A_q <= 1 - 2/sqrt(3) < 0) the direct form
    # (D - sqrt) / A_q adds magnitudes and is the stable one.
    denom_pos = D + sq
    denom_neg = Aq
    if guarded:
        pos = denom_pos.real if np.iscomplexobj(denom_pos) else denom_pos
        denom_pos = np.where(np.abs(pos) < TINY, TINY, denom_pos)
        neg = denom_neg.real if np.iscomplexobj(denom_neg) else denom_neg
        denom_neg = np.where(np.abs(neg) < TINY, TINY, denom_neg)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi1 = np.where(D.real >= 0.0, Cq / denom_pos, (D - sq) / denom_neg)
    lam0 = lhat * (xi1 + xi0)
    lam1 = lhat * (xi1 - xi0)

    xi1_err = None
    if not guarded:
        for name, lam in (("lambda_0", lam0), ("lambda_1", lam1)):
            arr = np.asarray(lam.real, dtype=np.float64)
            bad = ~np.isfinite(arr) | (arr <= 0.0)
            if np.any(bad):
                i = int(np.argmax(bad))
                raise SplineConvergenceError(
                    "Segment Bezier edge length is not finite and strictly positive",
                    index=i,
                    quantity=name,
                    value=float(arr[i]),
                    bound="> 0",
                )
        # First-order absolute error bound for xi1: the discriminant is
        # computed with absolute error ~ eps * S, so its square root
        # carries error ~ eps * S / (sqrt(Dq) + sqrt(eps * S)) -- the
        # sqrt(eps) regime near a vanishing discriminant is unavoidable.
        S = D * D + np.abs(Aq * Cq)
        d_sqrt = EPS * S / (sq + np.sqrt(EPS * S) + TINY)
        err_pos = np.abs(xi1) * d_sqrt / np.maximum(np.abs(denom_pos), TINY)
        err_neg = d_sqrt / np.maximum(np.abs(Aq), TINY)
        xi1_err = np.where(D >= 0.0, err_pos, err_neg) + 4.0 * EPS * np.abs(xi1)
    return xi0, xi1, lam0, lam1, beta1, xi1_err


# ---------------------------------------------------------------------------
# Segment assembly
# ---------------------------------------------------------------------------


def build_straight_segments(
    geometry: Geometry, *, segment_index_offset: int = 0
) -> list[PHSegment]:
    """Degenerate straight spline: one constant-hodograph segment per span."""
    phat = geometry.phat
    segments: list[PHSegment] = []
    m = phat.shape[0] - 1
    for i in range(m):
        p0 = complex(phat[i, 0], phat[i, 1])
        p1 = complex(phat[i + 1, 0], phat[i + 1, 1])
        w = cmath.sqrt(p1 - p0)
        ctrl = PHSegment.control_net(p0, p1, w, w)
        segments.append(
            PHSegment(
                index=segment_index_offset + i,
                w0=w,
                w1=w,
                chi_stable=0.0,
                ctrl=ctrl,
            )
        )
    return segments


def build_curved_segments(
    geometry: Geometry,
    bnd: BoundaryAngles,
    x: np.ndarray,
    *,
    segment_index_offset: int = 0,
) -> tuple[list[PHSegment], np.ndarray]:
    """Strictly validated segment construction at the solved tangents.

    Returns the segments together with the strict per-joint logarithmic
    curvature residuals ``F_i`` used for solver acceptance.
    """
    psi = geometry.psi
    phi = geometry.phi
    tau = geometry.tau
    lhat = geometry.lhat
    phat = geometry.phat
    m = psi.shape[0]

    alpha, beta = segment_alpha_beta(x, phi, bnd.phi0, bnd.phim)
    _, _, lam0, lam1, beta1, xi1_err = edge_quantities(alpha, beta, lhat, guarded=False)
    lam0 = np.asarray(lam0, dtype=np.float64)
    lam1 = np.asarray(lam1, dtype=np.float64)
    beta1 = np.asarray(beta1, dtype=np.float64)
    xi1_err = np.asarray(xi1_err, dtype=np.float64)

    # Strict acceptance residuals (spec section 9.3): the 2/3 factors cancel.
    log_sin = np.log(np.sin(beta1))
    log_l0 = np.log(lam0)
    log_l1 = np.log(lam1)
    log_k_end = log_sin + 0.5 * log_l0 - 1.5 * log_l1
    log_k_start = log_sin + 0.5 * log_l1 - 1.5 * log_l0
    residuals = log_k_end[:-1] - log_k_start[1:]

    # Consistently unwrapped tangent angles theta_0..theta_m.
    theta = np.empty(m + 1)
    theta[0] = bnd.theta0
    theta[1:m] = psi[: m - 1] + tau * x * phi
    theta[m] = bnd.thetam

    segments: list[PHSegment] = []
    for i in range(m):
        w0 = cmath.rect(math.sqrt(3.0 * float(lam0[i])), 0.5 * float(theta[i]))
        w1 = cmath.rect(math.sqrt(3.0 * float(lam1[i])), 0.5 * float(theta[i + 1]))
        p0 = complex(phat[i, 0], phat[i, 1])
        p1 = complex(phat[i + 1, 0], phat[i + 1, 1])
        # PH reconstruction identity (spec section 4.2).
        recon = (w0 * w0 + w0 * w1 + w1 * w1) / 3.0
        target = p1 - p0
        err = abs(recon - target)
        # Base tolerance plus the propagated first-order xi1 rounding
        # bound; the amplification covers d(recon)/d(xi1) including the
        # sqrt(lam0/lam1) sensitivity of the mixed leg.
        l0, l1 = float(lam0[i]), float(lam1[i])
        ratio_amp = 1.0 + 0.5 * (math.sqrt(l0 / l1) + math.sqrt(l1 / l0))
        tol = RECON_TOL * max(float(lhat[i]), 3.0 * l0, 3.0 * l1) + (
            16.0 * float(lhat[i]) * float(xi1_err[i]) * ratio_amp
        )
        if not err <= tol:
            raise SplineConvergenceError(
                "PH segment reconstruction residual exceeds the normalized "
                "construction tolerance",
                index=i,
                quantity="|(w0^2 + w0 w1 + w1^2)/3 - dP|",
                value=err,
                bound=f"<= {tol:.3e}",
            )
        ctrl = PHSegment.control_net(p0, p1, w0, w1)
        # chi = Im(conj(w0) w1) computed by the cancellation-free product
        # form; identical analytically, but it keeps full relative accuracy
        # when the tangent turn across the segment is tiny.
        chi = (
            3.0
            * tau
            * math.sqrt(float(lam0[i]) * float(lam1[i]))
            * math.sin(float(beta1[i]))
        )
        segments.append(
            PHSegment(
                index=segment_index_offset + i,
                w0=w0,
                w1=w1,
                chi_stable=chi,
                ctrl=ctrl,
            )
        )
    return segments, residuals
