"""Cubic PH spline family, open and cyclic G2 concrete topologies."""

from __future__ import annotations

import math
import numbers
from dataclasses import replace

import numpy as np

from ph_spline._constants import (
    EPS,
    EPS_KAPPA,
    EPS_TANGENT,
    F_TOL,
    RHO_MIN,
    ULP_SLACK,
)
from ph_spline.arclength import compensated_prefix_sums
from ph_spline.base import PHSpline
from ph_spline.construction import (
    analyze_geometry,
    analyze_closed_geometry,
    block_geometry,
    boundary_angles,
    build_curved_segments,
    build_closed_curved_segments,
    build_straight_segments,
    initial_internal_fractions,
    initial_closed_fractions,
    chord_weighted_closed_fractions,
    plan_spline,
    validate_points,
)
from ph_spline.exceptions import (
    ArcLengthOutOfRangeError,
    DegeneratePointDataError,
    G2VerificationError,
    InsufficientPointDataError,
    InterpolationDomainError,
    LengthResolutionError,
    NonAdmissibleSegmentError,
    NonRegularSplineError,
    NumericalPrecisionError,
    ParameterOutOfRangeError,
    SplineConvergenceError,
    UndefinedPrincipalNormalError,
)
from ph_spline.nonlinear import solve_closed_tangents, solve_internal_tangents
from ph_spline.nurbs import (
    NURBSHandle,
    build_offset_handle,
    validate_offset_distance,
)
from ph_spline.segment import PHSegment
from ph_spline.typing import PointSequence, Vector2

__all__ = ["CubicPHSpline", "CubicPHSplineClosed", "CubicPHSplineOpen"]


def _validate_scalar(name: str, value: object) -> float:
    """Accept real Python/NumPy scalars, reject Booleans, arrays, sequences."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a Boolean")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar, not {type(value).__name__}")
    return float(value)


class CubicPHSpline(PHSpline):
    """Abstract family base for planar cubic-PH splines.

    :class:`CubicPHSplineOpen` uses one segment per input span and one additional
    segment at every section-22 auxiliary inflection point.  Geometry is G2
    inside every convex sub-spline and G1 at auxiliary and straight/curved
    joints.  Construction either succeeds with every postcondition verified
    or raises a specific
    :class:`~ph_spline.exceptions.CubicPHSplineError` subclass.

    The instance is immutable from the public API.
    """

    __slots__ = (
        "_boundary_clamped",
        "_frozen",
        "_inflections",
        "_is_straight",
        "_joint_kinds",
        "_joint_tangents",
        "_knots",
        "_m",
        "_min_radii",
        "_n_segments",
        "_origin",
        "_points",
        "_prefix",
        "_prefix_user",
        "_scale",
        "_segment_points",
        "_segment_taus",
        "_segments",
        "_tau",
        "_total_length",
    )

    def __init__(self, points: PointSequence) -> None:
        object.__setattr__(self, "_frozen", False)
        points = validate_points(points)
        input_geometry = analyze_geometry(points)
        plan = plan_spline(input_geometry)
        m = points.shape[0] - 1

        segments: list[PHSegment] = []
        segment_taus: list[int] = []
        joint_kinds: list[str] = []
        joint_tangents: list[tuple[float, float] | None] = []
        segment_points: list[tuple[float, float]] = []
        knots_list: list[float] = []
        clamped_start = False
        clamped_end = False

        for block_index, block in enumerate(plan.blocks):
            geometry = block_geometry(plan, block)
            offset = len(segments)
            if block.kind == "straight":
                block_segments = build_straight_segments(
                    geometry, segment_index_offset=offset
                )
            else:
                at_global_start = float(block.knots[0]) == 0.0
                at_global_end = float(block.knots[-1]) == 1.0
                bnd = boundary_angles(
                    geometry,
                    start_tangent=block.start_tangent,
                    end_tangent=block.end_tangent,
                    free_start_support=(
                        input_geometry.phat[:3] if at_global_start else None
                    ),
                    free_end_support=(
                        input_geometry.phat[-3:] if at_global_end else None
                    ),
                )
                block_segments = self._solve_and_accept(
                    geometry, bnd, segment_index_offset=offset
                )
                if at_global_start:
                    clamped_start = bnd.clamped_start
                if at_global_end:
                    clamped_end = bnd.clamped_end

            if block_index > 0:
                previous = plan.blocks[block_index - 1]
                if previous.kind == "curved" and block.kind == "curved":
                    joint_kinds.append("inflection")
                else:
                    joint_kinds.append("transition")
                joint_tangents.append(block.start_tangent)
            joint_kinds.extend("g2" for _ in range(len(block_segments) - 1))
            joint_tangents.extend(None for _ in range(len(block_segments) - 1))
            segments.extend(block_segments)
            segment_taus.extend([block.tau] * len(block_segments))
            block_points = [tuple(map(float, row)) for row in block.points]
            block_knots = [float(v) for v in block.knots]
            if block_index == 0:
                segment_points.extend(block_points)
                knots_list.extend(block_knots)
            else:
                segment_points.extend(block_points[1:])
                knots_list.extend(block_knots[1:])

        n_segments = len(segments)
        if not (
            len(segment_points) == n_segments + 1
            and len(knots_list) == n_segments + 1
            and len(joint_kinds) == max(0, n_segments - 1)
        ):
            raise NumericalPrecisionError(
                "Internal block assembly produced inconsistent segment metadata",
                quantity="segment metadata sizes",
            )

        self._verify_regularity(segments)
        self._verify_admissibility(segments, segment_taus)
        self._verify_continuity(segments, joint_kinds, joint_tangents)

        lengths = [seg.length for seg in segments]
        prefix = compensated_prefix_sums(lengths)
        for i in range(n_segments):
            if not prefix[i + 1] > prefix[i]:
                raise LengthResolutionError(
                    "Prefix arc length failed to increase in binary64",
                    index=i,
                    quantity="C[i+1] - C[i]",
                    value=prefix[i + 1] - prefix[i],
                    bound="> 0",
                )

        pts = np.array(points, dtype=np.float64, copy=True)
        pts.setflags(write=False)
        seg_pts = np.array(segment_points, dtype=np.float64)
        seg_pts.setflags(write=False)
        knots = np.array(knots_list, dtype=np.float64)
        if not (knots[0] == 0.0 and knots[-1] == 1.0 and np.all(np.diff(knots) > 0.0)):
            raise NumericalPrecisionError(
                "Global segment knots are not strictly increasing",
                quantity="diff(knots)",
                bound="> 0",
            )
        knots.setflags(write=False)
        prefix_arr = np.array(prefix, dtype=np.float64)
        prefix_arr.setflags(write=False)
        prefix_user = input_geometry.scale * prefix_arr
        if not np.all(np.isfinite(prefix_user)):
            raise NumericalPrecisionError(
                "Total arc length overflowed in user coordinates",
                quantity="H * C[i]",
            )
        prefix_user.setflags(write=False)

        object.__setattr__(self, "_points", pts)
        object.__setattr__(self, "_segment_points", seg_pts)
        object.__setattr__(self, "_origin", input_geometry.origin)
        object.__setattr__(self, "_scale", input_geometry.scale)
        object.__setattr__(self, "_m", m)
        object.__setattr__(self, "_n_segments", n_segments)
        object.__setattr__(self, "_knots", knots)
        object.__setattr__(self, "_segments", tuple(segments))
        object.__setattr__(self, "_segment_taus", tuple(segment_taus))
        object.__setattr__(self, "_joint_kinds", tuple(joint_kinds))
        object.__setattr__(self, "_joint_tangents", tuple(joint_tangents))
        object.__setattr__(self, "_inflections", plan.inflections)
        object.__setattr__(self, "_prefix", prefix_arr)
        object.__setattr__(self, "_prefix_user", prefix_user)
        object.__setattr__(self, "_total_length", float(prefix_user[-1]))
        object.__setattr__(
            self,
            "_min_radii",
            self._compute_min_radii(segments, input_geometry.scale),
        )
        unique_taus = set(segment_taus)
        tau = next(iter(unique_taus)) if len(unique_taus) == 1 else 0
        object.__setattr__(self, "_tau", tau)
        object.__setattr__(self, "_is_straight", unique_taus == {0})
        object.__setattr__(self, "_boundary_clamped", (clamped_start, clamped_end))
        object.__setattr__(self, "_frozen", True)

    # ------------------------------------------------------------------
    # Immutability
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable; cannot set {name!r}"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot delete {name!r}"
        )

    # ------------------------------------------------------------------
    # Copy and pickle protocols
    # ------------------------------------------------------------------
    #
    # Immutability is a public-API contract, not a bar on ordinary Python
    # object handling: pickling, ``copy`` and ``deepcopy`` (undo stacks,
    # caches, multiprocessing) must work.  Restoration bypasses
    # ``__init__``, so it re-freezes the arrays and re-runs the
    # post-construction verifiers; a corrupted payload raises the same
    # typed exceptions as a failed construction instead of materializing
    # an unverified spline.

    def __getstate__(self) -> dict:
        return {name: getattr(self, name) for name in CubicPHSpline.__slots__}

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "_frozen", False)
        for name in CubicPHSpline.__slots__:
            if name == "_frozen":
                continue
            value = state[name]
            if isinstance(value, np.ndarray):
                value = np.asarray(value)
                value.setflags(write=False)
            object.__setattr__(self, name, value)
        segments = list(self._segments)
        self._verify_regularity(segments)
        self._verify_admissibility(segments, list(self._segment_taus))
        if self.closed:
            self._verify_closed_continuity(
                segments, list(self._joint_kinds), list(self._joint_tangents)
            )
        else:
            self._verify_continuity(
                segments, list(self._joint_kinds), list(self._joint_tangents)
            )
        object.__setattr__(self, "_frozen", True)

    def __repr__(self) -> str:
        kind = (
            "straight"
            if self._is_straight
            else "ccw"
            if self._tau > 0
            else "cw"
            if self._tau < 0
            else "general"
        )
        return (
            f"{type(self).__name__}({self.num_points} points, {self._n_segments} "
            f"segments, {kind})"
        )

    # ------------------------------------------------------------------
    # Nonlinear solve with a deterministic initializer chain (spec 9)
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_and_accept(
        geometry, bnd, *, segment_index_offset: int = 0
    ) -> list[PHSegment]:
        """Solve the G2 system and strictly accept, or raise.

        The primary start is the centered secant (spec 9.4).  If strict
        acceptance fails, one deterministic retry runs from the
        chord-length-weighted initializer (the alternative initializer
        permitted by spec 9.5), which handles extreme adjacent-chord
        ratios.  The acceptance gate is identical for both.
        """
        from ph_spline.construction import chord_weighted_fractions

        starts = (
            initial_internal_fractions(geometry),
            chord_weighted_fractions(geometry),
        )
        if starts[0].size == 0:
            segments, _ = build_curved_segments(
                geometry,
                bnd,
                starts[0],
                segment_index_offset=segment_index_offset,
            )
            return segments
        last_error: SplineConvergenceError | None = None
        for x0 in starts:
            try:
                x = solve_internal_tangents(
                    geometry.lhat, geometry.phi, bnd.phi0, bnd.phim, x0
                )
                if not (np.all(x > 0.0) and np.all(x < 1.0)):
                    raise SplineConvergenceError(
                        "A solved tangent direction left its admissible wedge",
                        quantity="x",
                    )
                segments, residuals = build_curved_segments(
                    geometry,
                    bnd,
                    x,
                    segment_index_offset=segment_index_offset,
                )
                f_max = float(np.max(np.abs(residuals))) if residuals.size else 0.0
                if not f_max <= F_TOL:
                    raise SplineConvergenceError(
                        "G2 curvature residual exceeds the acceptance tolerance",
                        quantity="max|F_i|",
                        value=f_max,
                        bound=f"<= {F_TOL:.1e}",
                    )
                return segments
            except SplineConvergenceError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    @staticmethod
    def _solve_closed_and_accept(geometry) -> list[PHSegment]:
        """Solve and strictly accept the cyclic convex G2 system."""
        starts = (
            initial_closed_fractions(geometry),
            chord_weighted_closed_fractions(geometry),
        )
        last_error: SplineConvergenceError | None = None
        for x0 in starts:
            try:
                x = solve_closed_tangents(geometry.lhat, geometry.phi, x0)
                if not (np.all(x > 0.0) and np.all(x < 1.0)):
                    raise SplineConvergenceError(
                        "A cyclic tangent direction left its admissible wedge",
                        quantity="x",
                    )
                segments, residuals = build_closed_curved_segments(geometry, x)
                residual = float(np.max(np.abs(residuals)))
                if not residual <= F_TOL:
                    raise SplineConvergenceError(
                        "Cyclic G2 curvature residual exceeds acceptance tolerance",
                        quantity="max|F_i|",
                        value=residual,
                        bound=f"<= {F_TOL:.1e}",
                    )
                return segments
            except SplineConvergenceError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # Post-construction verification (spec section 10)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_min_radii(
        segments: list[PHSegment], scale: float
    ) -> tuple[float, float]:
        """Exact smallest left/right curvature radii in user units.

        Per segment ``kappa(t) = 2 chi / sigma(t)^2`` with constant ``chi``,
        so the curvature extremum lies exactly at the speed minimum.  The
        discriminant identity ``A C - (B/2)^2 = chi^2`` gives the interior
        minimum ``sigma_min = chi^2 / A`` without cancellation; the endpoint
        speeds come from the exact preimage values.  A side with no
        curvature of that sign reports ``math.inf``.
        """
        best_left = math.inf
        best_right = math.inf
        for seg in segments:
            chi = seg.chi
            if chi == 0.0:
                continue
            s0 = seg.sigma(0.0)
            s1 = seg.sigma(1.0)
            s_min = s0 if s0 < s1 else s1
            if seg.A > 0.0:
                t_star = -0.5 * seg.B / seg.A
                if 0.0 < t_star < 1.0:
                    interior = chi * chi / seg.A
                    if interior < s_min:
                        s_min = interior
            radius = scale * s_min * s_min / (2.0 * abs(chi))
            if chi > 0.0:
                if radius < best_left:
                    best_left = radius
            elif radius < best_right:
                best_right = radius
        return (best_left, best_right)

    @staticmethod
    def _verify_regularity(segments: list[PHSegment]) -> None:
        for seg in segments:
            s_min, s_end_max = seg.sigma_extremes()
            if not s_min > 0.0:
                raise NonRegularSplineError(
                    "Segment speed reaches zero (cusp)",
                    index=seg.index,
                    quantity="sigma_min",
                    value=s_min,
                    bound="> 0",
                )
            if not s_min / s_end_max > RHO_MIN:
                raise NonRegularSplineError(
                    "Segment is nearly cuspidal",
                    index=seg.index,
                    quantity="sigma_min / max(sigma(0), sigma(1))",
                    value=s_min / s_end_max,
                    bound=f"> {RHO_MIN:.1e}",
                )

    @staticmethod
    def _verify_admissibility(segments: list[PHSegment], taus: list[int]) -> None:
        for seg, tau in zip(segments, taus):
            if tau == 0:
                if seg.chi != 0.0:
                    raise NonAdmissibleSegmentError(
                        "Straight-block segment has nonzero PH curvature sign",
                        index=seg.index,
                        quantity="chi",
                        value=seg.chi,
                        bound="== 0",
                    )
                continue
            if not tau * seg.chi > 0.0:
                raise NonAdmissibleSegmentError(
                    "Segment curvature sign disagrees with its convex block",
                    index=seg.index,
                    quantity="tau * chi",
                    value=tau * seg.chi,
                    bound="> 0",
                )
            c = seg.ctrl
            e0 = (c[1, 0] - c[0, 0], c[1, 1] - c[0, 1])
            e1 = (c[2, 0] - c[1, 0], c[2, 1] - c[1, 1])
            e2 = (c[3, 0] - c[2, 0], c[3, 1] - c[2, 1])
            cross01 = tau * (e0[0] * e1[1] - e0[1] * e1[0])
            cross12 = tau * (e1[0] * e2[1] - e1[1] * e2[0])
            for name, value in (
                ("tau * (dB0 x dB1)", cross01),
                ("tau * (dB1 x dB2)", cross12),
            ):
                if not value > 0.0:
                    raise NonAdmissibleSegmentError(
                        "Bezier control polygon violates oriented convexity",
                        index=seg.index,
                        quantity=name,
                        value=value,
                        bound="> 0",
                    )

    @staticmethod
    def _verify_continuity(
        segments: list[PHSegment],
        joint_kinds: list[str],
        joint_tangents: list[tuple[float, float] | None],
    ) -> None:
        for joint, (left, right, prescribed) in enumerate(
            zip(segments[:-1], segments[1:], joint_tangents)
        ):
            kind = joint_kinds[joint]
            tl = left.tangent_local(1.0)
            tr = right.tangent_local(0.0)
            t_gap = math.hypot(tl[0] - tr[0], tl[1] - tr[1])
            if not t_gap <= EPS_TANGENT:
                raise G2VerificationError(
                    "Tangent mismatch at an internal join",
                    index=right.index,
                    quantity="|T-(1) - T+(0)|",
                    value=t_gap,
                    bound=f"<= {EPS_TANGENT:.1e}",
                )
            kl = left.curvature_local(1.0)
            kr = right.curvature_local(0.0)
            if kind == "g2":
                denom = max(abs(kl), abs(kr), EPS)
                k_gap = abs(kl - kr) / denom
                if not k_gap <= EPS_KAPPA:
                    raise G2VerificationError(
                        "Curvature mismatch at an internal G2 join",
                        index=right.index,
                        quantity="|k-(1) - k+(0)| / max(|k-|, |k+|, eps)",
                        value=k_gap,
                        bound=f"<= {EPS_KAPPA:.1e}",
                    )
            elif kind == "inflection":
                if not kl * kr < 0.0:
                    raise G2VerificationError(
                        "Curvature does not change sign at an auxiliary point",
                        index=right.index,
                        quantity="k-(1) * k+(0)",
                        value=kl * kr,
                        bound="< 0",
                    )
            elif kind == "transition":
                if not ((left.chi == 0.0) ^ (right.chi == 0.0)):
                    raise G2VerificationError(
                        "Straight/curved transition does not have exactly one "
                        "zero-curvature side",
                        index=right.index,
                        quantity="zero-curvature sides",
                        value=(left.chi == 0.0, right.chi == 0.0),
                        bound="exactly one",
                    )
            if prescribed is not None:
                for label, tangent in (("left", tl), ("right", tr)):
                    gap = math.hypot(
                        tangent[0] - prescribed[0], tangent[1] - prescribed[1]
                    )
                    if not gap <= EPS_TANGENT:
                        raise G2VerificationError(
                            f"{label.capitalize()} tangent misses the prescribed "
                            "G1 direction",
                            index=right.index,
                            quantity="|T - d'|",
                            value=gap,
                            bound=f"<= {EPS_TANGENT:.1e}",
                        )

    @staticmethod
    def _verify_closed_continuity(
        segments: list[PHSegment],
        joint_kinds: list[str] | None = None,
        joint_tangents: list[tuple[float, float] | None] | None = None,
    ) -> None:
        """Independently verify the declared continuity at every cyclic join."""
        if joint_kinds is None:
            joint_kinds = ["g2"] * len(segments)
        if joint_tangents is None:
            joint_tangents = [None] * len(segments)
        if not (
            len(joint_kinds) == len(segments)
            and len(joint_tangents) == len(segments)
        ):
            raise NumericalPrecisionError(
                "Closed join metadata has inconsistent dimensions",
                quantity="closed join metadata sizes",
            )
        for joint, right in enumerate(segments):
            left = segments[joint - 1]
            tangent_left = left.tangent_local(1.0)
            tangent_right = right.tangent_local(0.0)
            tangent_gap = math.hypot(
                tangent_left[0] - tangent_right[0],
                tangent_left[1] - tangent_right[1],
            )
            if not tangent_gap <= EPS_TANGENT:
                raise G2VerificationError(
                    "Tangent mismatch at a closed cubic join",
                    index=joint,
                    quantity="|T-(1) - T+(0)|",
                    value=tangent_gap,
                    bound=f"<= {EPS_TANGENT:.1e}",
                )
            curvature_left = left.curvature_local(1.0)
            curvature_right = right.curvature_local(0.0)
            kind = joint_kinds[joint]
            if kind == "g2":
                denominator = max(abs(curvature_left), abs(curvature_right), EPS)
                curvature_gap = abs(curvature_left - curvature_right) / denominator
                if not curvature_gap <= EPS_KAPPA:
                    raise G2VerificationError(
                        "Curvature mismatch at a closed cubic G2 join",
                        index=joint,
                        quantity="|k-(1) - k+(0)| / max(|k-|, |k+|, eps)",
                        value=curvature_gap,
                        bound=f"<= {EPS_KAPPA:.1e}",
                    )
            elif kind == "inflection":
                if not curvature_left * curvature_right < 0.0:
                    raise G2VerificationError(
                        "Curvature does not change sign at a closed auxiliary join",
                        index=joint,
                        quantity="k-(1) * k+(0)",
                        value=curvature_left * curvature_right,
                        bound="< 0",
                    )
            elif kind == "transition":
                if not ((left.chi == 0.0) ^ (right.chi == 0.0)):
                    raise G2VerificationError(
                        "Closed transition does not have one zero-curvature side",
                        index=joint,
                        quantity="zero-curvature sides",
                        value=(left.chi == 0.0, right.chi == 0.0),
                        bound="exactly one",
                    )
            else:
                raise NumericalPrecisionError(
                    "Closed join has an unknown continuity classification",
                    index=joint,
                    quantity="joint kind",
                    value=kind,
                )
            prescribed = joint_tangents[joint]
            if prescribed is not None:
                for label, tangent in (
                    ("left", tangent_left),
                    ("right", tangent_right),
                ):
                    gap = math.hypot(
                        tangent[0] - prescribed[0], tangent[1] - prescribed[1]
                    )
                    if not gap <= EPS_TANGENT:
                        raise G2VerificationError(
                            f"{label.capitalize()} tangent misses the closed "
                            "prescribed G1 direction",
                            index=joint,
                            quantity="|T - d'|",
                            value=gap,
                            bound=f"<= {EPS_TANGENT:.1e}",
                        )

    # ------------------------------------------------------------------
    # Parameter dispatch
    # ------------------------------------------------------------------

    def _validate_u(self, u: object) -> float:
        v = _validate_scalar("u", u)
        if math.isnan(v):
            raise ParameterOutOfRangeError(
                "Parameter u must not be NaN", quantity="u", value=v
            )
        if v < 0.0:
            if v >= -ULP_SLACK * EPS:
                return 0.0
            raise ParameterOutOfRangeError(
                "Parameter u is below the domain [0, 1]",
                quantity="u",
                value=v,
                bound=">= 0",
            )
        if v > 1.0:
            if v <= 1.0 + ULP_SLACK * EPS:
                return 1.0
            raise ParameterOutOfRangeError(
                "Parameter u is above the domain [0, 1]",
                quantity="u",
                value=v,
                bound="<= 1",
            )
        return v

    def _validate_s(self, s: object) -> float:
        v = _validate_scalar("s", s)
        total = self._total_length
        if math.isnan(v):
            raise ArcLengthOutOfRangeError(
                "Arc length s must not be NaN", quantity="s", value=v
            )
        slack = ULP_SLACK * math.ulp(total)
        if v < 0.0:
            if v >= -slack:
                return 0.0
            raise ArcLengthOutOfRangeError(
                "Arc length s is below the domain [0, L]",
                quantity="s",
                value=v,
                bound=">= 0",
            )
        if v > total:
            if v <= total + slack:
                return total
            raise ArcLengthOutOfRangeError(
                "Arc length s is above the domain [0, L]",
                quantity="s",
                value=v,
                bound=f"<= {total!r}",
            )
        return v

    def _locate_u(self, u: float) -> tuple[int, float, int | None]:
        """Segment index, local parameter, and exact-knot index if any."""
        n = self._n_segments
        knots = self._knots
        j = int(np.searchsorted(knots, u))
        if j <= n and knots[j] == u:
            if j == n:
                return (n - 1, 1.0, n)
            return (j, 0.0, j)
        i = min(max(j - 1, 0), n - 1)
        width = float(knots[i + 1] - knots[i])
        t = (u - float(knots[i])) / width
        t = min(max(t, 0.0), 1.0)
        return (i, t, None)

    def _locate_s(self, s: float) -> tuple[int, float, int | None]:
        """Segment index, local normalized arc target, exact-prefix index."""
        n = self._n_segments
        pu = self._prefix_user
        j = int(np.searchsorted(pu, s))
        if j <= n and pu[j] == s:
            return (min(j, n - 1), 0.0, j)
        shat = s / self._scale
        prefix = self._prefix
        i = int(np.searchsorted(prefix, shat, side="right")) - 1
        if i < 0:
            i = 0
        elif i > n - 1:
            i = n - 1
        s_local = shat - float(prefix[i])
        s_local = max(s_local, 0.0)
        seg_len = self._segments[i].length
        s_local = min(s_local, seg_len)
        return (i, s_local, None)

    def _denormalize_point(self, x: float, y: float) -> Vector2:
        px = self._origin[0] + self._scale * x
        py = self._origin[1] + self._scale * y
        if not (math.isfinite(px) and math.isfinite(py)):
            raise NumericalPrecisionError(
                "Evaluated point is not finite", quantity="point"
            )
        return np.array([px, py], dtype=np.float64)

    def _frame_at(self, u: object) -> tuple[float, float]:
        """Unit tangent at ``u`` through the right-sided knot convention."""
        v = self._validate_u(u)
        i, t, _ = self._locate_u(v)
        return self._segments[i].tangent_local(t)

    # ------------------------------------------------------------------
    # Public geometric evaluation (spec sections 2.2 and 11)
    # ------------------------------------------------------------------

    @property
    def degree(self) -> int:
        return 3

    @property
    def num_points(self) -> int:
        return int(self._points.shape[0])

    @property
    def length(self) -> float:
        return self._total_length

    @property
    def min_curvature_radii(self) -> tuple[float, float]:
        """Smallest left/right curvature radii ``(rho_left, rho_right)``.

        ``rho_left`` is the smallest radius of curvature among left-turning
        (positive-curvature) points and bounds the cusp-free positive offset
        range; ``rho_right`` is its right-turning counterpart.  Every
        ``offset(d)`` with ``-rho_right < d < rho_left`` is free of cusps;
        equality reaches the cusp condition ``1 - d * kappa = 0`` exactly.
        A side with no curvature of that sign reports ``math.inf``.  The
        value is exact (closed form, a few ulps), computed once during
        construction, and returned in O(1).
        """
        return self._min_radii

    @property
    def aux_inflection_points(self) -> list[dict[str, float]]:
        """Auxiliary inflection points inserted during construction.

        Each item contains the global spline parameter ``u``, prefix arc
        length ``s``, and user-coordinate position ``x``, ``y``.  A fresh
        list and dictionaries are returned so callers cannot mutate the
        spline's internal construction data.
        """
        result: list[dict[str, float]] = []
        for info in self._inflections:
            u = float((info.span_index + info.rho) / self._m)
            result.append(
                {
                    "u": u,
                    "s": self.arc_length(u),
                    "x": float(info.point[0]),
                    "y": float(info.point[1]),
                }
            )
        return result

    def point(self, u: object) -> Vector2:
        """Point on the spline at global parameter ``u`` in ``[0, 1]``."""
        v = self._validate_u(u)
        i, t, knot = self._locate_u(v)
        if knot is not None:
            return np.array(self._segment_points[knot], dtype=np.float64)
        x, y = self._segments[i].point_local(t)
        return self._denormalize_point(x, y)

    def tangent(self, u: object) -> Vector2:
        """Unit tangent vector at ``u``."""
        tx, ty = self._frame_at(u)
        return np.array([tx, ty], dtype=np.float64)

    def normal(self, u: object, side: str = "left") -> Vector2:
        """Unit left or right normal at ``u``."""
        if side == "left":
            sign = 1.0
        elif side == "right":
            sign = -1.0
        else:
            raise ValueError(f'normal() side must be "left" or "right", got {side!r}')
        tx, ty = self._frame_at(u)
        return np.array([-sign * ty, sign * tx], dtype=np.float64)

    def principal_normal(self, u: object) -> Vector2:
        """Unit normal pointing toward the local center of curvature."""
        v = self._validate_u(u)
        i, t, _ = self._locate_u(v)
        seg = self._segments[i]
        kappa = seg.curvature_local(t)
        if kappa == 0.0:
            raise UndefinedPrincipalNormalError(
                "Principal normal is undefined on a straight segment",
                quantity="signed curvature",
                value=0.0,
            )
        tx, ty = seg.tangent_local(t)
        sign = 1.0 if kappa > 0.0 else -1.0
        return np.array([-sign * ty, sign * tx], dtype=np.float64)

    def signed_curvature(self, u: object) -> float:
        """Signed curvature at ``u`` (positive for left-turning splines)."""
        v = self._validate_u(u)
        i, t, _ = self._locate_u(v)
        return float(self._segments[i].curvature_local(t) / self._scale)

    def curvature_vector(self, u: object) -> Vector2:
        """Curvature vector ``K = kappa * N_left`` at ``u``."""
        v = self._validate_u(u)
        i, t, _ = self._locate_u(v)
        seg = self._segments[i]
        kappa = seg.curvature_local(t) / self._scale
        tx, ty = seg.tangent_local(t)
        return np.array([-kappa * ty, kappa * tx], dtype=np.float64)

    # ------------------------------------------------------------------
    # Exact parallel offset (spec section 11.7)
    # ------------------------------------------------------------------

    def offset(self, distance: object) -> NURBSHandle:
        """Exact rational NURBS parallel offset at a signed distance.

        Positive ``distance`` offsets along the left unit normal, negative
        along the right.  The result is an immutable, fully verified
        rational quintic :class:`~ph_spline.nurbs.NURBSHandle` over the
        unchanged global parameter; it is built from finite homogeneous
        Bernstein products of the stored PH data, never from sampling or
        fitting.
        """
        d = validate_offset_distance(distance)
        segments = self._segments
        origin = self._origin
        scale = self._scale
        span_controls: list[np.ndarray] = []
        span_speeds: list[np.ndarray] = []
        span_hodographs: list[np.ndarray] = []
        for seg in segments:
            span_controls.append(np.asarray(seg.ctrl, dtype=np.float64))
            w0 = seg.w0
            w1 = seg.w1
            # Degree-2 Bernstein speed coefficients of sigma = |w(t)|^2 from
            # the stored preimage: exact endpoint speeds and the mixed
            # midpoint product Re(conj(w0) w1).
            span_speeds.append(
                np.array(
                    [
                        seg.sigma(0.0),
                        (w0.conjugate() * w1).real,
                        seg.sigma(1.0),
                    ],
                    dtype=np.float64,
                )
            )
            h0 = w0 * w0
            h1 = w0 * w1
            h2 = w1 * w1
            span_hodographs.append(
                np.array(
                    [
                        [h0.real, h0.imag],
                        [h1.real, h1.imag],
                        [h2.real, h2.imag],
                    ],
                    dtype=np.float64,
                )
            )

        knots = self._knots

        def oracle(u: float) -> tuple[tuple[float, float], tuple[float, float]]:
            i, t, _ = self._locate_u(u)
            seg = segments[i]
            x, y = seg.point_local(t)
            tx, ty = seg.tangent_local(t)
            return (
                (origin[0] + scale * x, origin[1] + scale * y),
                (-ty, tx),
            )

        return build_offset_handle(
            span_controls=span_controls,
            span_speeds=span_speeds,
            span_hodographs=span_hodographs,
            hodograph_tolerance=1.0e-8,
            breakpoints=np.asarray(knots, dtype=np.float64),
            distance=d,
            distance_normalized=d / scale,
            origin=origin,
            scale=scale,
            closed=self.closed,
            join_tolerance=EPS_TANGENT,
            oracle=oracle,
            # Distance metric certificate: each cubic span has the linear
            # preimage w(t) = (1-t) w0 + t w1 and unit local width.
            metric_preimages=[[seg.w0, seg.w1] for seg in segments],
            metric_widths=[1.0] * len(segments),
        )

    # ------------------------------------------------------------------
    # Arc-length operations (spec section 14)
    # ------------------------------------------------------------------

    def arc_length(self, u: object) -> float:
        """Arc length from the spline start to parameter ``u``."""
        v = self._validate_u(u)
        i, t, knot = self._locate_u(v)
        if knot is not None:
            return float(self._prefix_user[knot])
        shat = float(self._prefix[i]) + self._segments[i].arc_length_local(t)
        return float(self._scale * shat)

    def parameter_at_length(self, s: object) -> float:
        """Global parameter ``u`` with ``arc_length(u) = s``."""
        v = self._validate_s(s)
        if v == self._total_length:
            return 1.0
        i, s_local, prefix_hit = self._locate_s(v)
        if prefix_hit is not None:
            return float(self._knots[prefix_hit])
        t = self._segments[i].invert_arc_length_local(s_local)
        u0 = float(self._knots[i])
        u1 = float(self._knots[i + 1])
        return float(min(u0 + t * (u1 - u0), 1.0))

    def point_at_length(self, s: object) -> Vector2:
        """Point at arc length ``s``; single segment location and inversion."""
        v = self._validate_s(s)
        i, s_local, prefix_hit = self._locate_s(v)
        if prefix_hit is not None:
            return np.array(self._segment_points[prefix_hit], dtype=np.float64)
        seg = self._segments[i]
        t = seg.invert_arc_length_local(s_local)
        x, y = seg.point_local(t)
        return self._denormalize_point(x, y)


class CubicPHSplineOpen(CubicPHSpline):
    """Immutable open cubic-PH interpolant for admissible planar points."""

    __slots__ = ()

    @property
    def closed(self) -> bool:
        return False


class CubicPHSplineClosed(CubicPHSpline):
    """Immutable closed cubic-PH interpolant for admissible cyclic points.

    Strictly convex cycles use the square cyclic G2 solve.  General cycles
    reuse the open family's deterministic auxiliary-inflection machinery;
    they are G2 within each same-sign run and G1 only at genuine sign changes
    and straight/curved transitions.
    """

    __slots__ = ()

    def __init__(self, points: PointSequence) -> None:
        object.__setattr__(self, "_frozen", False)
        points = validate_points(points)
        try:
            geometry = analyze_closed_geometry(points)
        except InterpolationDomainError as error:
            if error.quantity not in (
                "absolute normalized turn cross product",
                "cyclic turn sign",
            ):
                raise
            self._initialize_general_closed(points)
            return
        segments = self._solve_closed_and_accept(geometry)
        n = len(segments)
        taus = [geometry.tau] * n

        self._verify_regularity(segments)
        self._verify_admissibility(segments, taus)
        self._verify_closed_continuity(segments)

        prefix = compensated_prefix_sums([segment.length for segment in segments])
        for i in range(n):
            if not prefix[i + 1] > prefix[i]:
                raise LengthResolutionError(
                    "Closed prefix arc length failed to increase in binary64",
                    index=i,
                    quantity="C[i+1] - C[i]",
                    value=prefix[i + 1] - prefix[i],
                    bound="> 0",
                )

        user_points = np.array(points, dtype=np.float64, copy=True)
        user_points.setflags(write=False)
        segment_points = np.vstack((points, points[:1])).astype(
            np.float64, copy=False
        )
        segment_points.setflags(write=False)
        knots = np.arange(n + 1, dtype=np.float64) / n
        knots.setflags(write=False)
        prefix_normalized = np.asarray(prefix, dtype=np.float64)
        prefix_normalized.setflags(write=False)
        prefix_user = geometry.scale * prefix_normalized
        if not np.all(np.isfinite(prefix_user)):
            raise NumericalPrecisionError(
                "Closed total arc length overflowed in user coordinates",
                quantity="H * C[i]",
            )
        prefix_user.setflags(write=False)

        object.__setattr__(self, "_points", user_points)
        object.__setattr__(self, "_segment_points", segment_points)
        object.__setattr__(self, "_origin", geometry.origin)
        object.__setattr__(self, "_scale", geometry.scale)
        object.__setattr__(self, "_m", n)
        object.__setattr__(self, "_n_segments", n)
        object.__setattr__(self, "_knots", knots)
        object.__setattr__(self, "_segments", tuple(segments))
        object.__setattr__(self, "_segment_taus", tuple(taus))
        object.__setattr__(self, "_joint_kinds", tuple("g2" for _ in range(n)))
        object.__setattr__(self, "_joint_tangents", tuple(None for _ in range(n)))
        object.__setattr__(self, "_inflections", ())
        object.__setattr__(self, "_prefix", prefix_normalized)
        object.__setattr__(self, "_prefix_user", prefix_user)
        object.__setattr__(self, "_total_length", float(prefix_user[-1]))
        object.__setattr__(
            self, "_min_radii", self._compute_min_radii(segments, geometry.scale)
        )
        object.__setattr__(self, "_tau", geometry.tau)
        object.__setattr__(self, "_is_straight", False)
        object.__setattr__(self, "_boundary_clamped", (False, False))
        object.__setattr__(self, "_frozen", True)

    def _initialize_general_closed(self, points: np.ndarray) -> None:
        """Compile one central period of a five-period open construction.

        Repetition is a construction device, not approximation: every central
        convex block is bounded by the same deterministic auxiliary joints as
        it is in the infinite cyclic extension.  Cropping the central period
        removes both open free boundaries and preserves the existing
        subsegment-inflection implementation unchanged.  Two guard periods
        on each side also permit an independent adjacent-period check.
        """
        n_points = points.shape[0]
        if n_points < 3:
            raise InsufficientPointDataError(
                "At least three distinct points are required for a closed spline",
                quantity="len(points)",
                value=n_points,
                bound=">= 3",
            )
        if np.array_equal(points[0], points[-1]):
            raise DegeneratePointDataError(
                "Closed input lists the seam point twice; provide it once",
                index=(0, n_points - 1),
                quantity="seam point",
            )
        cycle = points.tolist()
        extended = cycle * 5 + [cycle[0]]
        periodic = CubicPHSplineOpen(extended)
        lower = 2.0 / 5.0
        upper = 3.0 / 5.0
        next_upper = 4.0 / 5.0
        start = int(np.argmin(np.abs(periodic._knots - lower)))
        end = int(np.argmin(np.abs(periodic._knots - upper)))
        knot_tolerance = 32.0 * EPS
        if not (
            abs(float(periodic._knots[start]) - lower) <= knot_tolerance
            and abs(float(periodic._knots[end]) - upper) <= knot_tolerance
            and end > start
        ):
            raise NumericalPrecisionError(
                "Could not isolate the central period of cyclic construction",
                quantity="central-period knot locations",
            )

        source_segments = periodic._segments[start:end]
        segments = [
            PHSegment(
                index=index,
                w0=source.w0,
                w1=source.w1,
                chi_stable=source.chi,
                ctrl=source.ctrl,
            )
            for index, source in enumerate(source_segments)
        ]
        segment_count = len(segments)
        taus = list(periodic._segment_taus[start:end])
        joint_kinds = [periodic._joint_kinds[end - 1]] + list(
            periodic._joint_kinds[start : end - 1]
        )
        joint_tangents = [periodic._joint_tangents[end - 1]] + list(
            periodic._joint_tangents[start : end - 1]
        )
        if joint_kinds[0] != "g2":
            raise InterpolationDomainError(
                "The selected closed seam is not inside a G2-compatible run",
                index=0,
                quantity="seam continuity",
                value=joint_kinds[0],
                bound="g2",
            )

        self._verify_regularity(segments)
        self._verify_admissibility(segments, taus)
        self._verify_closed_continuity(segments, joint_kinds, joint_tangents)

        # The following period must reproduce the same geometric spans.  This
        # is an independent check that three-period cropping was sufficient.
        next_end = int(np.argmin(np.abs(periodic._knots - next_upper)))
        next_period = periodic._segments[end:next_end]
        if len(next_period) != segment_count:
            raise NumericalPrecisionError(
                "Repeated cyclic construction produced inconsistent span counts",
                quantity="period span counts",
                value=(segment_count, len(next_period)),
                bound="equal",
            )
        for index, (left, right) in enumerate(zip(segments, next_period)):
            if not np.allclose(left.ctrl, right.ctrl, rtol=2e-12, atol=2e-12):
                raise NumericalPrecisionError(
                    "Repeated cyclic construction was not geometrically periodic",
                    index=index,
                    quantity="Bezier control-net repeat residual",
                )

        prefix = compensated_prefix_sums([segment.length for segment in segments])
        for i in range(segment_count):
            if not prefix[i + 1] > prefix[i]:
                raise LengthResolutionError(
                    "Closed prefix arc length failed to increase in binary64",
                    index=i,
                    quantity="C[i+1] - C[i]",
                    value=prefix[i + 1] - prefix[i],
                    bound="> 0",
                )
        segment_points = np.array(
            periodic._segment_points[start : end + 1], dtype=np.float64, copy=True
        )
        segment_points.setflags(write=False)
        knots = np.array(
            (periodic._knots[start : end + 1] - lower) / (upper - lower),
            dtype=np.float64,
        )
        knots[0], knots[-1] = 0.0, 1.0
        for point_index in range(1, n_points):
            target = point_index / n_points
            knot_index = int(np.argmin(np.abs(knots - target)))
            if abs(float(knots[knot_index]) - target) > 64.0 * EPS:
                raise NumericalPrecisionError(
                    "Cropped construction lost an authoritative user knot",
                    index=point_index,
                    quantity="nearest user-knot residual",
                    value=abs(float(knots[knot_index]) - target),
                    bound=f"<= {64.0 * EPS:.3e}",
                )
            knots[knot_index] = target
        if not np.all(np.diff(knots) > 0.0):
            raise NumericalPrecisionError(
                "Cropped closed knots are not strictly increasing",
                quantity="diff(knots)",
                bound="> 0",
            )
        knots.setflags(write=False)

        inflections = tuple(
            replace(info, span_index=info.span_index - 2 * n_points)
            for info in periodic._inflections
            if 2 * n_points <= info.span_index < 3 * n_points
        )
        user_points = np.array(points, dtype=np.float64, copy=True)
        user_points.setflags(write=False)
        prefix_normalized = np.asarray(prefix, dtype=np.float64)
        prefix_normalized.setflags(write=False)
        prefix_user = periodic._scale * prefix_normalized
        if not np.all(np.isfinite(prefix_user)):
            raise NumericalPrecisionError(
                "Closed total arc length overflowed in user coordinates",
                quantity="H * C[i]",
            )
        prefix_user.setflags(write=False)

        object.__setattr__(self, "_points", user_points)
        object.__setattr__(self, "_segment_points", segment_points)
        object.__setattr__(self, "_origin", periodic._origin)
        object.__setattr__(self, "_scale", periodic._scale)
        object.__setattr__(self, "_m", n_points)
        object.__setattr__(self, "_n_segments", segment_count)
        object.__setattr__(self, "_knots", knots)
        object.__setattr__(self, "_segments", tuple(segments))
        object.__setattr__(self, "_segment_taus", tuple(taus))
        object.__setattr__(self, "_joint_kinds", tuple(joint_kinds))
        object.__setattr__(self, "_joint_tangents", tuple(joint_tangents))
        object.__setattr__(self, "_inflections", inflections)
        object.__setattr__(self, "_prefix", prefix_normalized)
        object.__setattr__(self, "_prefix_user", prefix_user)
        object.__setattr__(self, "_total_length", float(prefix_user[-1]))
        object.__setattr__(
            self, "_min_radii", self._compute_min_radii(segments, periodic._scale)
        )
        unique_taus = set(taus)
        object.__setattr__(
            self, "_tau", next(iter(unique_taus)) if len(unique_taus) == 1 else 0
        )
        object.__setattr__(self, "_is_straight", unique_taus == {0})
        object.__setattr__(self, "_boundary_clamped", (False, False))
        object.__setattr__(self, "_frozen", True)

    @property
    def closed(self) -> bool:
        return True
