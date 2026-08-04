"""Public :class:`CubicPHSpline` class and global parameter dispatch."""

from __future__ import annotations

import math
import numbers

import numpy as np

from cubic_ph_spline._constants import (
    EPS,
    EPS_KAPPA,
    EPS_TANGENT,
    F_TOL,
    RHO_MIN,
    ULP_SLACK,
)
from cubic_ph_spline.arclength import compensated_prefix_sums
from cubic_ph_spline.construction import (
    analyze_geometry,
    block_geometry,
    boundary_angles,
    build_curved_segments,
    build_straight_segments,
    initial_internal_fractions,
    plan_spline,
    validate_points,
)
from cubic_ph_spline.exceptions import (
    ArcLengthOutOfRangeError,
    G2VerificationError,
    LengthResolutionError,
    NonAdmissibleSegmentError,
    NonRegularSplineError,
    NumericalPrecisionError,
    ParameterOutOfRangeError,
    SplineConvergenceError,
    UndefinedPrincipalNormalError,
)
from cubic_ph_spline.nonlinear import solve_internal_tangents
from cubic_ph_spline.segment import PHSegment
from cubic_ph_spline.typing import PointSequence, Vector2

__all__ = ["CubicPHSpline"]


def _validate_scalar(name: str, value: object) -> float:
    """Accept real Python/NumPy scalars, reject Booleans, arrays, sequences."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a Boolean")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar, not {type(value).__name__}")
    return float(value)


class CubicPHSpline:
    """Open planar cubic-PH spline for convex and general point data.

    ``CubicPHSpline(p)`` uses one segment per input span and one additional
    segment at every section-22 auxiliary inflection point.  Geometry is G2
    inside every convex sub-spline and G1 at auxiliary and straight/curved
    joints.  Construction either succeeds with every postcondition verified
    or raises a specific
    :class:`~cubic_ph_spline.exceptions.CubicPHSplineError` subclass.

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

    def __init__(self, p: PointSequence) -> None:
        object.__setattr__(self, "_frozen", False)
        points = validate_points(p)
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
            f"{type(self).__name__}({self._m + 1} points, {self._n_segments} "
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
        from cubic_ph_spline.construction import chord_weighted_fractions

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

    # ------------------------------------------------------------------
    # Post-construction verification (spec section 10)
    # ------------------------------------------------------------------

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
