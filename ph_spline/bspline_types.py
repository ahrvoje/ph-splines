"""Public value and policy types used by :class:`PHBSpline`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

EditRepair = Literal["strict_local", "expand", "global"]


@dataclass(frozen=True, slots=True)
class ContinuitySpec:
    """Requested or independently verified continuity orders."""

    g_order: int | None
    c_order: int | None
    curvature_order: int | None


@dataclass(frozen=True, slots=True)
class ConstructionPolicy:
    """Deterministic PH B-spline construction controls."""

    parameterization: Literal["centripetal", "chord", "uniform"] = "centripetal"
    shape_objective: Literal["preimage_strain", "guide_fairness"] = "preimage_strain"
    max_iterations: int = 48
    max_line_search_steps: int = 16
    max_hidden_spans_per_input_span: int = 8
    max_refinement_rounds: int = 6
    initial_trust_radius: float = 0.25
    interpolation_weight: float = 1.0
    guide_weight: float = 1.0
    strain_weight: float = 1.0e-3
    speed_variation_weight: float = 1.0e-4
    deterministic: bool = True


@dataclass(frozen=True, slots=True)
class EditingPolicy:
    """Transactional editing and repair controls."""

    default_repair: EditRepair = "strict_local"
    initial_patch_spans: int | None = None
    max_patch_spans: int = 64
    expansion_factor: float = 2.0
    leaf_capacity: int = 128
    preserve_outside_bitwise: bool = True


@dataclass(frozen=True, slots=True)
class InversePolicy:
    """Per-span monotone arc-length inversion controls."""

    lut_nodes_min: int = 8
    lut_nodes_max: int = 128
    lut_power_of_two: bool = True
    seed_kind: Literal["monotone_cubic", "linear"] = "monotone_cubic"
    fast_iterations: int = 2
    max_iterations: int = 67
    use_halley: bool = True
    fallback: Literal["itp", "bisection"] = "itp"
    endpoint_reverse_threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class NumericalPolicy:
    """Binary64 resource limits and scale-aware numerical tolerances."""

    dtype: Literal["float64"] = "float64"
    regularity_ratio_min: float = 1.0e-12
    max_preimage_degree: int = 16
    max_evaluation_order: int = 64
    max_regularization_subdivision_depth: int = 24
    parameter_ulp_slack: int = 4
    position_eps_factor: float = 256.0
    tangent_abs_tol: float = 1.0e-12
    curvature_rel_tol: float = 1.0e-10
    continuity_eps_factor: float = 1024.0
    inverse_eps_factor: float = 64.0
    use_longdouble_verification: Literal["auto", "never", "always"] = "auto"
    reject_unresolved_global_lengths: bool = True


@dataclass(frozen=True, slots=True)
class PointHandle:
    """Stable identity of one interpolation point."""

    id: int


@dataclass(frozen=True, slots=True)
class CurveLocation:
    """Versioned local position that avoids global-parameter resolution loss."""

    span_id: int
    local_u: float
    version: int


@dataclass(frozen=True, slots=True)
class LengthCoordinate:
    """Double-double compatible length value."""

    hi: float
    lo: float = 0.0

    def __float__(self) -> float:
        return self.hi + self.lo


@dataclass(frozen=True, slots=True)
class Frame2D:
    """Position and oriented Frenet data at one curve location."""

    point: NDArray[np.float64]
    tangent: NDArray[np.float64]
    left_normal: NDArray[np.float64]
    signed_curvature: float


@dataclass(frozen=True, slots=True)
class BuildDiagnostics:
    """Machine-readable construction and verification summary."""

    point_count: int
    span_count: int
    hidden_span_count: int
    preimage_degree: int
    iterations: int
    refinement_rounds: int
    max_interpolation_residual: float
    interpolation_bound: float
    max_continuity_residual: float
    continuity_bound: float
    min_regularity_ratio: float
    max_inverse_residual_ratio: float
    max_lut_nodes: int
    longdouble_verification_used: bool


@dataclass(frozen=True, slots=True)
class EditReport:
    """Result of one committed atomic edit."""

    operation: str
    version_before: int
    version_after: int
    affected_point_ids: tuple[int, ...]
    affected_span_ids: tuple[int, ...]
    rebuilt_span_count: int
    patch_span_count: int
    iterations: int
    refinement_rounds: int
    hidden_spans_added: int
    max_interpolation_residual: float
    max_continuity_residual: float
    min_regularity_ratio: float


@dataclass(frozen=True, slots=True)
class InsertResult:
    """Stable handle plus the report for an insertion."""

    handle: PointHandle
    report: EditReport


def readonly_vector(x: float, y: float) -> NDArray[np.float64]:
    """Return a fresh read-only binary64 two-vector."""

    result = np.array([x, y], dtype=np.float64)
    result.setflags(write=False)
    return result


__all__ = [
    "BuildDiagnostics",
    "ConstructionPolicy",
    "ContinuitySpec",
    "CurveLocation",
    "EditRepair",
    "EditReport",
    "EditingPolicy",
    "Frame2D",
    "InsertResult",
    "InversePolicy",
    "LengthCoordinate",
    "NumericalPolicy",
    "PointHandle",
]
