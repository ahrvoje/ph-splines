"""Public and internal type aliases (specification section 20)."""

from __future__ import annotations

from typing import TypeAlias, Union

import numpy as np
import numpy.typing as npt

__all__ = ["PointLike", "PointSequence", "RealScalar", "Vector2"]

#: Accepted real scalar types.  Booleans are explicitly rejected at runtime.
RealScalar: TypeAlias = Union[int, float, "np.integer", "np.floating"]

#: One input point: a list or tuple of exactly two real coordinates.
PointLike: TypeAlias = list[RealScalar] | tuple[RealScalar, RealScalar]

#: The constructor input: a list of points.
PointSequence: TypeAlias = list[PointLike]

#: A NumPy float64 array of shape ``(2,)``.
Vector2: TypeAlias = npt.NDArray[np.float64]
