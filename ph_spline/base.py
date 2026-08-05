"""Common abstract interface for planar Pythagorean-hodograph splines."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Vector2 = NDArray[np.float64]


class PHSpline(ABC):
    """Shared read-only geometry and distance interface for PH splines.

    Concrete spline families may add construction, editing, derivative, and
    diagnostic APIs.  The methods below are the polymorphic contract common
    to both :class:`CubicPHSpline` and :class:`PHBSpline`.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def degree(self) -> int:
        """Polynomial curve degree of each ordinary span."""

    @property
    @abstractmethod
    def num_points(self) -> int:
        """Number of authoritative interpolation points."""

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Whether traversal is periodic and the two parameter ends coincide."""

    @property
    @abstractmethod
    def length(self) -> float:
        """Total curve length in user coordinates."""

    @abstractmethod
    def point(self, u: object) -> Vector2:
        """Return the point at compatibility parameter ``u``."""

    @abstractmethod
    def tangent(self, u: object) -> Vector2:
        """Return the unit traversal tangent."""

    @abstractmethod
    def normal(self, u: object, side: str = "left") -> Vector2:
        """Return the selected oriented unit normal."""

    @abstractmethod
    def principal_normal(self, u: object) -> Vector2:
        """Return the unit normal toward the center of curvature."""

    @abstractmethod
    def signed_curvature(self, u: object) -> float:
        """Return signed scalar curvature."""

    @abstractmethod
    def curvature_vector(self, u: object) -> Vector2:
        """Return signed curvature times the left normal."""

    @abstractmethod
    def arc_length(self, u: object) -> float:
        """Return prefix arc length through parameter ``u``."""

    @abstractmethod
    def parameter_at_length(self, s: object) -> float:
        """Invert prefix arc length to the compatibility parameter."""

    @abstractmethod
    def point_at_length(self, s: object) -> Vector2:
        """Return the point reached after travelling distance ``s``."""


__all__ = ["PHSpline"]
