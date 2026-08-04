"""Robust planar predicates: orientation, circumcenter, proper intersection.

All predicates operate on already-normalized coordinates and use
floating-point error filters: a computed determinant whose magnitude does
not exceed its rounding-error bound is classified as zero rather than
trusted for its sign.
"""

from __future__ import annotations

import math

import numpy as np

from cubic_ph_spline._constants import EPS

__all__ = [
    "circumcenter",
    "orientation",
    "polyline_self_intersection",
]

#: Multiplier for the orientation-filter rounding bound.
_ORIENT_FILTER = 32.0 * EPS


def orientation(
    px: float, py: float, qx: float, qy: float, rx: float, ry: float
) -> int:
    """Filtered orientation sign of the triangle ``p, q, r``.

    Returns ``+1`` for counterclockwise, ``-1`` for clockwise and ``0``
    when the determinant magnitude is below its rounding-error bound.
    """
    t1 = (qx - px) * (ry - py)
    t2 = (qy - py) * (rx - px)
    det = t1 - t2
    err = _ORIENT_FILTER * (abs(t1) + abs(t2))
    if det > err:
        return 1
    if det < -err:
        return -1
    return 0


def circumcenter(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> tuple[float, float] | None:
    """Circumcenter of three points, or ``None`` if unreliable.

    Computed in coordinates translated to ``a`` for conditioning.  Returns
    ``None`` when the doubled-area determinant does not exceed its
    rounding-error bound (spec section 7.1).
    """
    bx = b[0] - a[0]
    by = b[1] - a[1]
    cx = c[0] - a[0]
    cy = c[1] - a[1]
    t1 = bx * cy
    t2 = by * cx
    d = 2.0 * (t1 - t2)
    if abs(d) <= 2.0 * _ORIENT_FILTER * (abs(t1) + abs(t2)):
        return None
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (cy * b2 - by * c2) / d
    uy = (bx * c2 - cx * b2) / d
    if not (math.isfinite(ux) and math.isfinite(uy)):
        return None
    return (a[0] + ux, a[1] + uy)


def _orientation_array(
    px: float,
    py: float,
    qx: float,
    qy: float,
    rx: np.ndarray,
    ry: np.ndarray,
) -> np.ndarray:
    """Vectorized filtered orientation of ``p, q`` against many points ``r``."""
    t1 = (qx - px) * (ry - py)
    t2 = (qy - py) * (rx - px)
    det = t1 - t2
    err = _ORIENT_FILTER * (np.abs(t1) + np.abs(t2))
    out = np.zeros(det.shape, dtype=np.int64)
    out[det > err] = 1
    out[det < -err] = -1
    return out


def polyline_self_intersection(points: np.ndarray) -> tuple[int, int] | None:
    """First properly intersecting pair of nonadjacent chords, or ``None``.

    ``points`` has shape ``(m + 1, 2)``.  Chord ``i`` joins point ``i`` to
    point ``i + 1``; chords ``i`` and ``j`` with ``j > i + 1`` are
    nonadjacent.  A proper intersection is a strict transversal crossing of
    both chord interiors: each chord's endpoints lie strictly on opposite
    sides of the other chord's supporting line.
    """
    m = points.shape[0] - 1
    if m < 3:
        return None
    x = points[:, 0]
    y = points[:, 1]
    x0, y0 = x[:-1], y[:-1]
    x1, y1 = x[1:], y[1:]
    lo_x = np.minimum(x0, x1)
    hi_x = np.maximum(x0, x1)
    lo_y = np.minimum(y0, y1)
    hi_y = np.maximum(y0, y1)
    for i in range(m - 2):
        j = np.arange(i + 2, m)
        # Bounding-box prefilter.
        cand = ~(
            (lo_x[j] > hi_x[i])
            | (hi_x[j] < lo_x[i])
            | (lo_y[j] > hi_y[i])
            | (hi_y[j] < lo_y[i])
        )
        if not np.any(cand):
            continue
        j = j[cand]
        o1 = _orientation_array(x0[i], y0[i], x1[i], y1[i], x0[j], y0[j])
        o2 = _orientation_array(x0[i], y0[i], x1[i], y1[i], x1[j], y1[j])
        straddle_ij = o1 * o2 < 0
        if not np.any(straddle_ij):
            continue
        j = j[straddle_ij]
        o3 = np.array(
            [
                orientation(x0[k], y0[k], x1[k], y1[k], x0[i], y0[i])
                * orientation(x0[k], y0[k], x1[k], y1[k], x1[i], y1[i])
                for k in j
            ],
            dtype=np.int64,
        )
        crossing = o3 < 0
        if np.any(crossing):
            return (i, int(j[np.argmax(crossing)]))
    return None
