"""Render 128 PHBSplineOpen-specific examples (8 feature families x 16 cases).

Run from the repository root:
    python examples/bspline/generate_features.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.bspline._common import (
    normalized_samples,
    normalized_vectors,
    render_curve,
)
from ph_spline import PHBSplineClosed, PHBSplineOpen

OUT = Path(__file__).resolve().parent / "features"
ORDERS = (2, 3, 4, 6, 8)


def open_points(case: int, count: int = 10) -> np.ndarray:
    x = np.linspace(0.0, 6.0, count)
    phase = 0.23 * case
    y = (0.28 + 0.018 * case) * np.sin((1.0 + 0.035 * case) * x + phase)
    y += 0.07 * np.sin((2.2 + 0.02 * case) * x - phase)
    return np.column_stack((x, y))


def closed_points(case: int) -> np.ndarray:
    count = 9 + case % 5
    angle = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    radius = 1.0 + (0.08 + 0.006 * case) * np.cos((3 + case % 4) * angle + 0.2 * case)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def output(family: int, case: int, name: str) -> Path:
    return OUT / f"{family:02d}_{name}" / f"{case + 1:03d}_{name}.png"


def main() -> None:
    count = 0
    for case in range(16):
        order = ORDERS[case % len(ORDERS)]

        curve = PHBSplineOpen(open_points(case), g_order=order)
        render_curve(
            curve,
            output(1, case, "continuity_orders"),
            f"G{order} continuity",
            "continuity order selects the PH degree",
        )
        count += 1

        curve = PHBSplineClosed(closed_points(case), g_order=order)
        stations = np.linspace(0.0, curve.length, 13, endpoint=False)
        station_u = curve.parameters_at_length(stations, assume_sorted=True)
        vectors = normalized_vectors(curve, station_u, curve.tangent)
        render_curve(
            curve,
            output(2, case, "closed_distance_stations"),
            "closed distance stations",
            "equal arc-length stations and seam-continuous tangents",
            vectors=vectors,
        )
        count += 1

        curve = PHBSplineOpen(open_points(case, 14), g_order=order)
        before = normalized_samples(curve)
        index = 4 + case % 6
        handle = curve.point_handle(index)
        target = curve.points[index] + [0.0, 0.12 * (-1.0 if case % 2 else 1.0)]
        report = curve.move_point(handle, target)
        render_curve(
            curve,
            output(3, case, "local_move"),
            "stable-handle local move",
            f"rebuilt {report.rebuilt_span_count} of {curve.num_spans} spans",
            before=before,
        )
        count += 1

        curve = PHBSplineOpen(open_points(case, 13), g_order=order)
        before = normalized_samples(curve)
        index = 3 + case % 7
        midpoint = 0.5 * (curve.points[index - 1] + curve.points[index])
        midpoint[1] += 0.04 * (-1.0 if case % 2 else 1.0)
        inserted = curve.insert_point(index, midpoint)
        render_curve(
            curve,
            output(4, case, "local_insert"),
            "local node insertion",
            f"handle {inserted.handle.id}; rebuilt {inserted.report.rebuilt_span_count} spans",
            before=before,
        )
        count += 1

        curve = PHBSplineOpen(open_points(case, 15), g_order=order)
        before = normalized_samples(curve)
        index = 4 + case % 7
        handle = curve.point_handle(index)
        report = curve.delete_point(handle)
        render_curve(
            curve,
            output(5, case, "local_delete"),
            "local node deletion",
            f"deleted stable handle {handle.id}; rebuilt {report.rebuilt_span_count} spans",
            before=before,
        )
        count += 1

        curve = PHBSplineOpen(open_points(case), g_order=max(order, 4))
        derivative_order = 1 + case % 4
        us = np.linspace(0.08, 0.92, 15)
        vectors = normalized_vectors(
            curve,
            us,
            lambda u, n=derivative_order, c=curve: c.derivative(u, n),
        )
        render_curve(
            curve,
            output(6, case, "parameter_derivative_jets"),
            f"parameter derivative order {derivative_order}",
            "elementary preimage-product derivative kernel",
            vectors=vectors,
        )
        count += 1

        curve = PHBSplineOpen(open_points(case), g_order=max(order, 4))
        derivative_order = 1 + case % 4
        us = np.linspace(0.08, 0.92, 15)
        vectors = normalized_vectors(
            curve,
            us,
            lambda u, n=derivative_order, c=curve: c.derivative(u, n, wrt="arc_length"),
        )
        render_curve(
            curve,
            output(7, case, "arc_derivative_jets"),
            f"arc-length derivative order {derivative_order}",
            (
                "centripetal-force vector: the G8 unit-speed second derivative"
                if case == 9
                else "one shared Taylor-series recurrence"
            ),
            vectors=vectors,
        )
        count += 1

        curve = PHBSplineOpen(open_points(case), g_order=max(order, 4))
        derivative_order = case % 4
        us = np.linspace(0.08, 0.92, 15)
        vectors = normalized_vectors(
            curve,
            us,
            lambda u, n=derivative_order, c=curve: c.curvature_vector(u, n),
        )
        render_curve(
            curve,
            output(8, case, "curvature_vector_jets"),
            f"curvature-vector derivative order {derivative_order}",
            "full vector derivative, not scalar curvature alone",
            vectors=vectors,
        )
        count += 1

        print(f"ok feature case {case + 1:02d}/16 ({count}/128)")
    assert count == 128
    print(f"Rendered all {count} PHBSplineOpen-specific examples to {OUT}")


if __name__ == "__main__":
    main()
