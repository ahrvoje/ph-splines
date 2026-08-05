"""Three reproducible PHBSpline benchmark sections.

Run:  python benchmarks/benchmark_bspline.py

The script reports:

1. the same convex/nonconvex size sweep as the cubic benchmark;
2. verified G3/G4/G6/G8 construction and random distance access; and
3. strict-local move/insert/delete timings at 100, 1_000, and 10_000 nodes
   for G2, G4, and G8.

Construction times are best-of-two below 10,000 nodes and single-run at
10,000 nodes; query times are best-of-three.
Editing times are medians of seven warmed public calls. Every query includes
the production forward residual gate and every edit includes candidate
verification and atomic commit.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
import time
from collections.abc import Callable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from ph_spline import PHBSpline

SIZES = (100, 1_000, 10_000)
HIGH_ORDERS = (3, 4, 6, 8)
EDIT_ORDERS = (2, 4, 8)
QUERIES = 1_000
EDIT_SAMPLES = 7


def convex_points(n: int) -> np.ndarray:
    t = np.linspace(0.0, 8.0 * math.pi, n)
    radius = 0.5 * np.exp(0.06 * t)
    return np.column_stack((radius * np.cos(t), radius * np.sin(t)))


def nonconvex_points(n: int) -> np.ndarray:
    x = np.linspace(-2.0, 2.0, n)
    return np.column_stack((x, 0.18 * x**3))


def editing_points(n: int) -> np.ndarray:
    x = np.linspace(0.0, 20.0 * math.pi, n)
    return np.column_stack((x, 0.2 * np.sin(x) + 0.01 * np.sin(3.0 * x)))


def best_time(action: Callable[[], object], repetitions: int = 3) -> float:
    best = math.inf
    gc_enabled = gc.isenabled()
    try:
        gc.disable()
        for _ in range(repetitions):
            start = time.perf_counter()
            action()
            best = min(best, time.perf_counter() - start)
    finally:
        if gc_enabled:
            gc.enable()
    return best


def construct(points: np.ndarray, order: int = 2) -> tuple[PHBSpline, float]:
    holder: list[PHBSpline] = []

    def action() -> None:
        holder[:] = [PHBSpline(points, g_order=order)]

    repetitions = 1 if len(points) >= 10_000 else 2
    elapsed = best_time(action, repetitions)
    return holder[0], elapsed


def query_time(curve: PHBSpline) -> float:
    rng = np.random.default_rng(20260805)
    targets = curve.length * rng.random(QUERIES)

    def action() -> None:
        for target in targets:
            curve.point_at_length(float(target))

    return best_time(action)


def section_1(sizes: tuple[int, ...]) -> None:
    print("\nSECTION 1 — cubic-style size and shape sweep")
    print("| kind | nodes | spans | construction [s] | point_at_length [us] |")
    print("|:--|--:|--:|--:|--:|")
    for n in sizes:
        for kind, points in (
            ("convex", convex_points(n)),
            ("nonconvex", nonconvex_points(n)),
        ):
            curve, build = construct(points)
            query = query_time(curve)
            print(
                f"| {kind} | {n:,} | {curve.num_spans:,} | {build:.4f} | {1e6 * query / QUERIES:.2f} |"
            )


def section_2(size: int) -> None:
    print("\nSECTION 2 — requested higher continuity")
    print(
        "| continuity | nodes | PH degree | construction [s] | point_at_length [us] |"
    )
    print("|:--|--:|--:|--:|--:|")
    points = editing_points(size)
    for order in HIGH_ORDERS:
        curve, build = construct(points, order)
        query = query_time(curve)
        print(
            f"| G{order} | {size:,} | {curve.degree} | {build:.4f} | {1e6 * query / QUERIES:.2f} |"
        )


def section_3(sizes: tuple[int, ...]) -> None:
    print("\nSECTION 3 — public strict-local editing, median end-to-end latency")
    print(
        "| nodes | continuity | move median [ms] | insert median [ms] | "
        "delete median [ms] | rebuilt move/insert/delete |"
    )
    print("|--:|:--|--:|--:|--:|:--|")
    for n in sizes:
        points = editing_points(n)
        for order in EDIT_ORDERS:
            curve = PHBSpline(points, g_order=order)
            center = curve.num_points // 2
            original = curve.points[center]
            moved = original + [0.0, 1.0e-3]

            # Warm both directions, then alternate them so every recorded
            # sample is one complete public mutation and atomic commit.
            curve.move_point(center, moved)
            curve.move_point(center, original)
            move_times = []
            move_reports = []
            for sample in range(EDIT_SAMPLES):
                target = moved if sample % 2 == 0 else original
                start = time.perf_counter_ns()
                move_reports.append(curve.move_point(center, target))
                move_times.append((time.perf_counter_ns() - start) * 1.0e-6)
            if EDIT_SAMPLES % 2:
                curve.move_point(center, original)

            inserted_point = 0.5 * (
                curve.points[center - 1] + curve.points[center]
            )
            inserted_point[1] += 5.0e-4

            warm_insert = curve.insert_point(center, inserted_point)
            curve.delete_point(warm_insert.handle)
            insert_times = []
            insert_reports = []
            for _ in range(EDIT_SAMPLES):
                start = time.perf_counter_ns()
                inserted = curve.insert_point(center, inserted_point)
                insert_times.append((time.perf_counter_ns() - start) * 1.0e-6)
                insert_reports.append(inserted.report)
                curve.delete_point(inserted.handle)

            warm_delete = curve.insert_point(center, inserted_point)
            curve.delete_point(warm_delete.handle)
            delete_times = []
            delete_reports = []
            for _ in range(EDIT_SAMPLES):
                inserted = curve.insert_point(center, inserted_point)
                start = time.perf_counter_ns()
                delete_reports.append(curve.delete_point(inserted.handle))
                delete_times.append((time.perf_counter_ns() - start) * 1.0e-6)

            rebuilt = (
                f"{move_reports[-1].rebuilt_span_count}/"
                f"{insert_reports[-1].rebuilt_span_count}/"
                f"{delete_reports[-1].rebuilt_span_count}"
            )
            print(
                f"| {n:,} | G{order} | {statistics.median(move_times):.2f} | "
                f"{statistics.median(insert_times):.2f} | "
                f"{statistics.median(delete_times):.2f} | {rebuilt} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use 100/1,000 nodes and 300 nodes for the continuity sweep",
    )
    args = parser.parse_args()
    sizes = (100, 1_000) if args.quick else SIZES
    section_1(sizes)
    section_2(300 if args.quick else 1_000)
    section_3(sizes)


if __name__ == "__main__":
    main()
