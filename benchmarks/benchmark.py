"""Benchmark convex and nonconvex construction/query performance.

Run:  python benchmarks/benchmark.py

For splines with 100 / 1_000 / 10_000 input points, compares a strictly
convex log spiral with a one-inflection cubic S curve.  It measures:

- full verified construction time (best of ``REPS`` runs), and
- 1000 ``point_at_length`` queries at seeded uniform arc lengths.

Construction includes section-22 preprocessing, every block-local simplicity
predicate, every nonlinear G2 solve, and the full verification battery.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from ph_spline import CubicPHSplineOpen

SIZES = [100, 1_000, 10_000]
QUERIES = 1_000
REPS = 3


def spiral_points(n: int) -> list[list[float]]:
    """Log-spiral samples: strictly convex, moderate uniform turns."""
    ts = np.linspace(0.0, 8.0 * math.pi, n)
    return [
        [
            0.5 * math.exp(0.06 * t) * math.cos(t),
            0.5 * math.exp(0.06 * t) * math.sin(t),
        ]
        for t in ts
    ]


def nonconvex_points(n: int) -> list[list[float]]:
    """Monotone cubic graph with one material curvature-sign change."""
    xs = np.linspace(-2.0, 2.0, n)
    return [[float(x), float(0.18 * x**3)] for x in xs]


def bench_one(pts: list[list[float]]) -> tuple[float, float, CubicPHSplineOpen]:
    n = len(pts)
    best_build = math.inf
    curve = None
    reps = REPS if n <= 1_000 else max(1, REPS - 1)
    for _ in range(reps):
        t0 = time.perf_counter()
        curve = CubicPHSplineOpen(pts)
        best_build = min(best_build, time.perf_counter() - t0)
    assert curve is not None

    L = curve.arc_length(1.0)
    rng = np.random.default_rng(20260804)
    targets = (L * rng.random(QUERIES)).tolist()
    best_query = math.inf
    for _ in range(REPS):
        t0 = time.perf_counter()
        for s in targets:
            curve.point_at_length(s)
        best_query = min(best_query, time.perf_counter() - t0)
    return best_build, best_query, curve


def main() -> None:
    print(
        f"{'kind':>10} {'points':>8} {'segments':>9} {'aux':>5} "
        f"{'build [s]':>10} {'1000 queries [ms]':>18} "
        f"{'per query [us]':>15}"
    )
    for n in SIZES:
        measured = []
        for kind, points in (
            ("convex", spiral_points(n)),
            ("nonconvex", nonconvex_points(n)),
        ):
            build, query, curve = bench_one(points)
            measured.append((build, query))
            print(
                f"{kind:>10} {n:>8} {len(curve._segments):>9} "
                f"{len(curve._inflections):>5} {build:>10.3f} "
                f"{1e3 * query:>18.1f} "
                f"{1e6 * query / QUERIES:>15.2f}"
            )
        build_ratio = measured[1][0] / measured[0][0]
        query_ratio = measured[1][1] / measured[0][1]
        print(
            f"{'ratio N/C':>10} {'':>8} {'':>9} {'':>5} "
            f"{build_ratio:>10.2f} {query_ratio:>18.2f}"
        )


if __name__ == "__main__":
    main()
