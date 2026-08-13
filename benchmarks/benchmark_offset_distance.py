"""Benchmark offset NURBS distance and inverse-distance queries.

Run:  python benchmarks/benchmark_offset_distance.py

Implements the reporting contract of the offset-distance specification
(section 13.2): for each configuration it reports median, p95, p99, and
worst *non-adaptive* per-query times over 10,000 deterministic seeded
queries, plus the certified-fallback (adaptive) count and total adaptive
time, measured separately for:

- ``length``               (stored total, O(1)),
- ``arc_length``           (cell lookup + elementary evaluation),
- ``parameter_at_length``  (safeguarded certified inverse), and
- ``point_at_length``      (inverse plus homogeneous de Boor point).

The prefix-lookup component of ``arc_length`` is also timed on its own,
and offset construction (geometry plus the complete verified metric
certificate) is reported for source sizes from 10 to 10,000 spans to
show the near-logarithmic query scaling required by section 14.6.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import ph_spline.offset_metric as om
from ph_spline import CubicPHSplineClosed, CubicPHSplineOpen, PHBSplineOpen

QUERIES = 10_000
SEED = 20260813


class AdaptiveCounter:
    """Counts and times certified-fallback entries during a run."""

    def __init__(self):
        self.count = 0
        self.seconds = 0.0
        self._original = None

    def __enter__(self):
        self._original = om.OffsetMetric._arc_length_certified

        counter = self

        def wrapped(metric, j, u):
            counter.count += 1
            t0 = time.perf_counter()
            try:
                return counter._original(metric, j, u)
            finally:
                counter.seconds += time.perf_counter() - t0

        om.OffsetMetric._arc_length_certified = wrapped
        return self

    def __exit__(self, *exc):
        om.OffsetMetric._arc_length_certified = self._original
        return False


def time_queries(fn, args_list):
    """Per-query wall times (seconds) for a list of prepared arguments."""
    times = np.empty(len(args_list))
    for i, a in enumerate(args_list):
        t0 = time.perf_counter()
        fn(a)
        times[i] = time.perf_counter() - t0
    return times


def report(label, times, adaptive=None):
    us = np.sort(times) * 1e6
    line = (
        f"  {label:22s} median {np.median(us):9.2f}  "
        f"p95 {us[int(0.95 * len(us))]:9.2f}  "
        f"p99 {us[int(0.99 * len(us))]:9.2f}  "
        f"worst {us[-1]:10.2f}"
    )
    if adaptive is not None:
        line += (
            f"   adaptive {adaptive.count}/{len(times)}"
            f" ({adaptive.seconds * 1e3:.2f} ms)"
        )
    print(line)


def bench_handle(name, handle):
    rng = np.random.default_rng(SEED)
    L = handle.length
    us = rng.uniform(0.0, 1.0, QUERIES)
    ss = rng.uniform(0.0, 1.0, QUERIES) * L

    print(f"{name}  (degree {handle.degree}, {handle.num_spans} NURBS spans, "
          f"{len(handle._metric._cells)} metric cells, L = {L:.6g})")

    t = time_queries(lambda _: handle.length, us)
    report("length", t)

    with AdaptiveCounter() as ac:
        t = time_queries(handle.arc_length, us)
    report("arc_length", t, ac)

    # prefix/cell lookup share alone
    metric = handle._metric
    from bisect import bisect_right

    bounds = metric._bounds
    t = time_queries(lambda u: bisect_right(bounds, u), us)
    report("(cell lookup only)", t)

    with AdaptiveCounter() as ac:
        t = time_queries(handle.parameter_at_length, ss)
    report("parameter_at_length", t, ac)

    with AdaptiveCounter() as ac:
        t = time_queries(handle.point_at_length, ss)
    report("point_at_length", t, ac)
    print()


def spiral_points(n, total_angle=8.0 * math.pi):
    ts = np.linspace(0.0, total_angle, n)
    return [
        [0.5 * math.exp(0.06 * t) * math.cos(t),
         0.5 * math.exp(0.06 * t) * math.sin(t)]
        for t in ts
    ]


def main() -> None:
    print(f"offset distance benchmark: {QUERIES} deterministic queries "
          f"per operation, times in microseconds\n")

    # -- representative handles ------------------------------------------
    cubic = CubicPHSplineOpen(spiral_points(100))
    bench_handle("cubic offset d=+0.04 (100 spans)", cubic.offset(0.04))

    diamond = CubicPHSplineClosed(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    bench_handle("cubic closed offset d=+1.0 (8 interior cusps)",
                 diamond.offset(1.0))

    bs = PHBSplineOpen(
        [[float(x), 0.6 * math.sin(1.1 * x)] for x in np.linspace(0, 9, 24)]
    )
    bench_handle("PH B-spline offset d=-0.25 (23 source spans)",
                 bs.offset(-0.25))

    # -- span-count scaling ----------------------------------------------
    print("span-count scaling (cubic spiral, d = +0.04):")
    print(f"  {'spans':>7} {'offset build [s]':>17} "
          f"{'arc_length med [us]':>20} {'inverse med [us]':>17}")
    for n in (10, 100, 1_000, 10_000):
        # keep the per-chord turn well below the admissibility bound for
        # small n by shrinking the total spiral angle with the size
        angle = min(8.0 * math.pi, 0.25 * math.pi * n / 10.0)
        curve = CubicPHSplineOpen(spiral_points(max(n + 1, 4), angle))
        t0 = time.perf_counter()
        h = curve.offset(0.04)
        build = time.perf_counter() - t0
        rng = np.random.default_rng(SEED)
        us = rng.uniform(0.0, 1.0, 2_000)
        ss = rng.uniform(0.0, 1.0, 2_000) * h.length
        t_arc = np.median(time_queries(h.arc_length, us)) * 1e6
        t_inv = np.median(time_queries(h.parameter_at_length, ss)) * 1e6
        print(f"  {h.num_spans:>7} {build:>17.3f} {t_arc:>20.2f} "
              f"{t_inv:>17.2f}")


if __name__ == "__main__":
    main()
