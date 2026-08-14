"""Closed-spline area benchmarks (ClosedSpline_Area_Specification.md, s. 16).

Run from the repository root:
    python benchmarks/benchmark_area.py

Reports, separately for both closed families and their exact closed
offsets:

- cold first-query, warm repeated-query, and post-local-edit timings at
  span counts of roughly 10, 100, and 1,000;
- several supported PH B-spline continuity orders at a fixed size;
- fast-path acceptance percentage and exact-fallback count;
- the maximum certified phase precision used by turning numbers;
- reused versus recomputed span contributions after local edits; and
- peak additional memory attributable to offset area provenance.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ph_spline.area as area_module
from ph_spline import CubicPHSplineClosed, PHBSplineClosed


def circle(n, radius=1.0):
    return [
        [radius * math.cos(2 * math.pi * k / n),
         radius * math.sin(2 * math.pi * k / n)]
        for k in range(n)
    ]


def wavy(n, wave=0.2, lobes=5):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n
        r = 1.0 + wave * math.cos(lobes * t)
        out.append([r * math.cos(t), r * math.sin(t)])
    return out


def best_of(runs, callable_):
    best = math.inf
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = callable_()
        best = min(best, time.perf_counter() - start)
    return best, result


def timed(callable_):
    start = time.perf_counter()
    result = callable_()
    return time.perf_counter() - start, result


def fmt(seconds: float) -> str:
    if seconds < 1.0e-3:
        return f"{seconds * 1e6:8.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:8.3f} ms"
    return f"{seconds:8.3f} s "


def provenance_bytes(handle) -> int:
    provenance = handle._area_provenance
    total = sum(array.nbytes for array in provenance.position_spans)
    total += sys.getsizeof(provenance.position_spans)
    return total


def stats_snapshot():
    return dict(area_module.statistics)


def stats_delta(before, after):
    return {key: after[key] - before[key] for key in before}


TOTALS = {key: 0 for key in area_module.statistics}


def consume_stats() -> None:
    """Fold the module counters into the run totals, then reset them."""
    for key, value in area_module.statistics.items():
        if key == "turning_max_precision":
            TOTALS[key] = max(TOTALS[key], value)
        elif key != "last_condition":
            TOTALS[key] += value
    area_module.reset_statistics()


def bench_family(label, factory, sizes):
    print(f"\n== {label}: cold / warm / post-edit by size ==")
    print(f"{'spans':>6} | {'construct':>11} | {'cold area':>11} | "
          f"{'warm area':>11} | {'post-edit':>11} | reused/recomputed")
    for points in sizes:
        consume_stats()
        build_time, curve = timed(lambda: factory(points))
        cold_time, _ = timed(lambda: curve.signed_area)
        warm_time, _ = best_of(5, lambda: curve.signed_area)
        editable = isinstance(curve, PHBSplineClosed)
        if editable:
            before = stats_snapshot()
            target = [1.02 * curve.points[1][0], 1.02 * curve.points[1][1]]
            curve.move_point(1, target)
            edit_time, _ = timed(lambda: curve.signed_area)
            delta = stats_delta(before, stats_snapshot())
            reuse = (f"{delta['span_reused']}/"
                     f"{delta['span_contributions']}")
        else:
            edit_time, reuse = float("nan"), "immutable"
        spans = (curve.num_spans if editable
                 else curve._n_segments)
        edit_text = "        n/a" if not editable else fmt(edit_time)
        print(f"{spans:>6} | {fmt(build_time)} | {fmt(cold_time)} | "
              f"{fmt(warm_time)} | {edit_text} | {reuse}")


def bench_offsets(label, factory, sizes, distance=0.1):
    print(f"\n== {label} closed offsets: cold / warm by size ==")
    print(f"{'spans':>6} | {'offset build':>12} | {'cold area':>11} | "
          f"{'warm area':>11} | {'turn bits':>9} | {'provenance':>11}")
    for points in sizes:
        curve = factory(points)
        build_time, handle = timed(lambda: curve.offset(distance))
        consume_stats()
        cold_time, _ = timed(lambda: handle.signed_area)
        warm_time, _ = best_of(5, lambda: handle.signed_area)
        bits = area_module.statistics["turning_max_precision"]
        spans = (curve.num_spans if isinstance(curve, PHBSplineClosed)
                 else curve._n_segments)
        print(f"{spans:>6} | {fmt(build_time)} | {fmt(cold_time)} | "
              f"{fmt(warm_time)} | {bits:>9} | "
              f"{provenance_bytes(handle):>9} B")


def bench_orders():
    print("\n== PHBSplineClosed area by continuity order (100 nodes) ==")
    print(f"{'order':>6} | {'degree':>6} | {'cold area':>11} | "
          f"{'warm area':>11} | {'offset cold':>11}")
    pts = wavy(100)
    for label, kwargs in (
        ("G2", {}),
        ("G3", {"g_order": 3}),
        ("G4", {"g_order": 4}),
    ):
        curve = PHBSplineClosed(pts, **kwargs)
        cold_time, _ = timed(lambda: curve.signed_area)
        warm_time, _ = best_of(5, lambda: curve.signed_area)
        handle = curve.offset(0.05)
        offset_cold, _ = timed(lambda: handle.signed_area)
        print(f"{label:>6} | {curve.degree:>6} | {fmt(cold_time)} | "
              f"{fmt(warm_time)} | {fmt(offset_cold)}")


def main() -> None:
    sizes_cubic = (10, 100, 1000)
    sizes_bspline = (8, 50, 500)  # closed B-splines compile 2 spans/node

    area_module.reset_statistics()
    bench_family("CubicPHSplineClosed",
                 lambda n: CubicPHSplineClosed(circle(n)), sizes_cubic)
    bench_family("PHBSplineClosed (G2)",
                 lambda n: PHBSplineClosed(circle(n)), sizes_bspline)
    bench_orders()
    bench_offsets("CubicPHSplineClosed",
                  lambda n: CubicPHSplineClosed(circle(n)), sizes_cubic)
    bench_offsets("PHBSplineClosed (G2)",
                  lambda n: PHBSplineClosed(circle(n)), sizes_bspline)

    consume_stats()
    queries = TOTALS["fast_accepted"] + TOTALS["exact_fallback"]
    accepted = (100.0 * TOTALS["fast_accepted"] / queries) if queries else 0.0
    print("\n== Arithmetic-path diagnostics (whole run) ==")
    print(f"source-area queries observed : {queries}")
    print(f"fast path accepted           : {TOTALS['fast_accepted']} "
          f"({accepted:.1f} %)")
    print(f"exact rational fallbacks     : {TOTALS['exact_fallback']}")
    print(f"span contributions computed  : {TOTALS['span_contributions']}")
    print(f"span contributions reused    : {TOTALS['span_reused']}")
    print(f"max certified phase precision: "
          f"{TOTALS['turning_max_precision']} bits")


if __name__ == "__main__":
    main()
