"""Paper-style arc-length traversal benchmark.

This benchmark mirrors the query counts in Table I of:

    L. Gajny, R. Bearee, E. Nyiri, and O. Gibaru,
    "Path planning with PH G2 splines in R2," 2012.
    https://doi.org/10.1109/IConSCS.2012.6502455

The historical columns below are transcribed from the paper.  The paper does
not publish the curve coefficients, feed-rate samples, source code, hardware,
or software environment needed for an exact rerun.  Consequently, the local
measurement uses the public ``point_at_length`` API on a representative
two-segment curved spline and the same numbers of sequential distance queries.
It is a workload reproduction, not a controlled cross-hardware speedup claim.

The paper's PH implementation solved the cubic arc-length equation with two or
three Newton-Raphson iterations per sample at tolerance 1e-13.  Its ordinary
polynomial comparator also evaluated arc length by numerical quadrature.  This
package instead starts from the elementary cubic inverse, applies bounded
residual polishing, and evaluates the point directly.

Run:  python benchmarks/benchmark_2012_interpolator.py
"""

from __future__ import annotations

import gc
import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from ph_spline import CubicPHSplineOpen

# sampling period [s], number of points, PH CPU time [s], polynomial CPU time [s]
PAPER_ROWS = (
    (1.0e-1, 30, 0.1100, 1.5620),
    (8.0e-2, 34, 0.1410, 1.7960),
    (6.0e-2, 40, 0.1560, 2.1250),
    (4.0e-2, 53, 0.2040, 2.8120),
    (2.0e-2, 91, 0.3430, 4.8600),
    (1.0e-2, 169, 0.6250, 9.1100),
    (8.0e-3, 209, 0.7810, 11.2650),
    (6.0e-3, 274, 1.0000, 14.6880),
    (4.0e-3, 403, 1.5320, 21.7040),
    (2.0e-3, 794, 3.0310, 42.6720),
    (1.0e-3, 1_576, 6.1090, 84.6720),
    (8.0e-4, 1_966, 7.2030, 105.7340),
    (6.0e-4, 2_619, 9.8590, 141.0320),
    (4.0e-4, 3_920, 13.9530, 210.8900),
    (2.0e-4, 7_828, 29.2030, 422.6090),
    (1.0e-4, 15_643, 59.3120, 853.7970),
)

TRIALS = 3
MIN_TRIAL_SECONDS = 0.10


def representative_curve() -> CubicPHSplineOpen:
    """Small curved spline; deliberately harder than a straight PH segment."""
    return CubicPHSplineOpen([[0.0, 0.0], [0.8, 0.45], [1.9, 1.25]])


def feedrate_targets(length: float, count: int) -> list[float]:
    """Monotone start/stop traversal targets from a smooth feed-rate law."""
    x = np.linspace(0.0, 1.0, count)
    # Integral of a nonnegative raised-cosine velocity, normalized to [0, 1].
    fractions = x - np.sin(2.0 * math.pi * x) / (2.0 * math.pi)
    return [float(length * f) for f in fractions]


def time_traversal(curve: CubicPHSplineOpen, targets: list[float]) -> float:
    """Best-of-three time for one complete traversal, with calibrated batching."""
    point_at_length = curve.point_at_length
    for s in targets[: min(len(targets), 256)]:
        point_at_length(s)

    start = time.perf_counter()
    for s in targets:
        point_at_length(s)
    probe = time.perf_counter() - start
    batch_repeats = max(1, math.ceil(MIN_TRIAL_SECONDS / max(probe, 1.0e-12)))

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        best = math.inf
        for _ in range(TRIALS):
            start = time.perf_counter()
            for _ in range(batch_repeats):
                for s in targets:
                    point_at_length(s)
            elapsed = time.perf_counter() - start
            best = min(best, elapsed / batch_repeats)
    finally:
        if was_enabled:
            gc.enable()
    return best


def main() -> None:
    curve = representative_curve()
    length = curve.arc_length(1.0)

    print(f"Python {platform.python_version()} / NumPy {np.__version__}")
    print(platform.platform())
    print(f"Representative curve: {len(curve._segments)} segments, length {length:.9g}")
    print(
        f"{'period [s]':>10} {'queries':>8} {'paper PH [s]':>13} "
        f"{'paper poly [s]':>14} {'ours [s]':>10} "
        f"{'ours [us/q]':>12} {'PH/ours':>10}"
    )
    for period, count, paper_ph, paper_polynomial in PAPER_ROWS:
        targets = feedrate_targets(length, count)
        ours = time_traversal(curve, targets)
        print(
            f"{period:>10.1e} {count:>8,d} {paper_ph:>13.4f} "
            f"{paper_polynomial:>14.4f} {ours:>10.6f} "
            f"{1.0e6 * ours / count:>12.3f} {paper_ph / ours:>10.1f}x"
        )


if __name__ == "__main__":
    main()
