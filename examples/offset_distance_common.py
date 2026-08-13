"""Shared case model, verification and rendering for the offset distance
query galleries.

Each of the four ``*_offset_distance`` example folders defines 32 cases and
calls :func:`run` with its curve factory - 128 studies in total.  Every
rendered image is built exclusively from verified public distance queries
on the exact offset NURBS handle:

- equally spaced *distance stations* placed with ``point_at_length`` along
  the true offset locus (not the source curve and not the parameter grid);
- an optional highlighted span between two exact travel distances;
- cusp markers at certified interior offset cusps, where the station flow
  visibly compresses;
- the total offset ``length`` and station spacing in the caption.

Before anything is drawn, each case re-verifies the offset geometry against
``r(u) + d * N_L(u)`` and the distance queries against round-trip residual
gates; a case that fails verification aborts the generation run.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offset_common import (  # noqa: E402  (shared gallery helpers)
    catmull_rom,
    meander,
    open_curve,
    periodic,
    polar,
    seeded_walk,
    verify_offset,
)

PAGE = "#f9f9f7"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK = "#172033"
MUTED = "#8a94a6"
NODE = "#b8bfcc"
LEFT_SIDE = "#246fc2"   # positive distances (left normal)
RIGHT_SIDE = "#c45b31"  # negative distances (right normal)
STATION = "#eb6834"
HIGHLIGHT = "#0e9b6c"
CUSP = "#c22440"

__all__ = [
    "DistanceCase",
    "catmull_rom",
    "meander",
    "open_curve",
    "periodic",
    "polar",
    "run",
    "seeded_walk",
]


@dataclass(frozen=True)
class DistanceCase:
    """One rendered offset distance-query study."""

    name: str
    points: list[list[float]]
    distance: float
    note: str
    category: str
    extra: tuple[float, ...] = ()
    stations: int = 25
    highlight: tuple[float, float] | None = None
    fill: bool = False
    kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_distance_api(handle) -> None:
    """Round-trip and monotonicity gates on the public distance queries."""
    L = handle.length
    if not (math.isfinite(L) and L > 0.0):
        raise RuntimeError(f"invalid offset length {L!r}")
    if handle.arc_length(0.0) != 0.0 or handle.arc_length(1.0) != L:
        raise RuntimeError("endpoint identities violated")
    if handle.parameter_at_length(0.0) != 0.0:
        raise RuntimeError("parameter_at_length(0) != 0")
    if handle.parameter_at_length(L) != 1.0:
        raise RuntimeError("parameter_at_length(L) != 1")
    rng = np.random.default_rng(20260813)
    prev = -1.0
    for u in np.linspace(0.0, 1.0, 161):
        a = handle.arc_length(float(u))
        if a < prev:
            raise RuntimeError("arc_length is not nondecreasing")
        prev = a
    for s in rng.uniform(0.0, 1.0, 40) * L:
        s = float(s)
        u = handle.parameter_at_length(s)
        if abs(handle.arc_length(u) - s) > 1e-10 * L + 8.0 * math.ulp(L):
            raise RuntimeError(f"inverse residual gate failed at s={s!r}")
        p1 = handle.point_at_length(s)
        p2 = handle.point(u)
        if not np.array_equal(p1, p2):
            raise RuntimeError("point_at_length disagrees with point(u)")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _sample(handle, count: int = 900) -> np.ndarray:
    return np.asarray(
        [handle.point(float(u)) for u in np.linspace(0.0, 1.0, count)]
    )


def _interior_cusps(handle) -> list[float]:
    """Global parameters of certified interior offset cusps."""
    return [c.parameter for c in handle.cusps if 0.0 < c.parameter < 1.0]


def render_case(
    out_dir: Path,
    index: int,
    case: DistanceCase,
    curve,
    family_label: str,
    *,
    check_only: bool = False,
) -> None:
    primary = curve.offset(case.distance)
    verify_offset(curve, primary, case.distance)
    verify_distance_api(primary)
    extras = []
    for d in case.extra:
        h = curve.offset(d)
        verify_offset(curve, h, d)
        extras.append((d, h))
    if check_only:
        return

    L = primary.length
    closed = primary.closed
    n_st = case.stations
    station_s = np.linspace(0.0, L, n_st, endpoint=not closed)
    station_pts = np.asarray(
        [primary.point_at_length(float(s)) for s in station_s]
    )
    spacing = station_s[1] - station_s[0] if n_st > 1 else L

    source = _sample(curve)
    offset_line = _sample(primary)

    fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=105, facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    if case.fill:
        ax.fill(source[:, 0], source[:, 1], color=INK, alpha=0.045, zorder=1)

    for d, h in extras:
        line = _sample(h, 500)
        ax.plot(line[:, 0], line[:, 1], linewidth=0.9, zorder=2, alpha=0.5,
                color=LEFT_SIDE if d >= 0 else RIGHT_SIDE)

    side_color = LEFT_SIDE if case.distance >= 0 else RIGHT_SIDE
    ax.plot(offset_line[:, 0], offset_line[:, 1], color=side_color,
            linewidth=1.9, zorder=3,
            label=f"exact offset locus (d = {case.distance:+g})")
    ax.plot(source[:, 0], source[:, 1], color=INK, linewidth=1.5, zorder=4,
            alpha=0.85, label=family_label)

    points = np.asarray(case.points, dtype=np.float64)
    step = max(1, math.ceil(len(points) / 36))
    shown = points[::step]
    ax.scatter(shown[:, 0], shown[:, 1], s=9, facecolor=SURFACE,
               edgecolor=NODE, linewidth=0.7, zorder=4)

    # Highlighted exact-travel span between two distances.
    if case.highlight is not None:
        f0, f1 = case.highlight
        s0, s1 = f0 * L, f1 * L
        u0 = primary.parameter_at_length(s0)
        u1 = primary.parameter_at_length(s1)
        seg = np.asarray(
            [primary.point(float(u)) for u in np.linspace(u0, u1, 240)]
        )
        ax.plot(seg[:, 0], seg[:, 1], color=HIGHLIGHT, linewidth=3.6,
                alpha=0.85, zorder=5, solid_capstyle="round",
                label=f"exact travel {s1 - s0:.3g} units")
        for s_end in (s0, s1):
            p = primary.point_at_length(s_end)
            ax.scatter([p[0]], [p[1]], marker="D", s=26, facecolor=SURFACE,
                       edgecolor=HIGHLIGHT, linewidth=1.3, zorder=7)

    # Distance stations measured along the offset locus.
    ax.scatter(station_pts[1:, 0], station_pts[1:, 1], s=13,
               facecolor=SURFACE, edgecolor=STATION, linewidth=0.95,
               zorder=6, label=f"stations every {spacing:.3g} units")
    ax.scatter([station_pts[0, 0]], [station_pts[0, 1]], marker="^", s=42,
               facecolor=STATION, edgecolor=INK, linewidth=0.6, zorder=7,
               label="s = 0")

    cusps = _interior_cusps(primary)
    if cusps:
        cp = np.asarray([primary.point(u) for u in cusps])
        ax.scatter(cp[:, 0], cp[:, 1], marker="x", s=34, color=CUSP,
                   linewidth=1.2, zorder=8,
                   label=f"{len(cusps)} certified offset cusps")

    fig.suptitle(f"{index:03d} · {case.name.replace('_', ' ')}", x=0.06,
                 ha="left", color=INK, fontsize=10.5)
    ax.set_title(
        f"{case.category} · {curve.num_points} nodes · degree-{primary.degree}"
        f" rational offset · L = {L:.4g}\n{case.note}",
        loc="left", color=INK, fontsize=7.15,
    )
    ax.legend(loc="best", frameon=False, fontsize=6.6)
    fig.savefig(out_dir / f"{index:03d}_{case.name}.png",
                bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)


def run(cases: list[DistanceCase], build_curve, out_dir: Path,
        family_label: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="verify all cases without rendering")
    args = parser.parse_args()
    if len(cases) != 32:
        raise SystemExit(f"expected exactly 32 cases, found {len(cases)}")
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise SystemExit("case names must be unique")
    failures: list[str] = []
    for index, case in enumerate(cases, start=1):
        try:
            curve = build_curve(case)
            render_case(out_dir, index, case, curve, family_label,
                        check_only=args.check_only)
        except Exception as error:  # noqa: BLE001 - report and continue
            failures.append(
                f"{index:03d}_{case.name}: {type(error).__name__}: {error}"
            )
            continue
        print(f"{'verified' if args.check_only else 'rendered'} "
              f"{index:03d}_{case.name}")
    if failures:
        print("\nFAILURES:")
        for line in failures:
            print(" ", line)
        raise SystemExit(1)
    print(f"\nAll 32 cases {'verified' if args.check_only else 'rendered'} "
          f"into {out_dir}")
