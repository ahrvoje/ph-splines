"""Shared case model, verification and rendering for the offset galleries.

Each of the four ``*_offset`` example folders defines 32 cases and calls
:func:`run` with its curve factory.  Every rendered image first verifies
each exact offset NURBS against ``r(u) + d * N_L(u)`` through the public
geometry API; a case that fails verification aborts the generation run.
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

PAGE = "#f9f9f7"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK = "#172033"
MUTED = "#8a94a6"
POINT = "#eb6834"
LEFT_SIDE = "#246fc2"   # positive distances (left normal)
RIGHT_SIDE = "#c45b31"  # negative distances (right normal)


@dataclass(frozen=True)
class OffsetCase:
    """One rendered offset study."""

    name: str
    points: list[list[float]]
    distances: tuple[float, ...]
    note: str
    category: str
    fill: bool = False
    kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reusable geometry builders
# ---------------------------------------------------------------------------


def periodic(function, count: int, *, phase: float = 0.017) -> list[list[float]]:
    t = phase + np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    x, y = function(t)
    return np.column_stack((x, y)).astype(np.float64).tolist()


def polar(radius, count: int, *, phase: float = 0.017) -> list[list[float]]:
    return periodic(lambda t: (radius(t) * np.cos(t), radius(t) * np.sin(t)),
                    count, phase=phase)


def open_curve(function, t0: float, t1: float, count: int) -> list[list[float]]:
    t = np.linspace(t0, t1, count)
    x, y = function(t)
    return np.column_stack((x, y)).astype(np.float64).tolist()


def catmull_rom(
    keys: list[tuple[float, float]], samples: int = 5, *, closed: bool = True
) -> list[list[float]]:
    """Sample a Catmull-Rom design polygon (periodic or clamped open)."""
    values = np.asarray(keys, dtype=np.float64)
    count = len(values)
    result: list[list[float]] = []
    spans = count if closed else count - 1
    for index in range(spans):
        if closed:
            p0 = values[(index - 1) % count]
            p1 = values[index]
            p2 = values[(index + 1) % count]
            p3 = values[(index + 2) % count]
        else:
            p0 = values[max(index - 1, 0)]
            p1 = values[index]
            p2 = values[index + 1]
            p3 = values[min(index + 2, count - 1)]
        for sample in range(samples):
            u = sample / samples
            point = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * u
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u**2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u**3
            )
            result.append(point.tolist())
    if not closed:
        result.append(values[-1].tolist())
    return result


def meander(
    length: float,
    amplitude,
    count: int,
    *,
    frequency: float = 1.0,
    phase: float = 0.0,
) -> list[list[float]]:
    """Sinuous open path with optionally varying amplitude."""
    x = np.linspace(0.0, length, count)
    if callable(amplitude):
        y = amplitude(x)
    else:
        y = amplitude * np.sin(2.0 * math.pi * frequency * x / length + phase)
    return np.column_stack((x, y)).astype(np.float64).tolist()


def seeded_walk(
    seed: int,
    count: int,
    *,
    step: float = 1.0,
    turn_scale: float = 0.45,
    drift: float = 0.0,
) -> list[list[float]]:
    """Deterministic random-walk polyline with bounded turns."""
    rng = np.random.default_rng(seed)
    heading = rng.uniform(-0.4, 0.4)
    x, y = 0.0, 0.0
    points = [[x, y]]
    for _ in range(count - 1):
        heading += rng.uniform(-turn_scale, turn_scale) + drift
        x += step * math.cos(heading)
        y += step * math.sin(heading)
        points.append([x, y])
    return points


# ---------------------------------------------------------------------------
# Verification and rendering
# ---------------------------------------------------------------------------


def verify_offset(curve, handle, distance: float, *, samples: int = 257) -> float:
    """Compare the handle against ``r(u) + d * N_L(u)``; return the worst gap."""
    extent = max(1.0, float(np.max(np.abs(curve.points if hasattr(curve, "points") else curve._points))))
    tolerance = 1e-8 * max(extent, abs(distance))
    worst = 0.0
    for u in np.linspace(0.0, 1.0, samples):
        u = float(u)
        expected = curve.point(u) + distance * curve.normal(u)
        gap = float(np.hypot(*(handle.point(u) - expected)))
        worst = max(worst, gap)
    if not worst <= tolerance:
        raise RuntimeError(
            f"offset verification failed: gap {worst:.3e} > {tolerance:.3e} "
            f"at distance {distance}"
        )
    return worst


def _offset_color(distance: float, rank: float) -> tuple:
    base = LEFT_SIDE if distance >= 0.0 else RIGHT_SIDE
    rgb = matplotlib.colors.to_rgb(base)
    # Blend toward the page for far offsets so the fan reads by depth.
    blend = 0.15 + 0.5 * rank
    return tuple((1.0 - blend) * c + blend * 1.0 for c in rgb)


def render_case(
    out_dir: Path,
    index: int,
    case: OffsetCase,
    curve,
    family_label: str,
    *,
    check_only: bool = False,
) -> None:
    handles = [(d, curve.offset(d)) for d in case.distances]
    for distance, handle in handles:
        verify_offset(curve, handle, distance)
    if check_only:
        return

    points = np.asarray(case.points, dtype=np.float64)
    u_grid = np.linspace(0.0, 1.0, 900)
    source = np.asarray([curve.point(float(u)) for u in u_grid])

    fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=105, facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    if case.fill:
        ax.fill(source[:, 0], source[:, 1], color=INK, alpha=0.045, zorder=1)

    magnitudes = sorted({abs(d) for d, _ in handles})
    labeled = {1.0: False, -1.0: False}
    for distance, handle in handles:
        rank = (
            magnitudes.index(abs(distance)) / max(1, len(magnitudes) - 1)
            if len(magnitudes) > 1
            else 0.0
        )
        samples = np.asarray([handle.point(float(u)) for u in u_grid])
        side = 1.0 if distance >= 0.0 else -1.0
        label = None
        if not labeled[side]:
            label = f"exact NURBS offsets ({'left' if side > 0 else 'right'})"
            labeled[side] = True
        ax.plot(
            samples[:, 0], samples[:, 1],
            color=_offset_color(distance, rank),
            linewidth=1.35, zorder=2, label=label,
        )

    ax.plot(source[:, 0], source[:, 1], color=INK, linewidth=2.1, zorder=3,
            label=family_label)
    step = max(1, math.ceil(len(points) / 40))
    shown = points[::step]
    ax.scatter(shown[:, 0], shown[:, 1], s=11, facecolor=SURFACE,
               edgecolor=POINT, linewidth=0.7, zorder=4,
               label="interpolation nodes")

    degree = handles[0][1].degree
    d_values = ", ".join(f"{d:+g}" for d, _ in handles)
    fig.suptitle(f"{index:03d} · {case.name.replace('_', ' ')}", x=0.06,
                 ha="left", color=INK, fontsize=10.5)
    ax.set_title(
        f"{case.category} · {curve.num_points} nodes · degree-{degree} "
        f"rational offsets · d = {d_values}\n{case.note}",
        loc="left", color=INK, fontsize=7.15,
    )
    ax.legend(loc="best", frameon=False, fontsize=6.8)
    fig.savefig(out_dir / f"{index:03d}_{case.name}.png",
                bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)


def run(cases: list[OffsetCase], build_curve, out_dir: Path,
        family_label: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="verify all offsets without rendering")
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
            failures.append(f"{index:03d}_{case.name}: {type(error).__name__}: {error}")
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
