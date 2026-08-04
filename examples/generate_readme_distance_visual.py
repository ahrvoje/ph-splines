"""Generate the shallow distance-domain visual used near the README opening."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from cubic_ph_spline import CubicPHSpline


OUT = os.path.join(os.path.dirname(__file__), "readme_distance_evaluation.png")


def main() -> None:
    # A shallow inspection route with several curvature-sign changes. Scale it
    # to exactly 20 m so the labels are ordinary engineering distances.
    base = np.array(
        [
            [0.0, 0.00],
            [1.0, 0.18],
            [2.0, 0.52],
            [3.1, 0.66],
            [4.2, 0.28],
            [5.2, -0.24],
            [6.4, -0.58],
            [7.6, -0.42],
            [8.7, 0.10],
            [9.7, 0.52],
            [10.8, 0.58],
            [12.0, 0.10],
        ],
        dtype=np.float64,
    )
    preliminary = CubicPHSpline(base.tolist())
    base *= 20.0 / preliminary.arc_length(1.0)
    curve = CubicPHSpline(base.tolist())
    total = curve.arc_length(1.0)
    if abs(total - 20.0) > 5e-13:
        raise AssertionError(f"expected a 20 m route, got {total!r}")

    distances = np.linspace(0.0, total, 900)
    route = np.array([curve.point_at_length(float(s)) for s in distances])
    metre_marks = np.array([curve.point_at_length(float(s)) for s in range(21)])
    major_distances = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    major_marks = np.array([curve.point_at_length(float(s)) for s in major_distances])

    interval = np.linspace(7.0, 12.0, 220)
    interval_xy = np.array([curve.point_at_length(float(s)) for s in interval])
    a = curve.point_at_length(7.0)
    b = curve.point_at_length(12.0)
    middle = curve.point_at_length(9.5)

    fig, ax = plt.subplots(figsize=(12.0, 2.65), dpi=140)
    fig.subplots_adjust(left=0.018, right=0.985, top=0.89, bottom=0.10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    # Recessive construction data, then the distance-addressable route.
    ax.plot(
        base[:, 0], base[:, 1],
        linestyle=(0, (3, 4)), color="#a0a7b1", linewidth=0.8, zorder=1,
    )
    ax.plot(route[:, 0], route[:, 1], color="#2474d2", linewidth=3.0, zorder=2)
    ax.scatter(
        base[:, 0], base[:, 1], s=15, facecolor="white", edgecolor="#59636f",
        linewidth=0.8, zorder=3,
    )

    # Uniform one-metre work/sensor locations and labelled five-metre stations.
    ax.scatter(
        metre_marks[:, 0], metre_marks[:, 1], s=18, facecolor="#e94f4f",
        edgecolor="white", linewidth=0.75, zorder=4,
    )
    ax.scatter(
        major_marks[:, 0], major_marks[:, 1], s=48, facecolor="#e94f4f",
        edgecolor="white", linewidth=1.25, zorder=5,
    )
    for distance, point in zip(major_distances, major_marks):
        vertical = 15 if distance in (5.0, 15.0) else -20
        label = (
            "START · s = 0 m"
            if distance == 0.0
            else ("s = 20 m · END" if distance == 20.0 else f"s = {distance:.0f} m")
        )
        ax.annotate(
            label,
            xy=point,
            xytext=(0, vertical),
            textcoords="offset points",
            ha="center",
            va="bottom" if vertical > 0 else "top",
            fontsize=8.2,
            color="#3d4650",
            fontweight="bold" if distance in (0.0, 20.0) else "normal",
        )

    # A concrete pair of distance queries. The orange curve—not its chord—is
    # the measured five-metre interval.
    ax.plot(interval_xy[:, 0], interval_xy[:, 1], color="#f2992e", linewidth=5.2, zorder=3)
    ax.scatter(
        [a[0], b[0]], [a[1], b[1]], s=62, facecolor="#f2992e",
        edgecolor="white", linewidth=1.4, zorder=6,
    )
    ax.annotate(
        "A · s = 7 m", xy=a, xytext=(-6, -31),
        textcoords="offset points", ha="right", va="top", fontsize=8.0,
        color="#9a5800", arrowprops={"arrowstyle": "-", "color": "#c87912", "lw": 0.8},
    )
    ax.annotate(
        "B · s = 12 m", xy=b, xytext=(7, 28),
        textcoords="offset points", ha="left", va="bottom", fontsize=8.0,
        color="#9a5800", arrowprops={"arrowstyle": "-", "color": "#c87912", "lw": 0.8},
    )
    ax.annotate(
        "path distance A → B = 5 m",
        xy=middle,
        xytext=(0, 35),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#8a4d00",
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "#c87912", "lw": 0.9},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#efc17e", "alpha": 0.95},
    )

    ax.margins(x=0.035, y=0.28)
    fig.savefig(OUT, facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
