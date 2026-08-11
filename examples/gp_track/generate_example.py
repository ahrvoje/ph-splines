"""GP track exact-offset showcase.

Run from the repository root:
    python examples/gp_track/generate_example.py

A clockwise grand-prix circuit centerline - pit straight, a tight first
corner, a long uphill run into a hairpin, a downhill sweep through a
right-hander and two left kinks, a fast right, and a double-right final
complex - is interpolated as a closed cubic PH spline. The 13 m F1 track
width is realized with exact rational NURBS offsets at +-6.5 m, a
red-white kerb band with an additional offset layer at +-7.35 m, and a
racing lap - built inside the same width envelope - runs wide on entry,
clips every apex, and releases wide on exit. Corner insets zoom into the
hairpin, the first corner, and the final complex. Every offset handle is
verified against ``r(u) + d * N_L(u)`` before rendering.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offset_common import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    PAGE,
    SURFACE,
    verify_offset,
)

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ph_spline import CubicPHSplineClosed  # noqa: E402

OUT = Path(__file__).resolve().parent

ASPHALT = "#e6e5e0"
EDGE = "#565f73"
KERB_RED = "#c2242b"
KERB_WHITE = "#fcfcfb"
LAP_BLUE = "#1e63d0"

HALF_WIDTH = 6.5     # 13 m F1-grade track width
KERB_INNER = 6.6     # kerb band inner boundary, outside the track edge
KERB_OUTER = 10.2    # kerb band outer boundary (kerb plus painted apron)
CAR_MARGIN = 1.15    # half car width + small safety gap
APEX = HALF_WIDTH - CAR_MARGIN
TARGET_LENGTH = 4318.0  # lap length in meters

# ---------------------------------------------------------------------------
# Track geometry: clockwise lap, headings in math degrees
# ---------------------------------------------------------------------------

# (name, delta_deg, length_or_radius, is_arc): a lap starts at the exit of
# the final corner onto the pit straight, travelling west. Right-handers
# turn clockwise (negative delta). The deltas sum to exactly -360.
FEATURES = [
    ("pit_straight", -4.0, 700.0, False),
    ("t1", -112.0, 45.0, True),
    ("uphill_run", 11.0, 940.0, False),
    ("t3_hairpin", -124.0, 33.0, True),
    ("downhill_run", -13.0, 500.0, False),
    ("t4", -74.0, 62.0, True),
    ("link_t4_t5", -2.0, 110.0, False),
    ("t5", 42.0, 125.0, True),
    ("link_t5_t6", 2.0, 70.0, False),
    ("t6", 44.0, 105.0, True),
    ("run_to_t7", -3.0, 290.0, False),
    ("t7", -49.0, 88.0, True),
    ("t8_sweep", 36.0, 480.0, True),
    ("run_to_t9", -2.0, 200.0, False),
    ("t9", -57.0, 115.0, True),
    ("link_t9_t10", -1.0, 75.0, False),
    ("t10", -54.0, 56.0, True),
]

START_HEADING = 178.0  # travelling west along the pit straight
# Closure corrections go only into the three long runs whose headings are
# far apart; short links are excluded so no leg can be driven negative.
ADJUSTABLE = ("pit_straight", "uphill_run", "downhill_run")


def _trace(features):
    """Integrate the feature list into a dense polyline with stations."""
    heading = math.radians(START_HEADING)
    x, y, s = 0.0, 0.0, 0.0
    dense = [(0.0, 0.0, 0.0)]
    stations = {}
    for name, delta_deg, size, is_arc in features:
        delta = math.radians(delta_deg)
        if is_arc:
            length = abs(delta) * size
        else:
            length = size
            size = length / abs(delta) if delta != 0.0 else math.inf
        steps = max(8, int(math.ceil(length / 4.0)))
        s0 = s
        for _ in range(steps):
            step_delta = delta / steps
            chord = (
                2.0 * size * math.sin(abs(step_delta) / 2.0)
                if math.isfinite(size) and step_delta != 0.0
                else length / steps
            )
            mid = heading + step_delta / 2.0
            x += chord * math.cos(mid)
            y += chord * math.sin(mid)
            heading += step_delta
            s += chord
            dense.append((x, y, s))
        stations[name] = (s0, s, delta_deg, size)
    return np.asarray(dense), stations


def build_centerline():
    """Close the loop exactly, then emit interpolation nodes."""
    features = [list(item) for item in FEATURES]
    for _ in range(4):
        dense, stations = _trace([tuple(item) for item in features])
        gap = dense[-1, :2]
        if np.hypot(*gap) < 1.0e-6:
            break
        # Distribute the closure error over the long runs by a minimum-norm
        # least-squares length adjustment along each run's mid heading.
        columns = []
        for name in ADJUSTABLE:
            s0, s1, _, _ = stations[name]
            i0 = int(np.searchsorted(dense[:, 2], 0.5 * (s0 + s1)))
            direction = dense[min(i0 + 1, len(dense) - 1), :2] - dense[i0, :2]
            columns.append(direction / np.linalg.norm(direction))
        matrix = np.column_stack(columns)
        adjust, *_ = np.linalg.lstsq(matrix, -gap, rcond=None)
        for name, value in zip(ADJUSTABLE, adjust):
            for item in features:
                if item[0] == name:
                    item[2] += float(value)
    dense, stations = _trace([tuple(item) for item in features])
    assert np.hypot(*dense[-1, :2]) < 1.0e-6, "track loop failed to close"
    # A negative adjusted leg would fold the trace back onto itself and
    # produce a curvature spike whose exact offset must then loop.
    for name, _, size, is_arc in features:
        assert is_arc or size > 40.0, f"leg {name} collapsed to {size:.1f} m"
    assert np.all(np.diff(dense[:, 2]) > 0.0), "stations are not monotone"

    # Scale uniformly to the target lap length; the offsets keep their
    # physical meaning because everything below is in meters.
    factor = TARGET_LENGTH / float(dense[-1, 2])
    dense[:, :] *= factor
    stations = {
        name: (s0 * factor, s1 * factor, delta_deg, radius * factor)
        for name, (s0, s1, delta_deg, radius) in stations.items()
    }
    total = float(dense[-1, 2])
    nodes = []
    for name, (s0, s1, delta_deg, radius) in stations.items():
        length = s1 - s0
        if radius < 250.0:
            count = max(2, int(math.ceil(abs(delta_deg) / 13.0)))
        else:
            count = max(2, int(math.ceil(length / 95.0)))
        for k in range(count):
            s_node = s0 + length * k / count
            nodes.append((
                float(np.interp(s_node, dense[:, 2], dense[:, 0])),
                float(np.interp(s_node, dense[:, 2], dense[:, 1])),
            ))
    centerline = CubicPHSplineClosed(nodes)
    # The interpolated lap must stay far from the offset cusp condition
    # 1 - d*kappa = 0 for every drawn layer, kerbs included.
    kappa = max(
        abs(centerline.signed_curvature(float(u)))
        for u in np.linspace(0.0, 1.0, 4001)
    )
    assert 1.0 / kappa > 2.0 * KERB_OUTER, (
        f"centerline radius {1.0 / kappa:.1f} m too tight for the offsets"
    )
    return centerline, np.asarray(nodes), stations, total


# ---------------------------------------------------------------------------
# Racing lap: wide on entry, apex on the inside edge, wide on exit
# ---------------------------------------------------------------------------


def racing_waypoints(stations):
    """Lateral targets (station, offset): +e is left of travel = outside."""
    def corner(name):
        s0, s1, delta, _ = stations[name]
        return s0, s1, 0.5 * (s0 + s1), 1.0 if delta > 0.0 else -1.0

    a = APEX
    w = []
    t1 = corner("t1")
    t3 = corner("t3_hairpin")
    t4 = corner("t4")
    t5 = corner("t5")
    t6 = corner("t6")
    t7 = corner("t7")
    sweep = corner("t8_sweep")
    t9 = corner("t9")
    t10 = corner("t10")

    w += [(60.0, a), (t1[0] - 70.0, a), (t1[2], -a), (t1[1] + 80.0, 0.9 * a)]
    w += [(t3[0] - 90.0, a), (t3[2], -a), (t3[1] + 90.0, 0.92 * a)]
    w += [(t4[0] - 80.0, a), (t4[2], -a), (t4[1] + 70.0, 0.85 * a)]
    w += [(t5[0] - 60.0, -0.72 * a), (t5[2], 0.88 * a)]
    w += [(0.5 * (t5[1] + t6[0]), 0.6 * a), (t6[2], 0.94 * a)]
    w += [(t6[1] + 70.0, -0.5 * a)]
    w += [(t7[0] - 70.0, 0.92 * a), (t7[2], -a), (t7[1] + 80.0, 0.64 * a)]
    w += [(sweep[2], 0.3 * a)]
    w += [(t9[0] - 90.0, a), (t9[2], -0.97 * a)]
    w += [(0.5 * (t9[1] + t10[0]), -0.2 * a), (t10[2], -a),
          (t10[1] + 60.0, a)]
    return sorted(w)


def lateral_profile(waypoints, total):
    """Periodic C1 cosine blend through the lateral waypoints."""
    s_values = [s for s, _ in waypoints]
    e_values = [e for _, e in waypoints]
    s_values.append(s_values[0] + total)
    e_values.append(e_values[0])

    def profile(s):
        s = s % total
        if s < s_values[0]:
            s += total
        index = int(np.searchsorted(s_values, s, side="right")) - 1
        index = max(0, min(index, len(s_values) - 2))
        span = s_values[index + 1] - s_values[index]
        t = (s - s_values[index]) / span
        blend = 0.5 - 0.5 * math.cos(math.pi * t)
        return e_values[index] + blend * (e_values[index + 1] - e_values[index])

    return profile


def build_racing_line(centerline, stations, total):
    profile = lateral_profile(racing_waypoints(stations), total)
    length = centerline.length
    count = int(round(total / 29.0))
    nodes = []
    for k in range(count):
        s = total * k / count
        u = centerline.parameter_at_length(s / total * length)
        point = centerline.point(u)
        normal = centerline.normal(u)
        nodes.append((point + profile(s) * normal).tolist())
    return CubicPHSplineClosed(nodes)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _sample(spline_or_handle, count=1500):
    return np.asarray(
        [spline_or_handle.point(float(u)) for u in np.linspace(0.0, 1.0, count)]
    )


def _station_curve(handle, centerline, total, s0, s1, step=2.5):
    """Sample one offset handle over a station range of the source lap."""
    length = centerline.length
    values = []
    for s in np.arange(s0, s1, step):
        u = centerline.parameter_at_length((s % total) / total * length)
        values.append(handle.point(float(u)))
    return np.asarray(values)


def render(check_only: bool) -> None:
    centerline, nodes, stations, total = build_centerline()
    edges = {side: centerline.offset(side * HALF_WIDTH) for side in (+1, -1)}
    kerb_inner = {
        side: centerline.offset(side * KERB_INNER) for side in (+1, -1)
    }
    kerb_outer = {
        side: centerline.offset(side * KERB_OUTER) for side in (+1, -1)
    }
    for handles, d in (
        (edges, HALF_WIDTH), (kerb_inner, KERB_INNER),
        (kerb_outer, KERB_OUTER),
    ):
        for side in (+1, -1):
            verify_offset(centerline, handles[side], side * d)
    lap = build_racing_line(centerline, stations, total)
    print(f"track {centerline.length:.0f} m, lap {lap.length:.0f} m, "
          f"{centerline.num_points} centerline nodes, offsets verified")
    if check_only:
        return

    fig, ax = plt.subplots(figsize=(8.4, 7.0), dpi=292, facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("east  [m]", color=MUTED, fontsize=7)
    ax.set_ylabel("north  [m]", color=MUTED, fontsize=7)
    ax.tick_params(labelsize=6.5, colors=MUTED)

    left = _sample(edges[+1])
    right = _sample(edges[-1])
    ribbon = np.vstack((left, right[::-1]))
    center = _sample(centerline)
    lap_samples = _sample(lap, 2200)

    def draw_kerbs(target):
        # The kerb band is an exact data-space polygon strip between the
        # inner and outer kerb offsets, so it stays outside the track edge
        # at every zoom level.
        chunk = max(1, int(round(10.5 / 2.5)))
        for name, (s0, s1, delta, radius) in stations.items():
            if radius >= 250.0 or abs(delta) < 40.0:
                continue
            inside = 1.0 if delta > 0.0 else -1.0
            for side, za, zb in ((inside, s0 - 14.0, s1 + 18.0),
                                 (-inside, 0.5 * (s0 + s1), s1 + 52.0)):
                inner = _station_curve(
                    kerb_inner[int(side)], centerline, total, za, zb
                )
                outer = _station_curve(
                    kerb_outer[int(side)], centerline, total, za, zb
                )
                for start in range(0, len(inner) - 1, chunk):
                    color = (
                        KERB_RED if (start // chunk) % 2 == 0 else KERB_WHITE
                    )
                    stop = min(start + chunk + 1, len(inner))
                    quad = np.vstack(
                        (inner[start:stop], outer[start:stop][::-1])
                    )
                    target.fill(quad[:, 0], quad[:, 1], facecolor=color,
                                edgecolor="#b9b4ac", linewidth=0.25,
                                zorder=2)

    def draw_world(target, *, edge_width, lap_width, center_width):
        target.fill(ribbon[:, 0], ribbon[:, 1], color=ASPHALT, zorder=1)
        for samples in (left, right):
            target.plot(samples[:, 0], samples[:, 1], color=EDGE,
                        linewidth=edge_width, zorder=3)
        draw_kerbs(target)
        target.plot(center[:, 0], center[:, 1], color=MUTED,
                    linewidth=center_width, linestyle=(0, (5, 4)), zorder=4)
        target.plot(lap_samples[:, 0], lap_samples[:, 1], color=LAP_BLUE,
                    linewidth=lap_width, zorder=5)

    draw_world(ax, edge_width=0.9, lap_width=1.6, center_width=0.7)
    ax.plot([], [], color=MUTED, linewidth=0.7, linestyle=(0, (5, 4)),
            label="closed cubic PH centerline")
    ax.plot([], [], color=EDGE, linewidth=1.2,
            label="track edges · exact NURBS offsets ±6.5 m")
    ax.plot([], [], color=KERB_RED, linewidth=2.6,
            label="kerb band · offset layers 6.6-10.2 m")
    ax.plot([], [], color=LAP_BLUE, linewidth=1.8,
            label="racing lap · PH spline")

    # start/finish mark and race direction
    pit0, pit1, _, _ = stations["pit_straight"]
    s_sf = pit0 + 0.62 * (pit1 - pit0)
    u_sf = centerline.parameter_at_length(s_sf / total * centerline.length)
    p_sf = centerline.point(u_sf)
    n_sf = centerline.normal(u_sf)
    t_sf = centerline.tangent(u_sf)
    bar = np.asarray([p_sf + HALF_WIDTH * n_sf, p_sf - HALF_WIDTH * n_sf])
    ax.plot(bar[:, 0], bar[:, 1], color=INK, linewidth=3.2,
            solid_capstyle="butt", zorder=6)
    ax.annotate("S/F", bar[0] + np.array([0.0, -34.0]), color=INK,
                fontsize=7, fontweight="bold", ha="center", zorder=6)
    ax.annotate("", xy=p_sf + 90.0 * t_sf + 16.0 * n_sf,
                xytext=p_sf + 20.0 * t_sf + 16.0 * n_sf,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.1),
                zorder=6)

    # neutral corner numbers, placed outside each corner
    labels = {
        "t1": ("T1", 55.0),
        "t3_hairpin": ("T3", 60.0),
        "t4": ("T4", 80.0),
        "t6": ("T5-T6", 70.0),
        "t7": ("T7", 70.0),
        "t9": ("T9", 80.0),
        "t10": ("T10", 60.0),
    }
    def corner_point(name):
        s0, s1, _, _ = stations[name]
        u = centerline.parameter_at_length(
            0.5 * (s0 + s1) / total * centerline.length
        )
        return centerline.point(u), centerline.normal(u)

    for name, (text, distance) in labels.items():
        _, _, delta, _ = stations[name]
        outward = (1.0 if delta > 0.0 else -1.0) * -1.0
        point, normal = corner_point(name)
        ax.annotate(text, point + outward * distance * normal, color=MUTED,
                    fontsize=6.2, ha="center", va="center", zorder=6)

    # corner insets: hairpin, first corner, and the final complex, each
    # placed next to its corner and large enough for close analysis
    def add_inset(bounds, x_window, y_window, title):
        inset = ax.inset_axes(bounds)
        inset.set_facecolor(SURFACE)
        inset.set_aspect("equal")
        draw_world(inset, edge_width=1.2, lap_width=2.4, center_width=0.9)
        inset.set_xlim(*x_window)
        inset.set_ylim(*y_window)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(MUTED)
            spine.set_linewidth(0.7)
        inset.set_title(title, color=MUTED, fontsize=6.4, pad=2.0)
        ax.indicate_inset_zoom(inset, edgecolor=MUTED, linewidth=0.7,
                               alpha=0.6)
        return inset

    t3_focus, _ = corner_point("t3_hairpin")
    add_inset(
        [0.02, 0.60, 0.36, 0.36],
        (t3_focus[0] - 46.0, t3_focus[0] + 46.0),
        (t3_focus[1] - 71.3, t3_focus[1] + 20.7),
        "T3 hairpin",
    )
    t1_focus, _ = corner_point("t1")
    add_inset(
        [0.005, 0.25, 0.324, 0.324],
        (t1_focus[0] - 42.0, t1_focus[0] + 42.0),
        (t1_focus[1] - 35.2, t1_focus[1] + 48.8),
        "T1",
    )
    t9_focus, _ = corner_point("t9")
    t10_focus, _ = corner_point("t10")
    apex_focus = t9_focus + 0.15 * (t10_focus - t9_focus)
    half = 66.0
    add_inset(
        [0.72, 0.02, 0.324, 0.324],
        (apex_focus[0] - half, apex_focus[0] + half),
        (apex_focus[1] - half, apex_focus[1] + half),
        "T9",
    )

    fig.suptitle("GP Track", x=0.07, ha="left", color=INK, fontsize=11.5)
    ax.set_title(
        f"clockwise lap · {centerline.num_points}-node closed cubic PH "
        f"centerline · 13 m F1 width from exact degree-5 rational offsets\n"
        f"racing lap rides the width envelope: wide entry, apex at the "
        f"inside edge, wide exit — {lap.length:.0f} m per lap",
        loc="left", color=INK, fontsize=7.4,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=6.6)
    fig.savefig(OUT / "gp_track.png", bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    print(f"rendered {OUT / 'gp_track.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    render(parser.parse_args().check_only)
