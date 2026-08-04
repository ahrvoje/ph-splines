"""Shared data generators for the acceptance test suite."""

from __future__ import annotations

import math

import numpy as np
import pytest


def circle_points(R=1.0, a0=0.0, a1=1.5, n=8, cw=False, center=(0.0, 0.0)):
    angles = np.linspace(a0, a1, n)
    if cw:
        angles = -angles
    return [[center[0] + R * math.cos(a), center[1] + R * math.sin(a)] for a in angles]


def spiral_points(a=1.0, b=0.12, t0=0.0, t1=4 * math.pi, n=30):
    ts = np.linspace(t0, t1, n)
    return [
        [a * math.exp(b * t) * math.cos(t), a * math.exp(b * t) * math.sin(t)]
        for t in ts
    ]


def parabola_points(x0=0.2, x1=2.0, n=7):
    xs = np.linspace(x0, x1, n)
    return [[float(x), float(x * x)] for x in xs]


def polyline_from_turns(lengths, turns, tau=1, psi0=0.0, origin=(0.0, 0.0)):
    """Open polyline with prescribed chord lengths and interior turns."""
    pts = [list(origin)]
    psi = psi0
    for k, L in enumerate(lengths):
        if k > 0:
            psi += tau * turns[k - 1]
        p = pts[-1]
        pts.append([p[0] + L * math.cos(psi), p[1] + L * math.sin(psi)])
    return pts


def nonuniform_arc_points():
    """Circle samples with chord lengths spanning about six decades."""
    R = 10.0
    angles = [0.0]
    step = 1e-7
    while len(angles) < 9:
        angles.append(angles[-1] + step)
        step *= 10.0**0.75
    return [[R * math.cos(a), R * math.sin(a)] for a in angles]


CURVED_CASES = {
    "circle_ccw": circle_points(R=2.0, a0=-0.6, a1=1.2, n=7),
    "circle_cw": circle_points(R=1.5, a0=0.2, a1=1.9, n=9, cw=True),
    "circle_dense": circle_points(R=1.0, a0=0.0, a1=2.0, n=41),
    "parabola": parabola_points(),
    "spiral": spiral_points(),
    "near_bound": polyline_from_turns([1.0, 1.0, 1.0], [2.0, 2.0]),
    "nonuniform": nonuniform_arc_points(),
    "tiny_curvature": circle_points(R=1e8, a0=0.0, a1=1e-3, n=6),
    "offset_scaled": circle_points(R=3.0, a0=0.3, a1=1.7, n=8, center=(500.0, -200.0)),
}

STRAIGHT_CASES = {
    "two_point": [[0.0, 0.0], [3.0, 4.0]],
    "collinear_uniform": [[float(i), 2.0 * i] for i in range(6)],
    "collinear_nonuniform": [
        [0.0, 0.0],
        [0.1, 0.05],
        [1.0, 0.5],
        [4.5, 2.25],
        [11.0, 5.5],
    ],
}


@pytest.fixture(params=sorted(CURVED_CASES), ids=sorted(CURVED_CASES))
def curved_case(request):
    return CURVED_CASES[request.param]


@pytest.fixture(params=sorted(STRAIGHT_CASES), ids=sorted(STRAIGHT_CASES))
def straight_case(request):
    return STRAIGHT_CASES[request.param]


@pytest.fixture(
    params=sorted(CURVED_CASES) + sorted(STRAIGHT_CASES),
    ids=sorted(CURVED_CASES) + sorted(STRAIGHT_CASES),
)
def any_case(request):
    if request.param in CURVED_CASES:
        return CURVED_CASES[request.param]
    return STRAIGHT_CASES[request.param]
