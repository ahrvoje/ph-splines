"""Acceptance tests for ``ClosedSpline_FillArea_Specification.md``.

Covers API topology (section 2), the simplicity fast path, decomposition
of figure-eight and limacon sources against an independent crossing-split
oracle, a cusp-forming offset, invariance laws, laziness and caching, and
serialization.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
from scipy.optimize import fsolve

import ph_spline.fill_area as fill_module
from ph_spline import (
    ClosedNURBSHandle,
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    NumericalPrecisionError,
    PHBSplineClosed,
    PHBSplineClosedSnapshot,
    PHBSplineOpen,
)


def circle(n=16, radius=1.0, cw=False):
    sign = -1.0 if cw else 1.0
    return [
        [radius * math.cos(sign * 2 * math.pi * k / n),
         radius * math.sin(sign * 2 * math.pi * k / n)]
        for k in range(n)
    ]


def figure_eight(n=32, phase=0.05, transform=None):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n + phase
        point = (math.sin(t), math.sin(t) * math.cos(t))
        if transform is not None:
            point = transform(point)
        out.append(list(point))
    return out


def limacon(n=24, phase=0.03):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n + phase
        r = 0.5 + math.cos(t)
        out.append([r * math.cos(t), r * math.sin(t)])
    return out


def octagon():
    return [[1.2, 0.0], [0.9, 0.9], [0.0, 1.2], [-0.9, 0.9], [-1.2, 0.0],
            [-0.9, -0.9], [0.0, -1.2], [0.9, -0.9]]


@pytest.fixture()
def stats():
    fill_module.reset_statistics()
    yield fill_module.statistics
    fill_module.reset_statistics()


# ---------------------------------------------------------------------------
# Independent crossing-split oracle (public API + quadrature only)
# ---------------------------------------------------------------------------


def find_crossing(curve, seed_u, seed_v):
    solution = fsolve(
        lambda x: curve.point(float(x[0])) - curve.point(float(x[1])),
        [seed_u, seed_v],
        xtol=1e-12,
    )
    u, v = sorted(float(value) for value in solution)
    residual = np.hypot(*(curve.point(u) - curve.point(v)))
    assert residual < 1e-12
    return u, v


def green_integral(curve, a, b):
    nodes, weights = np.polynomial.legendre.leggauss(24)
    knots = [float(k) for k in np.asarray(curve._span_knots) if a < k < b]
    total = 0.0
    for lo, hi in zip([a] + knots, knots + [b]):
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        for x, w in zip(nodes, weights):
            u = float(mid + half * x)
            p = curve.point(u)
            d = curve.derivative(u)
            total += w * half * (p[0] * d[1] - p[1] * d[0])
    return 0.5 * total


# ---------------------------------------------------------------------------
# API and topology
# ---------------------------------------------------------------------------


class TestFillAreaAPI:
    def test_closed_types_expose_fill_area(self):
        cubic = CubicPHSplineClosed(circle())
        bspline = PHBSplineClosed(octagon())
        for curve in (cubic, bspline):
            assert isinstance(curve.fill_area, float)
            assert curve.fill_area >= 0.0
        snapshot = bspline.snapshot()
        assert isinstance(snapshot, PHBSplineClosedSnapshot)
        assert snapshot.fill_area == bspline.fill_area
        handle = cubic.offset(0.1)
        assert isinstance(handle, ClosedNURBSHandle)
        assert isinstance(handle.fill_area, float)

    def test_open_types_have_no_fill_area(self):
        open_cubic = CubicPHSplineOpen([[0.0, 0.0], [1.0, 0.4], [2.0, 1.3]])
        open_bspline = PHBSplineOpen(
            [[0.0, 0.0], [1.0, 0.4], [2.0, -0.7], [3.0, 1.1]]
        )
        for target in (
            open_cubic,
            open_bspline,
            open_bspline.snapshot(),
            open_cubic.offset(0.1),
        ):
            assert not hasattr(target, "fill_area")


# ---------------------------------------------------------------------------
# Simplicity fast path (bitwise equality with ``area``)
# ---------------------------------------------------------------------------


class TestSimpleFastPath:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: CubicPHSplineClosed(circle()),
            lambda: CubicPHSplineClosed(circle(cw=True)),
            lambda: PHBSplineClosed(octagon()),
            lambda: PHBSplineClosed(octagon(), g_order=3),
        ],
        ids=["cubic_ccw", "cubic_cw", "bspline_g2", "bspline_g3"],
    )
    def test_simple_sources_bitwise(self, factory, stats):
        curve = factory()
        assert curve.fill_area == curve.area
        assert stats["simple_fast_path"] == 1
        assert stats["crossings_certified"] == 0

    def test_simple_offsets_bitwise(self, stats):
        curve = CubicPHSplineClosed(circle(n=24))
        for distance in (0.0, 0.25, -0.4):
            handle = curve.offset(distance)
            assert handle.fill_area == handle.area

    def test_reversed_loop_offset_bitwise(self):
        # d beyond every curvature radius: a simple reversed loop.
        handle = CubicPHSplineClosed(circle(n=24)).offset(1.6)
        assert handle.fill_area == handle.area


# ---------------------------------------------------------------------------
# Decomposition cases
# ---------------------------------------------------------------------------


class TestDecomposition:
    def test_figure_eight_matches_oracle(self, stats):
        curve = PHBSplineClosed(figure_eight())
        u, v = find_crossing(curve, 0.99, 0.50)
        lobe = green_integral(curve, u, v)
        expected = abs(lobe) + abs(curve.signed_area - lobe)
        fill = curve.fill_area
        assert fill == pytest.approx(expected, abs=1e-9)
        assert fill > curve.area  # opposite lobes cancel algebraically
        assert fill == pytest.approx(4.0 / 3.0, rel=1e-4)
        assert stats["crossings_certified"] == 1
        assert stats["loops_decomposed"] == 2

    def test_limacon_matches_outer_loop(self, stats):
        curve = PHBSplineClosed(limacon())
        u, v = find_crossing(curve, 0.30, 0.72)
        inner = green_integral(curve, u, v)
        outer = curve.signed_area - inner
        fill = curve.fill_area
        assert fill == pytest.approx(abs(outer), abs=1e-9)
        assert fill < curve.area  # the doubly wound core counts once
        assert stats["crossings_certified"] == 1
        assert stats["loops_decomposed"] == 2

    def test_cusped_offset_decomposes(self, stats):
        source = CubicPHSplineClosed(
            [[2.0 * math.cos(2 * math.pi * k / 24 + 0.013),
              1.0 * math.sin(2 * math.pi * k / 24 + 0.013)]
             for k in range(24)]
        )
        handle = source.offset(0.55)
        assert handle.cusps
        fill = handle.fill_area
        assert 0.0 < fill
        assert fill != handle.area
        assert stats["crossings_certified"] > 0

    def test_invariance_laws(self):
        reference = PHBSplineClosed(figure_eight()).fill_area
        reflected = PHBSplineClosed(
            figure_eight(transform=lambda p: (p[0], -p[1]))
        ).fill_area
        rotated = PHBSplineClosed(
            figure_eight(transform=lambda p: (-p[1], p[0]))
        ).fill_area
        reversed_fill = PHBSplineClosed(figure_eight()[::-1]).fill_area
        scaled = PHBSplineClosed(
            figure_eight(transform=lambda p: (2.0 * p[0], 2.0 * p[1]))
        ).fill_area
        assert reflected == pytest.approx(reference, rel=1e-9)
        assert rotated == pytest.approx(reference, rel=1e-9)
        assert reversed_fill == pytest.approx(reference, rel=1e-9)
        assert scaled == pytest.approx(4.0 * reference, rel=1e-9)


# ---------------------------------------------------------------------------
# Laziness, caching, versions, snapshots
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_no_fill_work_outside_queries(self, stats):
        curve = PHBSplineClosed(circle(n=12))
        _ = curve.signed_area
        curve.move_point(1, [1.03 * math.cos(0.5236), 1.03 * math.sin(0.5236)])
        snapshot = curve.snapshot()
        handle = curve.offset(0.1)
        pickle.loads(pickle.dumps(curve))
        pickle.loads(pickle.dumps(handle))
        assert stats["queries"] == 0
        _ = snapshot  # snapshots never trigger fill work either

    def test_warm_queries_cached(self, stats):
        curve = CubicPHSplineClosed(circle(n=12))
        first = curve.fill_area
        assert stats["queries"] == 1
        assert curve.fill_area == first
        assert stats["queries"] == 1

    def test_version_invalidation(self, stats):
        curve = PHBSplineClosed(circle(n=12))
        before = curve.fill_area
        assert stats["queries"] == 1
        curve.move_point(2, [0.55, 0.88])
        after = curve.fill_area
        assert stats["queries"] == 2
        assert after != before
        assert curve.fill_area == after
        assert stats["queries"] == 2

    def test_snapshot_retains_captured_fill(self):
        curve = PHBSplineClosed(circle(n=12))
        before = curve.fill_area
        snapshot = curve.snapshot()
        curve.move_point(0, [1.08, 0.02])
        assert snapshot.fill_area == before
        assert curve.fill_area != before

    def test_offset_handle_cached(self, stats):
        handle = CubicPHSplineClosed(circle(n=12)).offset(0.2)
        first = handle.fill_area
        queries = stats["queries"]
        assert handle.fill_area == first
        assert stats["queries"] == queries

    def test_fill_does_not_leak_into_area_cache_semantics(self, stats):
        curve = CubicPHSplineClosed(circle(n=12))
        _ = curve.signed_area
        assert stats["queries"] == 0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_cubic_round_trip(self, stats):
        curve = CubicPHSplineClosed(circle(n=12))
        value = curve.fill_area
        restored = pickle.loads(pickle.dumps(curve))
        assert restored._fill_area_cache is None
        assert stats["queries"] == 1
        assert restored.fill_area == value

    def test_bspline_round_trip(self, stats):
        curve = PHBSplineClosed(octagon())
        value = curve.fill_area
        restored = pickle.loads(pickle.dumps(curve))
        assert "_fill_area_cache" not in restored.__dict__
        assert restored.fill_area == value

    def test_handle_round_trip(self, stats):
        handle = CubicPHSplineClosed(circle(n=12)).offset(0.2)
        value = handle.fill_area
        restored = pickle.loads(pickle.dumps(handle))
        assert restored._fill_area_cache is None
        assert restored.fill_area == value
