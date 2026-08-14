"""Acceptance tests for ``ClosedSpline_Area_Specification.md``.

Coverage map:

- section 14.2  algebraic coefficient-table and kernel identities;
- section 15.1  API and structural topology;
- section 15.2  exact elementary shapes and metamorphic laws;
- section 15.3  closed cubic PH cases against an exact rational oracle;
- section 15.4  closed PH B-spline lifecycle, editing, reuse, snapshots;
- section 15.5  exact offset area, signs, cusps, turning numbers, and the
  independent high-precision parallel-curve oracle;
- section 15.6  numerical stress (translation, scale, subnormals, zeros,
  cancellation, overflow, rounding boundaries, concurrency);
- section 15.7  serialization.

The exact rational polygon/Bezier oracles below use their own power-basis
conversion and integration code so they share no pipeline with the
production Bernstein kernels.
"""

from __future__ import annotations

import math
import pickle
import threading
from fractions import Fraction
from math import comb

import numpy as np
import pytest

import ph_spline.area as area_module
from ph_spline import (
    ClosedNURBSHandle,
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    NumericalPolicy,
    NumericalPrecisionError,
    NURBSHandle,
    OffsetConstructionError,
    PHBSplineClosed,
    PHBSplineClosedSnapshot,
    PHBSplineOpen,
    PHBSplineSnapshot,
)
from ph_spline.area import (
    OffsetAreaProvenance,
    area_coefficients,
    offset_signed_area,
    source_signed_area,
    turning_number,
)

# ---------------------------------------------------------------------------
# Independent exact oracles (test-only pipelines)
# ---------------------------------------------------------------------------


def _power_coefficients(controls: list[Fraction]) -> list[Fraction]:
    """Bernstein-to-power conversion written independently for the tests."""
    degree = len(controls) - 1
    out = []
    for k in range(degree + 1):
        acc = Fraction(0)
        for j in range(k + 1):
            term = comb(k, j) * controls[j]
            acc += term if (k - j) % 2 == 0 else -term
        out.append(comb(degree, k) * acc)
    return out


def _power_product(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _power_derivative(p: list[Fraction]) -> list[Fraction]:
    return [k * p[k] for k in range(1, len(p))] or [Fraction(0)]


def span_area_oracle(controls) -> Fraction:
    """Exact ``(1/2) integral (x y' - y x') dt`` by power-basis integration."""
    xs = _power_coefficients([Fraction(float(row[0])) for row in controls])
    ys = _power_coefficients([Fraction(float(row[1])) for row in controls])
    integrand = [
        u - v
        for u, v in zip(
            _power_product(xs, _power_derivative(ys))
            + [Fraction(0)] * len(xs),
            _power_product(ys, _power_derivative(xs))
            + [Fraction(0)] * len(xs),
        )
    ]
    return sum(
        (c / (2 * (k + 1)) for k, c in enumerate(integrand)), Fraction(0)
    )


def chain_area_oracle(spans, scale: float) -> Fraction:
    """Exact composite area (spans + cyclic residual joins) times ``H**2``."""
    total = sum((span_area_oracle(c) for c in spans), Fraction(0))
    count = len(spans)
    for index in range(count):
        end = spans[index][-1]
        start = spans[(index + 1) % count][0]
        ex, ey = Fraction(float(end[0])), Fraction(float(end[1]))
        qx, qy = Fraction(float(start[0])), Fraction(float(start[1]))
        total += (ex * qy - ey * qx) / 2
    return Fraction(scale) * Fraction(scale) * total


def cubic_closed_oracle(curve: CubicPHSplineClosed) -> Fraction:
    return chain_area_oracle(
        [segment.ctrl for segment in curve._segments], curve._scale
    )


def bspline_closed_oracle(curve) -> Fraction:
    spans = [
        np.column_stack((span.position.real, span.position.imag))
        for span in curve._state.spans
    ]
    return chain_area_oracle(spans, float(curve._state.scale))


def offset_area_oracle(handle, digits: int = 45) -> float:
    """High-precision line integral of the exact-reference parallel curve.

    Integrates ``(1/2) Im(conj(z_d) z_d')`` per source span with
    ``z_d = z + d_hat * i * w**2 / |w|**2`` built from the captured exact
    source state (position controls, PH preimages, widths), then applies
    the same cyclic residual join closure and the ``H**2`` scaling.  This
    is the test oracle of spec 14.3; it never enters production code.
    """
    import mpmath as mp

    provenance = handle._area_provenance
    spans_exact, H, d, _ = provenance.metric.exact_source_state()
    with mp.workdps(digits):
        d_hat = mp.mpf(d) / mp.mpf(H.numerator) * mp.mpf(H.denominator)

        def bezier(coeffs, t):
            n = len(coeffs) - 1
            return sum(
                mp.binomial(n, k) * (1 - t) ** (n - k) * t**k * coeffs[k]
                for k in range(n + 1)
            )

        total = mp.mpf(0)
        ends = []
        for controls, span in zip(provenance.position_spans, spans_exact):
            z_coeffs = [
                mp.mpc(float(row[0]), float(row[1])) for row in controls
            ]
            w_coeffs = [
                mp.mpc(
                    mp.mpf(re.numerator) / mp.mpf(re.denominator),
                    mp.mpf(im.numerator) / mp.mpf(im.denominator),
                )
                for re, im in zip(span.wre, span.wim)
            ]
            h = mp.mpf(span.h.numerator) / mp.mpf(span.h.denominator)
            m = span.m

            def z_d(t, z_coeffs=z_coeffs, w_coeffs=w_coeffs):
                w = bezier(w_coeffs, t)
                return bezier(z_coeffs, t) + d_hat * 1j * w * w / (
                    w.real * w.real + w.imag * w.imag
                )

            def integrand(t, z_coeffs=z_coeffs, w_coeffs=w_coeffs, h=h, m=m):
                w = bezier(w_coeffs, t)
                sigma = w.real * w.real + w.imag * w.imag
                if m == 0:
                    dw = mp.mpc(0)
                else:
                    dw = m * bezier(
                        [b - a for a, b in zip(w_coeffs, w_coeffs[1:])], t
                    )
                dz = h * w * w
                # Direct derivative of N_L: d/dt [i w^2 / sigma].
                dzd = dz + d_hat * 1j * (
                    (2 * w * dw * sigma - w * w * (2 * (w.conjugate() * dw).real))
                    / (sigma * sigma)
                )
                z = z_d(t, z_coeffs, w_coeffs)
                return (z.conjugate() * dzd).imag / 2

            total += mp.quad(integrand, [0, 1])
            ends.append(
                (
                    z_d(mp.mpf(0), z_coeffs, w_coeffs),
                    z_d(mp.mpf(1), z_coeffs, w_coeffs),
                )
            )
        count = len(ends)
        for index in range(count):
            end = ends[index][1]
            start = ends[(index + 1) % count][0]
            total += (end.conjugate() * start).imag / 2
        h_scale = mp.mpf(H.numerator) / mp.mpf(H.denominator)
        return float(h_scale * h_scale * total)


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------


def circle(n=24, radius=1.0, center=(0.0, 0.0), phase=0.0, cw=False):
    sign = -1.0 if cw else 1.0
    return [
        [
            center[0] + radius * math.cos(sign * (2 * math.pi * k / n + phase)),
            center[1] + radius * math.sin(sign * (2 * math.pi * k / n + phase)),
        ]
        for k in range(n)
    ]


def star_points(n=16, wave=0.35, lobes=4):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n
        r = 1.0 + wave * math.cos(lobes * t)
        out.append([r * math.cos(t), r * math.sin(t)])
    return out


def figure_eight_points(n=32, phase=0.05):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n + phase
        out.append([math.sin(t), math.sin(t) * math.cos(t)])
    return out


def limacon_points(n=48, phase=0.03):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n + phase
        r = 0.5 + math.cos(t)
        out.append([r * math.cos(t), r * math.sin(t)])
    return out


def octagon_points(scale=1.0):
    return [
        [scale * 1.2, 0.0],
        [scale * 0.9, scale * 0.9],
        [0.0, scale * 1.2],
        [-scale * 0.9, scale * 0.9],
        [-scale * 1.2, 0.0],
        [-scale * 0.9, -scale * 0.9],
        [0.0, -scale * 1.2],
        [scale * 0.9, -scale * 0.9],
    ]


def polygon_chain(vertices):
    """Closed polygon as a degree-1 span chain for the internal kernel."""
    count = len(vertices)
    return [
        np.array(
            [vertices[i], vertices[(i + 1) % count]], dtype=np.float64
        )
        for i in range(count)
    ]


def shoelace(vertices) -> Fraction:
    total = Fraction(0)
    count = len(vertices)
    for i in range(count):
        ax, ay = vertices[i]
        bx, by = vertices[(i + 1) % count]
        total += Fraction(float(ax)) * Fraction(float(by))
        total -= Fraction(float(ay)) * Fraction(float(bx))
    return total / 2


SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


@pytest.fixture()
def stats():
    area_module.reset_statistics()
    yield area_module.statistics
    area_module.reset_statistics()


# ---------------------------------------------------------------------------
# 14.2  Algebraic implementation checks
# ---------------------------------------------------------------------------


class TestCoefficientTables:
    def test_degree_one_is_shoelace(self):
        assert area_coefficients(1) == ((0, 1, Fraction(1, 2)),)

    def test_degree_two(self):
        table = {(a, b): k for a, b, k in area_coefficients(2)}
        assert table == {
            (0, 1): Fraction(1, 3),
            (0, 2): Fraction(1, 6),
            (1, 2): Fraction(1, 3),
        }

    def test_degree_three_matches_cubic_formula(self):
        table = {(a, b): k for a, b, k in area_coefficients(3)}
        assert table == {
            (0, 1): Fraction(3, 10),
            (2, 3): Fraction(3, 10),
            (0, 2): Fraction(3, 20),
            (1, 2): Fraction(3, 20),
            (1, 3): Fraction(3, 20),
            (0, 3): Fraction(1, 20),
        }

    @pytest.mark.parametrize("degree", list(range(1, 34)))
    def test_strict_positivity(self, degree):
        for a, b, coefficient in area_coefficients(degree):
            assert 0 <= a < b <= degree
            assert coefficient > 0

    @pytest.mark.parametrize("degree", list(range(1, 20)))
    def test_direct_formula_equivalence(self, degree):
        """(5.1) rearranged pairwise equals the (5.2) table exactly."""

        def w(i, j):
            if not 0 <= j <= degree - 1:
                return Fraction(0)
            return Fraction(
                comb(degree, i) * comb(degree - 1, j),
                comb(2 * degree - 1, i + j),
            )

        direct = {}
        for a in range(degree + 1):
            for b in range(a + 1, degree + 1):
                direct[(a, b)] = (
                    w(a, b - 1) - w(a, b) - w(b, a - 1) + w(b, a)
                ) / 4
        table = {(a, b): k for a, b, k in area_coefficients(degree)}
        assert direct == table

    @pytest.mark.parametrize("degree", [1, 2, 3, 5, 9, 17])
    def test_power_basis_integration_agreement(self, degree):
        """The table sum equals independent exact power integration."""
        rng = np.random.default_rng(degree)
        controls = rng.uniform(-2.0, 2.0, size=(degree + 1, 2))
        table_sum = sum(
            (
                k
                * (
                    Fraction(float(controls[a, 0]))
                    * Fraction(float(controls[b, 1]))
                    - Fraction(float(controls[a, 1]))
                    * Fraction(float(controls[b, 0]))
                )
                for a, b, k in area_coefficients(degree)
            ),
            Fraction(0),
        )
        assert table_sum == span_area_oracle(controls)

    @pytest.mark.parametrize("split", [Fraction(1, 2), Fraction(1, 3)])
    def test_subdivision_invariance(self, split):
        """Exact de Casteljau subdivision preserves the summed span area."""
        rng = np.random.default_rng(7)
        controls = rng.uniform(-1.5, 1.5, size=(4, 2))
        exact = [
            (Fraction(float(x)), Fraction(float(y))) for x, y in controls
        ]

        def de_casteljau_split(points, t):
            work = list(points)
            left, right = [work[0]], [work[-1]]
            while len(work) > 1:
                work = [
                    (
                        (1 - t) * p[0] + t * q[0],
                        (1 - t) * p[1] + t * q[1],
                    )
                    for p, q in zip(work, work[1:])
                ]
                left.append(work[0])
                right.append(work[-1])
            right.reverse()
            return left, right

        def table_area(points):
            return sum(
                (
                    k * (points[a][0] * points[b][1] - points[a][1] * points[b][0])
                    for a, b, k in area_coefficients(len(points) - 1)
                ),
                Fraction(0),
            )

        left, right = de_casteljau_split(exact, split)
        assert table_area(left) + table_area(right) == table_area(exact)


# ---------------------------------------------------------------------------
# 15.2  Exact elementary shapes and metamorphic laws (internal kernel)
# ---------------------------------------------------------------------------


class TestKernelElementary:
    @pytest.mark.parametrize(
        "vertices",
        [
            SQUARE,
            SQUARE[::-1],
            [[0.5, 0.25], [3.0, 1.0], [2.0, 4.5], [-1.5, 3.0], [-2.0, 0.5]],
        ],
    )
    def test_degree_one_recovers_shoelace(self, vertices):
        value = source_signed_area(polygon_chain(vertices), 1.0)
        assert value == float(shoelace(vertices))

    def test_cubic_loop_exact_rational_area(self):
        spans = [
            np.array(
                [[0.0, 0.0], [2.0, 0.5], [2.0, 2.5], [0.0, 2.0]], float
            ),
            np.array(
                [[0.0, 2.0], [-2.0, 1.5], [-2.0, 0.75], [0.0, 0.0]], float
            ),
        ]
        assert source_signed_area(spans, 1.0) == float(
            chain_area_oracle(spans, 1.0)
        )

    def test_translation_invariance(self):
        spans = polygon_chain(SQUARE)
        moved = [span + np.array([1.0e6, -3.0e5]) for span in spans]
        assert source_signed_area(moved, 1.0) == source_signed_area(spans, 1.0)

    def test_quarter_turn_rotation_invariance(self):
        spans = polygon_chain(
            [[0.25, 0.125], [2.0, 0.5], [1.5, 3.0], [-0.5, 1.0]]
        )
        rotated = [
            np.column_stack((-span[:, 1], span[:, 0])) for span in spans
        ]
        assert source_signed_area(rotated, 1.0) == source_signed_area(
            spans, 1.0
        )

    def test_scale_law(self):
        spans = polygon_chain(SQUARE)
        scaled = [4.0 * span for span in spans]
        assert source_signed_area(scaled, 1.0) == 16.0 * source_signed_area(
            spans, 1.0
        )
        assert source_signed_area(spans, 4.0) == 16.0 * source_signed_area(
            spans, 1.0
        )

    def test_reflection_negates(self):
        spans = polygon_chain(
            [[0.25, 0.125], [2.0, 0.5], [1.5, 3.0], [-0.5, 1.0]]
        )
        reflected = [
            np.column_stack((span[:, 0], -span[:, 1])) for span in spans
        ]
        assert source_signed_area(reflected, 1.0) == -source_signed_area(
            spans, 1.0
        )

    def test_reversal_negates(self):
        spans = polygon_chain(
            [[0.25, 0.125], [2.0, 0.5], [1.5, 3.0], [-0.5, 1.0]]
        )
        reversed_chain = [span[::-1].copy() for span in spans[::-1]]
        assert source_signed_area(reversed_chain, 1.0) == -source_signed_area(
            spans, 1.0
        )

    def test_cyclic_seam_invariance(self):
        spans = polygon_chain(
            [[0.5, 0.25], [3.0, 1.0], [2.0, 4.5], [-1.5, 3.0], [-2.0, 0.5]]
        )
        reference = source_signed_area(spans, 1.0)
        for shift in range(1, len(spans)):
            rotated = spans[shift:] + spans[:shift]
            assert source_signed_area(rotated, 1.0) == reference

    def test_join_closure_of_ulp_gapped_chain(self):
        """A few-ulp endpoint gap stays translation invariant (spec 4.4).

        The translation is chosen so every perturbed coordinate remains
        exactly representable after the shift; the stored chain is then
        translated without rounding and the connector convention must
        make the two stored areas bitwise equal.
        """
        spans = [np.array(span, float) for span in polygon_chain(SQUARE)]
        spans[1][0, 0] = math.nextafter(spans[1][0, 0], 2.0)
        spans[2][0, 1] = math.nextafter(spans[2][0, 1], 2.0)
        moved = [span + np.array([0.5, -0.75]) for span in spans]
        assert source_signed_area(spans, 1.0) == source_signed_area(moved, 1.0)
        assert source_signed_area(spans, 1.0) == float(
            chain_area_oracle(spans, 1.0)
        )


# ---------------------------------------------------------------------------
# 15.1  API and structural topology
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cubic_closed():
    return CubicPHSplineClosed(circle())


@pytest.fixture(scope="module")
def bspline_closed():
    return PHBSplineClosed(octagon_points())


class TestPublicAPI:

    def test_closed_types_expose_both_properties(
        self, cubic_closed, bspline_closed
    ):
        for curve in (cubic_closed, bspline_closed):
            assert isinstance(curve.signed_area, float)
            assert isinstance(curve.area, float)
            assert curve.area == abs(curve.signed_area)

    def test_open_types_have_no_area_members(self):
        open_cubic = CubicPHSplineOpen([[0.0, 0.0], [1.0, 0.4], [2.0, 1.3]])
        open_bspline = PHBSplineOpen(
            [[0.0, 0.0], [1.0, 0.4], [2.0, -0.7], [3.0, 1.1]]
        )
        for target in (
            open_cubic,
            open_bspline,
            open_bspline.snapshot(),
            open_cubic.offset(0.1),
            open_bspline.offset(0.1),
        ):
            assert not hasattr(target, "area")
            assert not hasattr(target, "signed_area")

    def test_closed_offsets_are_closed_handles(
        self, cubic_closed, bspline_closed
    ):
        for curve in (cubic_closed, bspline_closed):
            handle = curve.offset(0.0)
            assert isinstance(handle, ClosedNURBSHandle)
            assert isinstance(handle, NURBSHandle)
            assert isinstance(handle.signed_area, float)
            assert handle.area == abs(handle.signed_area)

    def test_open_offsets_are_base_handles(self):
        handle = CubicPHSplineOpen([[0.0, 0.0], [1.0, 0.4], [2.0, 1.3]]).offset(
            0.1
        )
        assert type(handle) is NURBSHandle

    def test_closed_snapshot_type_and_properties(self, bspline_closed):
        snapshot = bspline_closed.snapshot()
        assert isinstance(snapshot, PHBSplineClosedSnapshot)
        assert isinstance(snapshot, PHBSplineSnapshot)
        assert snapshot.signed_area == bspline_closed.signed_area
        assert snapshot.area == bspline_closed.area
        assert snapshot.version == bspline_closed.version

    def test_static_return_annotations(self):
        assert (
            "ClosedNURBSHandle"
            in CubicPHSplineClosed.offset.__annotations__["return"]
        )
        assert (
            "ClosedNURBSHandle"
            in PHBSplineClosed.offset.__annotations__["return"]
        )
        assert (
            "PHBSplineClosedSnapshot"
            in PHBSplineClosed.snapshot.__annotations__["return"]
        )

    def test_exact_zero_publishes_positive_zero(self):
        out_and_back = [
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([[1.0, 1.0], [0.0, 0.0]]),
        ]
        value = source_signed_area(out_and_back, 1.0)
        assert value == 0.0 and math.copysign(1.0, value) == 1.0

    def test_underflowed_zero_area_is_positive(self):
        clockwise = polygon_chain(SQUARE[::-1])
        signed = source_signed_area(clockwise, 1.0e-200)
        assert signed == 0.0 and math.copysign(1.0, signed) == -1.0
        assert abs(signed) == 0.0 and math.copysign(1.0, abs(signed)) == 1.0

    def test_orientation_signs(self, cubic_closed):
        assert cubic_closed.signed_area > 0.0
        clockwise = CubicPHSplineClosed(circle(cw=True))
        assert clockwise.signed_area < 0.0
        assert clockwise.area == abs(clockwise.signed_area)


# ---------------------------------------------------------------------------
# 15.3  Cubic PH closed cases
# ---------------------------------------------------------------------------


class TestCubicClosed:
    @pytest.mark.parametrize(
        "points",
        [
            circle(),
            circle(cw=True),
            circle(n=3, radius=2.0),
            circle(n=200),
            star_points(),
            star_points(n=24, wave=0.25, lobes=5),
            octagon_points(),
        ],
        ids=[
            "convex_ccw",
            "convex_cw",
            "minimum_points",
            "many_segments",
            "nonconvex_star4",
            "nonconvex_star5",
            "octagon",
        ],
    )
    def test_exact_oracle_equality(self, points):
        curve = CubicPHSplineClosed(points)
        assert curve.signed_area == float(cubic_closed_oracle(curve))
        assert curve.area == abs(curve.signed_area)

    def test_straight_and_curved_transitions(self):
        pts = []
        nline, narc = 5, 8
        for i in range(1, nline + 1):
            pts.append([-2 + 4 * i / (nline + 1), -1.0])
        for k in range(narc + 1):
            t = -math.pi / 2 + math.pi * k / narc
            pts.append([2 + math.cos(t), math.sin(t)])
        for i in range(1, nline + 1):
            pts.append([2 - 4 * i / (nline + 1), 1.0])
        for k in range(narc + 1):
            t = math.pi / 2 + math.pi * k / narc
            pts.append([-2 + math.cos(t), math.sin(t)])
        stadium = CubicPHSplineClosed(pts)
        assert any(seg.chi == 0.0 for seg in stadium._segments)
        assert stadium.signed_area == float(cubic_closed_oracle(stadium))
        assert stadium.signed_area == pytest.approx(8.0 + math.pi, rel=1e-3)

    def test_circle_magnitude_and_reversal(self):
        ccw = CubicPHSplineClosed(circle())
        cw = CubicPHSplineClosed(circle(cw=True))
        assert ccw.signed_area == pytest.approx(math.pi, rel=1e-4)
        assert cw.signed_area == pytest.approx(-math.pi, rel=1e-4)
        assert cw.area == pytest.approx(ccw.area, rel=1e-12)

    def test_seam_rotation_stability(self):
        base = circle(n=16)
        reference = CubicPHSplineClosed(base).signed_area
        for shift in (3, 7, 11):
            rotated = base[shift:] + base[:shift]
            value = CubicPHSplineClosed(rotated).signed_area
            assert value == pytest.approx(reference, rel=1e-12)

    def test_aux_inflection_general_path(self):
        curve = CubicPHSplineClosed(star_points(n=20, wave=0.4))
        assert curve.signed_area == float(cubic_closed_oracle(curve))

    def test_lazy_cache_and_warm_query(self, stats):
        curve = CubicPHSplineClosed(circle(n=8))
        assert stats["span_contributions"] == 0
        first = curve.signed_area
        cold_calls = stats["span_contributions"]
        assert cold_calls > 0
        assert curve.signed_area == first and curve.area == abs(first)
        assert stats["span_contributions"] == cold_calls

    def test_construction_calls_no_kernel(self, stats):
        CubicPHSplineClosed(circle(n=12))
        assert stats["span_contributions"] == 0
        assert stats["fast_accepted"] == 0
        assert stats["exact_fallback"] == 0


# ---------------------------------------------------------------------------
# 15.4  PH B-spline closed cases
# ---------------------------------------------------------------------------


class TestPHBSplineClosed:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"g_order": 3},
            {"g_order": 4},
            {"c_order": 2},
            {"curvature_order": 1},
            {
                "g_order": 3,
                "numerics": NumericalPolicy(max_preimage_degree=3),
            },
        ],
        ids=["default_g2", "g3", "g4", "c2", "curvature1", "configured_max"],
    )
    def test_exact_oracle_equality_across_orders(self, kwargs):
        curve = PHBSplineClosed(octagon_points(), **kwargs)
        assert curve.degree == 2 * curve.preimage_degree + 1
        assert curve.signed_area == float(bspline_closed_oracle(curve))
        assert curve.area == abs(curve.signed_area)

    @pytest.mark.parametrize(
        "parameterization", ["centripetal", "chord", "uniform"]
    )
    def test_parameterization_independence(self, parameterization):
        from ph_spline import ConstructionPolicy

        curve = PHBSplineClosed(
            octagon_points(),
            construction=ConstructionPolicy(parameterization=parameterization),
        )
        assert curve.signed_area == float(bspline_closed_oracle(curve))

    def test_cold_and_warm_queries(self, stats):
        curve = PHBSplineClosed(octagon_points())
        assert stats["span_contributions"] == 0
        value = curve.signed_area
        cold = stats["span_contributions"]
        assert cold == curve.num_spans
        assert curve.signed_area == value
        assert stats["span_contributions"] == cold

    @pytest.mark.parametrize("repair", ["strict_local", "expand", "global"])
    def test_edits_stay_exact(self, repair):
        curve = PHBSplineClosed(circle(n=12))
        curve.move_point(0, [1.05, 0.02], repair=repair)
        assert curve.signed_area == float(bspline_closed_oracle(curve))

    def test_edit_operations_and_versioned_cache(self):
        curve = PHBSplineClosed(circle(n=12))
        values = {curve.version: curve.signed_area}
        curve.move_point(3, [0.1, 1.05])
        values[curve.version] = curve.signed_area
        curve.insert_point(5, [-0.55, 0.87])
        values[curve.version] = curve.signed_area
        curve.delete_point(5)
        values[curve.version] = curve.signed_area
        with curve.edit(repair="expand") as edit:
            edit.move_point(1, [0.83, 0.55])
            edit.insert_point(7, [-0.95, -0.4])
        values[curve.version] = curve.signed_area
        assert len(values) == 5
        assert curve.signed_area == float(bspline_closed_oracle(curve))

    def test_seam_point_edit(self):
        curve = PHBSplineClosed(circle(n=12))
        before = curve.signed_area
        curve.move_point(0, [1.06, 0.015])
        after = curve.signed_area
        assert after != before
        assert after == float(bspline_closed_oracle(curve))

    def test_span_reuse_after_local_edit(self, stats):
        curve = PHBSplineClosed(circle(n=16))
        _ = curve.signed_area
        base_calls = stats["span_contributions"]
        report = curve.move_point(4, [math.cos(0.5) * 1.04, math.sin(0.5)])
        assert stats["span_contributions"] == base_calls
        _ = curve.signed_area
        recomputed = stats["span_contributions"] - base_calls
        assert recomputed == report.rebuilt_span_count
        assert stats["span_reused"] == curve.num_spans - recomputed

    def test_no_reuse_after_global_rebuild(self, stats):
        curve = PHBSplineClosed(circle(n=12))
        _ = curve.signed_area
        base_calls = stats["span_contributions"]
        base_reused = stats["span_reused"]
        curve.move_point(2, [0.52, 0.87], repair="global")
        _ = curve.signed_area
        assert stats["span_contributions"] - base_calls == curve.num_spans
        assert stats["span_reused"] == base_reused

    def test_failed_edit_retains_cache(self, stats):
        curve = PHBSplineClosed(octagon_points())
        value = curve.signed_area
        version = curve.version
        calls = stats["span_contributions"]
        with pytest.raises(Exception):
            curve.move_point(0, [1.2, float("nan")])
        assert curve.version == version
        assert curve.signed_area == value
        assert stats["span_contributions"] == calls

    def test_construction_and_commit_call_no_kernel(self, stats):
        curve = PHBSplineClosed(circle(n=12))
        curve.move_point(1, [math.cos(2 * math.pi / 12) * 1.03, 0.5])
        with curve.edit() as edit:
            edit.move_point(2, [0.52, 0.88])
        assert stats["span_contributions"] == 0

    def test_snapshot_isolation(self):
        curve = PHBSplineClosed(circle(n=12))
        before = curve.signed_area
        snapshot = curve.snapshot()
        curve.move_point(0, [1.08, 0.03])
        after = curve.signed_area
        assert snapshot.signed_area == before
        assert after != before
        late_snapshot = curve.snapshot()
        curve.move_point(0, [0.97, -0.02])
        assert late_snapshot.signed_area == after

    def test_snapshot_computes_lazily_without_source_pollution(self, stats):
        curve = PHBSplineClosed(octagon_points())
        snapshot = curve.snapshot()
        assert stats["span_contributions"] == 0
        value = snapshot.signed_area
        assert value == float(bspline_closed_oracle(curve))

    def test_concurrent_edit_and_query(self):
        curve = PHBSplineClosed(circle(n=16))
        errors = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    signed = curve.signed_area
                    assert isinstance(signed, float) and math.isfinite(signed)
            except Exception as exc:  # pragma: no cover - failure capture
                errors.append(exc)

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for step in range(12):
                angle = 2 * math.pi * (step % 16) / 16
                radius = 1.0 + 0.03 * ((-1) ** step)
                curve.move_point(
                    step % 16,
                    [radius * math.cos(angle), radius * math.sin(angle)],
                )
        finally:
            stop.set()
            thread.join()
        assert not errors
        assert curve.signed_area == float(bspline_closed_oracle(curve))


# ---------------------------------------------------------------------------
# 15.5  Offset cases
# ---------------------------------------------------------------------------


def offset_identity_check(handle, tolerance=5.0e-13):
    """Verify A_d = A_0 - d L_0 + pi nu d**2 against the mpmath oracle."""
    oracle = offset_area_oracle(handle)
    scale = max(1.0, abs(oracle))
    assert abs(handle.signed_area - oracle) <= tolerance * scale


@pytest.fixture(scope="module")
def cubic_source():
    return CubicPHSplineClosed(circle(n=16))


@pytest.fixture(scope="module")
def bspline_source():
    return PHBSplineClosed(octagon_points())


class TestOffsetArea:

    def test_zero_distance_bitwise(self, cubic_source, bspline_source):
        for source in (cubic_source, bspline_source):
            assert source.offset(0.0).signed_area == source.signed_area

    @pytest.mark.parametrize("distance", [0.2, -0.35])
    def test_cusp_free_offsets(self, cubic_source, bspline_source, distance):
        for source in (cubic_source, bspline_source):
            rho_left, rho_right = source.min_curvature_radii
            assert -rho_right < distance < rho_left
            handle = source.offset(distance)
            offset_identity_check(handle)

    def test_beyond_cusp_offsets(self):
        source = CubicPHSplineClosed(star_points(n=24, wave=0.25, lobes=5))
        rho_left, _ = source.min_curvature_radii
        handle = source.offset(1.75 * rho_left)
        assert handle.cusps
        offset_identity_check(handle)

    def test_reversed_loop_offset(self, cubic_source):
        handle = cubic_source.offset(1.6)
        offset_identity_check(handle)
        assert handle.signed_area == pytest.approx(
            math.pi * 0.36, rel=5.0e-3
        )

    def test_circle_sign_conventions(self):
        ccw = CubicPHSplineClosed(circle())
        cw = CubicPHSplineClosed(circle(cw=True))
        d = 0.25
        assert ccw.offset(d).signed_area == pytest.approx(
            math.pi * (1.0 - d) ** 2, rel=1e-3
        )
        assert cw.offset(d).signed_area == pytest.approx(
            -math.pi * (1.0 + d) ** 2, rel=1e-3
        )
        assert ccw.offset(-d).signed_area == pytest.approx(
            math.pi * (1.0 + d) ** 2, rel=1e-3
        )
        assert cw.offset(-d).signed_area == pytest.approx(
            -math.pi * (1.0 - d) ** 2, rel=1e-3
        )

    def test_turning_number_values(self):
        assert turning_number(
            CubicPHSplineClosed(circle(n=12))
            .offset(0.1)
            ._area_provenance.metric
        ) == 1
        assert turning_number(
            CubicPHSplineClosed(circle(n=12, cw=True))
            .offset(0.1)
            ._area_provenance.metric
        ) == -1

    def test_turning_number_zero_figure_eight(self):
        source = PHBSplineClosed(figure_eight_points())
        assert abs(source.signed_area) < 1.0e-15
        handle = source.offset(0.05)
        assert turning_number(handle._area_provenance.metric) == 0
        offset_identity_check(handle)

    def test_turning_number_two_limacon(self):
        source = PHBSplineClosed(limacon_points())
        handle = source.offset(0.02)
        assert turning_number(handle._area_provenance.metric) == 2
        offset_identity_check(handle)

    def test_straight_spans_zero_phase_cells(self):
        pts = []
        nline, narc = 5, 8
        for i in range(1, nline + 1):
            pts.append([-2 + 4 * i / (nline + 1), -1.0])
        for k in range(narc + 1):
            t = -math.pi / 2 + math.pi * k / narc
            pts.append([2 + math.cos(t), math.sin(t)])
        for i in range(1, nline + 1):
            pts.append([2 - 4 * i / (nline + 1), 1.0])
        for k in range(narc + 1):
            t = math.pi / 2 + math.pi * k / narc
            pts.append([-2 + math.cos(t), math.sin(t)])
        stadium = CubicPHSplineClosed(pts)
        handle = stadium.offset(0.15)
        metric = handle._area_provenance.metric
        spans, _, _, _ = metric.exact_source_state()
        cell_spans = {index for index, _, _ in metric.phase_cells()}
        constant = [i for i, s in enumerate(spans) if s.tau_zero]
        assert constant and not cell_spans.intersection(constant)
        assert turning_number(metric) == 1
        offset_identity_check(handle)

    def test_gauge_sign_flip_preserves_turning(self):
        from ph_spline.offset_metric import build_offset_metric

        source = PHBSplineClosed(octagon_points())
        handle = source.offset(0.1)
        state = handle._area_provenance.metric.state()
        flipped = dict(state)
        flipped["preimages"] = [
            [(-re, -im) for re, im in span] if index % 2 else list(span)
            for index, span in enumerate(state["preimages"])
        ]
        flipped_metric = build_offset_metric(
            span_preimages=[
                [complex(re, im) for re, im in span]
                for span in flipped["preimages"]
            ],
            span_widths=flipped["widths"],
            breakpoints=flipped["breakpoints"],
            distance=flipped["distance"],
            scale=flipped["scale"],
            closed=flipped["closed"],
        )
        assert turning_number(flipped_metric) == turning_number(
            handle._area_provenance.metric
        )

    def test_reversal_with_both_offset_signs(self):
        forward = PHBSplineClosed(octagon_points())
        backward = PHBSplineClosed(octagon_points()[::-1])
        for d in (0.15, -0.15):
            offset_identity_check(forward.offset(d))
            offset_identity_check(backward.offset(d))

    def test_handle_survives_source_edits(self):
        source = PHBSplineClosed(circle(n=12))
        handle = source.offset(0.1)
        cold = handle.signed_area
        source.move_point(0, [1.07, 0.02])
        assert handle.signed_area == cold
        fresh = source.offset(0.1)
        assert fresh.signed_area != cold

    def test_offset_construction_computes_no_area(self, stats):
        source = CubicPHSplineClosed(circle(n=12))
        area_module.reset_statistics()
        handle = source.offset(0.2)
        assert stats["span_contributions"] == 0
        assert stats["turning_max_precision"] == 0
        _ = handle.signed_area
        assert stats["turning_max_precision"] >= 256

    def test_incremental_distance_identity(self):
        """A(d2) - A(d1) = -(d2 - d1) L0 + pi nu (d2^2 - d1^2)."""
        source = CubicPHSplineClosed(circle(n=16))
        d1, d2 = 0.15, 0.4
        h1, h2 = source.offset(d1), source.offset(d2)
        spans, H, _, _ = h1._area_provenance.metric.exact_source_state()
        length = area_module._exact_source_length(spans, H)
        nu = turning_number(h1._area_provenance.metric)
        predicted = (
            -(d2 - d1) * float(length) + math.pi * nu * (d2 * d2 - d1 * d1)
        )
        assert h2.signed_area - h1.signed_area == pytest.approx(
            predicted, abs=1.0e-13
        )


# ---------------------------------------------------------------------------
# 15.6  Numerical stress
# ---------------------------------------------------------------------------


class TestNumericalStress:
    def test_far_translated_small_shape(self):
        near = CubicPHSplineClosed(circle(n=12))
        far = CubicPHSplineClosed(circle(n=12, center=(1.0e9, -1.0e9)))
        assert far.signed_area == float(cubic_closed_oracle(far))
        assert far.signed_area == pytest.approx(near.signed_area, rel=1e-6)

    def test_kernel_far_translation_exactness(self):
        spans = polygon_chain(SQUARE)
        far = [span + np.array([1.0e15, -1.0e15]) for span in spans]
        assert source_signed_area(far, 1.0) == float(
            chain_area_oracle(far, 1.0)
        )

    @pytest.mark.parametrize("radius", [1.0e-150, 1.0e-30, 1.0e30, 1.0e100])
    def test_scale_extremes(self, radius):
        curve = CubicPHSplineClosed(circle(n=12, radius=radius))
        assert curve.signed_area == float(cubic_closed_oracle(curve))
        assert curve.signed_area == pytest.approx(
            math.pi * radius * radius, rel=1e-3
        )

    def test_subnormal_exact_area(self):
        spans = polygon_chain(SQUARE)
        scale = 1.0e-160
        value = source_signed_area(spans, scale)
        assert value == float(Fraction(scale) * Fraction(scale))
        assert 0.0 < value < 2.3e-308

    def test_area_rounding_to_zero(self):
        value = source_signed_area(polygon_chain(SQUARE), 1.0e-200)
        assert value == 0.0 and math.copysign(1.0, value) == 1.0

    def test_nearly_cancelling_lobes(self, stats):
        side = 1.0 + 2.0**-30
        ccw = SQUARE
        cw = [
            [3.0, 0.0],
            [3.0, side],
            [3.0 + side, side],
            [3.0 + side, 0.0],
        ]
        spans = polygon_chain(ccw) + polygon_chain(cw)
        expected = shoelace(ccw) + shoelace(cw)
        assert abs(expected) < 1.0e-8
        assert source_signed_area(spans, 1.0) == float(expected)

    def test_offset_triple_cancellation(self):
        source = PHBSplineClosed(figure_eight_points())
        handle0 = source.offset(0.0)
        spans, H, _, _ = handle0._area_provenance.metric.exact_source_state()
        length = area_module._exact_source_length(spans, H)
        area_exact = chain_area_oracle(
            handle0._area_provenance.position_spans, float(H)
        )
        # nu = 0, so A_d = A_0 - d L_0; choose d to cancel almost exactly.
        d = float(area_exact / length)
        handle = source.offset(d)
        expected = area_exact - Fraction(d) * length
        assert handle.signed_area == float(expected)
        assert abs(handle.signed_area) < 1.0e-30

    def test_naive_overflow_avoided(self):
        spans = [1.0e-100 * span for span in polygon_chain(SQUARE)]
        scale = 1.0e180
        assert math.isinf(scale * scale)
        value = source_signed_area(spans, scale)
        assert value == float(chain_area_oracle(spans, scale))
        assert value == pytest.approx(1.0e160, rel=1e-12)

    def test_exact_overflow_raises(self):
        with pytest.raises(NumericalPrecisionError) as info:
            source_signed_area(polygon_chain(SQUARE), 1.0e200)
        assert info.value.operation == "area"

    def test_public_overflow_raises(self):
        curve = CubicPHSplineClosed(circle(n=12, radius=1.0e155))
        with pytest.raises(NumericalPrecisionError):
            _ = curve.area

    def test_rounding_boundary_cases(self):
        def bumped_square(depth):
            # A downward spike outside the unit square adds area
            # depth * 2**-21 to the enclosed region.
            return [
                [0.0, 0.0],
                [0.25, 0.0],
                [0.25 + 2.0**-21, -depth],
                [0.25 + 2.0**-20, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]

        tie_vertices = bumped_square(2.0**-32)
        above_vertices = bumped_square(2.0**-32 + 2.0**-84)
        assert shoelace(tie_vertices) == 1 + Fraction(1, 2**53)
        assert (
            shoelace(above_vertices)
            == 1 + Fraction(1, 2**53) + Fraction(1, 2**105)
        )
        # Exactly at the binary64 midpoint 1 + 2**-53 (ties to even -> 1.0)
        # and immediately above it (-> 1 + 2**-52).
        assert source_signed_area(polygon_chain(tie_vertices), 1.0) == 1.0
        assert (
            source_signed_area(polygon_chain(above_vertices), 1.0)
            == 1.0 + 2.0**-52
        )

    def test_fast_path_and_fallback_agree(self, stats):
        spans = polygon_chain(
            [[0.5, 0.25], [3.0, 1.0], [2.0, 4.5], [-1.5, 3.0], [-2.0, 0.5]]
        )
        fast = source_signed_area(spans, 1.0)
        assert stats["fast_accepted"] == 1
        exact = area_module._publish_float(
            Fraction(1) * area_module._normalized_area_exact(spans)
        )
        assert fast == exact

    def test_degree_table_cache_race(self):
        area_module._FRACTION_TABLES.pop(9, None)
        area_module._BALL_TABLES.pop(9, None)
        results = []

        def worker():
            results.append(area_coefficients(9))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(table == results[0] for table in results)

    def test_concurrent_first_queries(self):
        curve = CubicPHSplineClosed(circle(n=16))
        results = []
        barrier = threading.Barrier(6)

        def worker():
            barrier.wait()
            results.append(curve.signed_area)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(set(results)) == 1
        assert results[0] == float(cubic_closed_oracle(curve))


# ---------------------------------------------------------------------------
# 15.7  Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_cubic_round_trip_before_and_after_query(self, stats):
        curve = CubicPHSplineClosed(circle(n=12))
        cold_copy = pickle.loads(pickle.dumps(curve))
        assert stats["span_contributions"] == 0
        assert cold_copy._area_cache is None
        value = curve.signed_area
        warm_copy = pickle.loads(pickle.dumps(curve))
        assert warm_copy._area_cache is None
        assert warm_copy.signed_area == value
        assert cold_copy.signed_area == value

    def test_bspline_round_trip_before_and_after_query(self, stats):
        curve = PHBSplineClosed(octagon_points())
        cold_copy = pickle.loads(pickle.dumps(curve))
        assert stats["span_contributions"] == 0
        assert "_area_cache" not in cold_copy.__dict__
        value = curve.signed_area
        warm_copy = pickle.loads(pickle.dumps(curve))
        assert "_area_cache" not in warm_copy.__dict__
        assert warm_copy.signed_area == value
        assert cold_copy.signed_area == value

    def test_closed_handle_round_trip(self, stats):
        source = CubicPHSplineClosed(circle(n=12))
        handle = source.offset(0.2)
        cold_copy = pickle.loads(pickle.dumps(handle))
        assert stats["span_contributions"] == 0
        assert stats["turning_max_precision"] == 0
        assert isinstance(cold_copy, ClosedNURBSHandle)
        assert cold_copy._area_cache is None
        value = handle.signed_area
        warm_copy = pickle.loads(pickle.dumps(handle))
        assert warm_copy._area_cache is None
        assert warm_copy.signed_area == value
        assert np.array_equal(warm_copy.control_points, handle.control_points)

    def test_restored_provenance_arrays_read_only(self):
        handle = CubicPHSplineClosed(circle(n=8)).offset(0.1)
        restored = pickle.loads(pickle.dumps(handle))
        for array in restored._area_provenance.position_spans:
            assert not array.flags.writeable

    def test_corrupted_provenance_rejected(self):
        handle = CubicPHSplineClosed(circle(n=8)).offset(0.1)
        state = handle.__getstate__()
        truncated = dict(state)
        truncated["_area_position_spans"] = state["_area_position_spans"][:-1]
        target = object.__new__(ClosedNURBSHandle)
        with pytest.raises(OffsetConstructionError):
            target.__setstate__(truncated)
        missing = {
            key: value
            for key, value in state.items()
            if key != "_area_position_spans"
        }
        target = object.__new__(ClosedNURBSHandle)
        with pytest.raises(OffsetConstructionError):
            target.__setstate__(missing)
        bad_shape = dict(state)
        bad_shape["_area_position_spans"] = tuple(
            array[:, :1] for array in state["_area_position_spans"]
        )
        target = object.__new__(ClosedNURBSHandle)
        with pytest.raises(OffsetConstructionError):
            target.__setstate__(bad_shape)

    def test_snapshot_round_trip(self):
        curve = PHBSplineClosed(octagon_points())
        snapshot = curve.snapshot()
        value = snapshot.signed_area
        restored = pickle.loads(pickle.dumps(snapshot))
        assert isinstance(restored, PHBSplineClosedSnapshot)
        assert restored.signed_area == value

    def test_provenance_validation_direct(self):
        handle = CubicPHSplineClosed(circle(n=8)).offset(0.1)
        metric = handle._area_provenance.metric
        good = handle._area_provenance.position_spans
        with pytest.raises(OffsetConstructionError):
            OffsetAreaProvenance(position_spans=good[:-1], metric=metric)
        with pytest.raises(OffsetConstructionError):
            OffsetAreaProvenance(position_spans=good, metric=None)
        again = OffsetAreaProvenance(position_spans=good, metric=metric)
        assert offset_signed_area(again) == handle.signed_area
