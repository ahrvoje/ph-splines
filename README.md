# PH Splines

The provided Pythagorean-hodograph (PH) spline objects feature highly efficient, accurate and robust random-distance access through the `point_at_length` API.

- Cubic PH Spline (`CubicPHSplineOpen`, `CubicPHSplineClosed`) is highly efficient for static use cases.
- PH B-spline (`PHBSplineOpen`, `PHBSplineClosed`) supports dynamic editing (moving, adding and deleting nodes) with prescribed continuity-order constraints.

## 1. Cubic PH Spline

Point-interpolating planar **cubic Pythagorean-hodograph (PH) splines** with
verified geometry, exact arc length and fast distance-domain evaluation.

![A 20-metre spline route with exact distance stations and two points five metres of travel apart](examples/cubic/readme_distance_evaluation.png)

*Locate any position by distance travelled or select two points an exact path
distance apart.*

`CubicPHSplineOpen(points)` accepts convex, collinear and admissible nonconvex
point data directly, without requiring tangent or curvature data.
`CubicPHSplineClosed(points)` accepts cyclic data without a repeated final
point. Strictly convex cycles use a square cyclic solve and are G² at every
join. General cycles reuse the same auxiliary subsegment construction as the
open class: the seam remains G², convex runs remain G², and only genuine
curvature-sign changes and straight/curved transitions are G¹. Every segment,
declared continuity condition and numerical solve is independently verified.

The distinguishing feature is fast, accuracy-verified arc-length evaluation
for splines through arbitrary planar points: `point_at_length(d)` locates the
point reached after travelling `d` units, while queries at `s` and `s + d`
locate two points exactly `d` units of curve length apart. Each query uses one
O(log n) prefix lookup and a constant-cost closed-form inversion with bounded
polishing—without numerical quadrature or geometric search. Its arc-length
residual is verified near binary64 machine precision, and benchmarks remain
about 5-6 µs per query from 100 to 10,000 segments. To the best of our
knowledge, this is the first implementation for arbitrary planar-point splines
to combine closed-form arc-length inversion, near-machine-precision
verification and single-digit-microsecond random distance access.

```python
from ph_spline import CubicPHSplineOpen

curve = CubicPHSplineOpen([[0.0, 0.0], [1.0, 0.4], [2.0, 1.3], [2.6, 2.4]])

curve.point(0.5)               # float64 array (2,)
curve.tangent(0.5)             # unit tangent
curve.normal(0.5, side="left") # unit normal
curve.principal_normal(0.5)    # toward the center of curvature
curve.signed_curvature(0.5)    # float (sign = turning direction)
curve.curvature_vector(0.5)    # kappa * N_left
curve.aux_inflection_points    # [] (none inserted for this convex input)

L = curve.arc_length(1.0)      # exact (closed form, compensated prefix sums)
u = curve.parameter_at_length(0.5 * L)
curve.point_at_length(0.5 * L) # one locate + one inversion
```

```python
import numpy as np
from ph_spline import CubicPHSplineClosed

loop = CubicPHSplineClosed([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
assert loop.closed and np.array_equal(loop.point(0.0), loop.point(1.0))
```

### Gallery

| | |
|---|---|
| ![Open irregular coastline](examples/cubic/nonconvex/20_coastline.png) | ![Closed turbine blade section](examples/cubic_closed/002_turbine_blade_section.png) |
| *An open, random-walk coastline with tight coves, broad headlands and many inflections.* | *A cambered axial-turbine section with a finite trailing-edge radius.* |
| ![Tokamak flux surface](examples/cubic_closed/003_tokamak_flux_surface.png) | ![Open exhaust-header centerline](examples/cubic/nonconvex/58_exhaust_header.png) |
| *A D-shaped magnetic-confinement surface with curvature vectors.* | *An open, asymmetric exhaust runner with alternating bends and distance stations.* |
| ![Noncircular gear](examples/cubic_closed/005_noncircular_gear.png) | ![Open overhand trace](examples/cubic/pathological/36_overhand_scribble.png) |
| *An elliptic pitch profile carrying fourteen smoothly resolved teeth.* | *An irregular open trace with self-crossings, tight reversals and broad sweeping arcs.* |

### Where distance-domain evaluation matters

Fast, verified distance queries matter wherever work must be scheduled or
distributed uniformly along a path: contour design, CNC and robot motion,
autonomous navigation, constant-speed animation, spatial and GIS queries,
collision sampling, physical simulation, sensing, inspection, and placement
of textures, annotations or material along a curve.

### Background

Write a planar polynomial curve as `z(t) = x(t) + iy(t)`, with velocity
`z'(t)`. Its speed `sqrt(x'(t)^2 + y'(t)^2)` is generally not polynomial. A PH
curve instead makes `|z'(t)|^2 = x'(t)^2 + y'(t)^2 = sigma(t)^2`, so speed and
its arc-length integral are polynomials.

This implementation joins planar cubic PH Bézier segments with complex form
`z'(t) = (a + bt)^2`. Their speed `|a + bt|^2` is quadratic, so arc length is
cubic; distance inversion is one monotone cubic solve, expressible with cube
roots or hyperbolic functions. Bounded Newton iterations only correct rounding.

### Numerical design highlights

- All construction in normalized coordinates (origin `P0`, scale = longest
  chord); hypot-based norms; chord-ratio representability guard.
- Closed-form, bitwise-deterministic nonconvex preprocessing. Auxiliary points
  are clamped to chord fractions `[1/16, 15/16]`; their prescribed tangents are
  never boundary-clamped, and every fallback retains a strictly positive tilt.
- Bounded trust-region tridiagonal solve for the internal tangent angles,
  with machine-precision complex-step Jacobians (three-coloring), a
  deterministic fallback initializer for extreme chord ratios, and a damped,
  box-projected Newton polish. Solver output is never trusted: strict
  independent acceptance (residual gate `1e-11`, regularity, control-polygon
  admissibility, PH reconstruction with conditioning-aware tolerance, and
  per-joint G²/G¹ verification) follows every solve.
- Cancellation-resistant closed forms throughout: `sinc`-based angle ratios,
  rationalized quadratic roots chosen per branch sign, completed-square arc
  length, scaled hyperbolic/Cardano depressed-cubic inversion with
  factored recovery of `t` (never `y - h`), end-reversed inversion near
  `t = 1`, safeguarded Newton with a fixed iteration bound and an explicit
  arc-length residual gate `64 eps L + 4 ulp(s)`.
- Works verified from coordinate magnitudes `1e-150` to `1e307`, curvatures
  `1e-12` to `1e12`, chord ratios down to ~`5e-13`, and systems beyond 1000
  segments (sparse banded path).

### API overview

`CubicPHSplineOpen(points)` and `CubicPHSplineClosed(points)` are the
immutable concrete classes; `CubicPHSpline` is their abstract family base.
Both expose point, frame, curvature and exact distance-domain queries, including
`point_at_length`. The closed class lists its seam point once and verifies the
cyclic seam independently. `aux_inflection_points` reports any inserted
curvature-sign transitions.

All package exceptions derive from `PHSplineError`.
`CubicPHSplineError` and `PHBSplineError` are sibling family roots; input
failures are also `ValueError` instances and numerical failures are also
`RuntimeError` instances.

### Benchmarks

`python benchmarks/benchmark.py` compares a strictly convex log spiral with a
one-inflection cubic S curve, best of three on Windows 11 / Python 3.14.6 /
NumPy 2.x. Both paths include their complete construction and verification
contracts. Queries use seeded uniform arc lengths.

| kind | points | segments | aux | construction | 1000 × `point_at_length` | per query |
|:-----|-------:|---------:|----:|-------------:|-------------------------:|----------:|
| convex | 100 | 99 | 0 | 0.022 s | 5.8 ms | 5.8 µs |
| nonconvex | 100 | 100 | 1 | 0.022 s | 5.7 ms | 5.7 µs |
| convex | 1 000 | 999 | 0 | 0.029 s | 5.7 ms | 5.7 µs |
| nonconvex | 1 000 | 1 000 | 1 | 0.047 s | 5.7 ms | 5.7 µs |
| convex | 10 000 | 9 999 | 0 | 0.373 s | 6.1 ms | 6.1 µs |
| nonconvex | 10 000 | 10 000 | 1 | 0.337 s | 5.8 ms | 5.8 µs |

Construction stays in the same range for the two contracts. Query cost is
nearly flat over the tested sizes: one O(log n) prefix search plus one
constant-cost elementary local inversion, never an iterative geometric search.

## 2. PH B-spline

The PH B-spline family is the editable, variable-order counterpart.
`PHBSplineOpen` and `PHBSplineClosed` accurately interpolate their respective
planar point topologies, verify the requested continuity and
regularity before publication, and retain analytic PH speed and arc-length
polynomials on every span. The default request is G²; `g_order`, `c_order` and
`curvature_order` select higher geometric, parametric or arc-length-curvature
continuity. The minimum sufficient preimage order is selected directly as
`max(2, g_order, c_order, curvature_order + 2)`, giving PH degree
`2 * order + 1`; a simple midpoint knot supplies shape freedom without
inflating that degree.

```python
import numpy as np
from ph_spline import PHBSplineOpen

curve = PHBSplineOpen(
    [[0.0, 0.0], [1.0, 0.4], [2.0, -0.7], [3.0, 1.1], [2.2, 2.0]],
    g_order=4,
)

# Analytic distance access is retained at variable degree.
midpoint = curve.point_at_length(0.5 * curve.length)

# Derivatives are full vectors of arbitrary configured order.
velocity = curve.derivative(0.4)
intrinsic_jet = curve.jet(0.4, 6, wrt="arc_length")
curvature_jet = curve.curvature_vector_jet(0.4, 4)

# Handles survive insertion and index shifts; edits commit atomically.
handle = curve.point_handle(2)
report = curve.move_point(handle, [2.1, -0.9])
inserted = curve.insert_point(3, [2.6, 0.1])
curve.delete_point(inserted.handle)

# Several edits can share one verified commit.
with curve.edit() as edit:
    edit.move_point(handle, [2.0, -0.8])
    edit.insert_point(3, [2.5, 0.2])

snapshot = curve.snapshot(compact=True)  # immutable query-only view
stations = snapshot.points_at_length(
    np.linspace(0.0, snapshot.length, 1000), assume_sorted=True
)
```

### Editing and higher-order geometry

Move, insertion and deletion rebuild a bounded coefficient neighborhood and
structurally share unchanged span arrays. Stable `PointHandle` values are not
renumbered by insertion; deleted handles and pre-edit `CurveLocation` values
raise typed stale-reference errors. `repair="strict_local"` is the default and
never silently becomes a global rebuild. `repair="expand"` admits a larger
configured patch, while `repair="global"` makes full reconstruction explicit.
Failed edits leave points, handles, spans, version and cached metric data
unchanged.

`derivative(u, order, wrt=...)` evaluates parameter or intrinsic arc-length
derivatives. `curvature_vector(u, order)` returns derivatives of the complete
curvature vector—not only derivatives of scalar curvature—and
`curvature_vector_jet` shares one Taylor-series recurrence. The elementary
Bernstein/preimage product identities are used directly; finite-difference
geometry and numerical quadrature are absent from these kernels.

### API overview

`PHBSplineOpen(points, ...)` and `PHBSplineClosed(points, ...)` are the
mutable concrete classes; `PHBSpline` is their abstract family base. Optional
continuity keywords request G, C or arc-length-curvature continuity, with G² as
the default.

The family provides scalar and batch geometry, arbitrary-order parameter and
arc-length derivative jets, curvature-vector jets, analytic distance queries,
stable point handles, atomic move/insert/delete operations, edit transactions
and immutable snapshots. Construction and edits publish state only after
interpolation, continuity, regularity and metric verification succeed.

### PH B-spline gallery

The isolated [`examples/bspline`](examples/bspline) directory contains
generators and rendered output for all 224 referenced input cases, plus 128 PH
B-spline-specific cases in eight feature families.

| | |
|---|---|
| ![Closed G8 radial zigzag star](examples/bspline/pathological/14_star_zigzag_radial.png) | ![Closed distance stations](examples/bspline/features/02_closed_distance_stations/005_closed_distance_stations.png) |
| *A closed, symmetric radial zigzag star built from degree-17 G8 PH spans.* | *A closed curve with equal arc-length stations.* |
| ![Local move example](examples/bspline/features/03_local_move/009_local_move.png) | ![G8 centripetal-force example](examples/bspline/features/07_arc_derivative_jets/010_arc_derivative_jets.png) |
| *A handle-based G2 move recompiles ten local spans.* | *The G8 second arc-length derivative: the centripetal-force vector.* |

### Numerical design

- Construction uses a translated and scaled complex frame, a degree exactly
  deduced from the continuity request, one simple midpoint knot per input
  interval, analytic PH displacement constraints and an independent
  post-build interpolation/continuity check.
- Each immutable span stores one authoritative Bernstein preimage. Position,
  speed and forward/reverse arc coefficients follow from finite Bernstein
  products and antiderivatives.
- Regularity is certified by recursively bounding the distance from the
  origin to Bernstein control boxes; sampling is used only for branch ranking.
- High-order join evaluation retains canonical endpoint preimage jets, which
  avoids catastrophic cancellation in alternating Bernstein differences.
- Arc inversion starts from a monotone per-span lookup bracket and applies
  bounded safeguarded Newton correction. Only stable Bernstein evaluation can
  satisfy the final near-binary64 forward residual gate.
- Global construction or an edit either returns a finite, verified curve or a
  structured `PHBSplineError`; continuity and degree are never downgraded.

### Benchmarks: size and shape

`python benchmarks/benchmark_bspline.py` prints all three tables below. These
construction figures are best-of-two below 10,000 nodes and single-run at
10,000 nodes; distance-query figures are best-of-three measurements on a
13th Gen Intel Core i9-13900K running Windows 11, Python 3.14.6, NumPy 2.5.1
and SciPy 1.18.0.
Each random distance result includes lookup, analytic polynomial inversion,
forward residual acceptance and point evaluation.

| kind | nodes | spans | construction | `point_at_length` |
|:--|--:|--:|--:|--:|
| convex | 100 | 198 | 0.0405 s | 56.96 µs |
| nonconvex | 100 | 198 | 0.0399 s | 55.61 µs |
| convex | 1,000 | 1,998 | 0.3981 s | 45.69 µs |
| nonconvex | 1,000 | 1,998 | 0.3609 s | 46.06 µs |
| convex | 10,000 | 19,998 | 3.6540 s | 47.44 µs |
| nonconvex | 10,000 | 19,998 | 3.5769 s | 46.10 µs |

Random distance access remains nearly flat as span count grows; its higher
constant than `CubicPHSplineOpen` buys variable degree, generic regular geometry
and editable state.

### Benchmarks: higher continuity

The second sweep fixes 1,000 nodes and raises the verified continuity request.
Both construction and distance-query cost rise with polynomial degree, rather
than with a hidden iterative quadrature tolerance.

| continuity | PH degree | construction | `point_at_length` |
|:--|--:|--:|--:|
| G3 | 7 | 0.4747 s | 58.23 µs |
| G4 | 9 | 0.6006 s | 65.11 µs |
| G6 | 13 | 0.9142 s | 118.02 µs |
| G8 | 17 | 1.2785 s | 145.36 µs |

### Benchmarks: dynamic editing

These are median end-to-end latencies from seven warmed calls to the public
editing API. They include validation, candidate construction, independent
verification and atomic commit. Exterior span polynomials, arc-length tables
and inverse kernels remain bitwise shared. With the current one-midpoint basis,
a single-node strict-local Gρ edit rebuilds `2(ρ + 3)` compiled spans:
10/14/22 for G2/G4/G8. The current flat-array commit path still
performs O(n) validation, prefix rebuilding and publication, so the table
reports actual user-visible latency rather than presenting bounded numerical
recompilation as total-size-independent execution.

| nodes | continuity | median move | median insert | median delete | rebuilt spans |
|--:|:--|--:|--:|--:|:--|
| 100 | G2 | 3.70 ms | 3.88 ms | 3.71 ms | 10/10/10 |
| 100 | G4 | 5.90 ms | 6.54 ms | 6.50 ms | 14/14/14 |
| 100 | G8 | 17.02 ms | 17.29 ms | 17.31 ms | 22/22/22 |
| 1,000 | G2 | 8.83 ms | 11.95 ms | 11.34 ms | 10/10/10 |
| 1,000 | G4 | 11.11 ms | 13.77 ms | 13.01 ms | 14/14/14 |
| 1,000 | G8 | 19.35 ms | 23.04 ms | 22.16 ms | 22/22/22 |
| 10,000 | G2 | 59.50 ms | 92.92 ms | 91.28 ms | 10/10/10 |
| 10,000 | G4 | 60.69 ms | 94.32 ms | 93.19 ms | 14/14/14 |
| 10,000 | G8 | 69.95 ms | 103.73 ms | 101.58 ms | 22/22/22 |

## References

- Jaklič, Kozak, Krajnc, Vitrih and Žagar, [*On interpolation by planar cubic
  G² Pythagorean-hodograph spline curves*](https://users.fmf.uni-lj.si/knez/clanki/CubicPHG2Spline-rev.pdf).
- Albrecht, Beccari, Canonne and Romani, [*Pythagorean-Hodograph B-Spline
  Curves* (PDF)](https://arxiv.org/pdf/1609.07888) and
  [abstract/metadata](https://arxiv.org/abs/1609.07888).
- Farouki, [*Arc lengths of rational Pythagorean-hodograph
  curves*](https://escholarship.org/content/qt90s84043/qt90s84043.pdf).
- Farouki and Sakkalis, [*Real rational curves are not ‘unit
  speed’*](https://doi.org/10.1016/0167-8396(91)90040-I).
- Knez, Pelosi and Sampoli, [*Construction of G² planar Hermite interpolants
  with prescribed arc lengths*](https://arxiv.org/pdf/2202.11371).
- Gajny, Béarée, Nyiri and Gibaru, [*Path planning with PH G² splines in
  R²*](https://doi.org/10.1109/IConSCS.2012.6502455).
