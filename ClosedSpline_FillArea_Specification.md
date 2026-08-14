# Fill Area of Closed PH Splines and Their Exact Offset NURBS

**Status:** normative implementation specification and implementation handoff<br>
**Applies to:** `CubicPHSplineClosed`, `PHBSplineClosed`, closed PH B-spline
snapshots, and the exact closed offset `ClosedNURBSHandle`<br>
**Extends:** `ClosedSpline_Area_Specification.md`; the algebraic
`signed_area`/`area` interface, its exact-reference model, its arithmetic
layers, and its lazy-cache contracts all remain unchanged and are
prerequisites of this document<br>
**Technical confidence:** high for transversal self-intersection
configurations; degenerate tangential contacts terminate in the documented
typed failure, never in an unverified value<br>
**Does not specify:** open curves, even-odd fill, union of several curves,
offset trimming, or Boolean region operations

The words **SHALL**, **SHALL NOT**, **SHOULD**, and **MAY** are normative.

# 1. Executive decisions

1. `fill_area` is the *nonzero-winding fill area*: the Lebesgue measure of
   the set of points about which the closed curve winds a nonzero number of
   times,

   $$
   F=\iint_{\mathbb R^2}\big[\operatorname{wind}(r,q)\neq 0\big]\,dq.
   $$

   This is the region an ordinary nonzero-winding rasterizer would paint.
   It is the "physical" enclosed area of a self-intersecting curve.
2. `fill_area` is a nonnegative scalar in squared user-coordinate units.
   It is invariant under traversal reversal, cyclic seam motion,
   translation, rotation, and reflection, and scales quadratically.
3. Only the closed area-capable types of the area specification expose
   `fill_area`: `CubicPHSplineClosed`, `PHBSplineClosed`, the closed
   PH B-spline snapshot, and `ClosedNURBSHandle`. Open types SHALL NOT
   expose the member, and no error-raising placeholder is permitted.
4. For a curve whose crossing certification proves *no self-intersection*,
   `fill_area` SHALL return bitwise `abs(signed_area)` — the already
   correctly rounded `area` value. No decomposition work runs.
5. For a self-intersecting curve, `fill_area` is computed by certified
   decomposition of the exact-reference locus at its certified transversal
   self-intersections into simple sub-loops, the laminar containment
   forest of those loops, and winding-gated face-area summation
   (sections 5-8).
6. Calculation is lazy and cached exactly like `signed_area`:
   construction, edit commit, snapshot creation, offset construction,
   copying, and deserialization SHALL NOT run any fill computation; the
   first query per committed state computes and caches; repeated queries
   are O(1).
7. Fill areas of exact offsets SHALL be computed from the captured
   exact-reference source state (area provenance and offset metric),
   never from the rounded public NURBS arrays, and never by sampling,
   polygonization, or an external clipping library.
8. There is no universal order between `fill_area` and `area`: a
   figure-eight has `fill_area > area` (lobes cancel algebraically) and a
   limacon with an inner loop has `fill_area < area` (the doubly wound
   core counts twice algebraically, once physically). Implementations
   SHALL NOT assume either inequality.
9. A finite value SHALL be published only when a certified enclosure of
   the exact fill area determines one binary64 rounding. Unresolvable
   degenerate contact (for example an exact tangential self-osculation)
   SHALL raise the existing `NumericalPrecisionError` with
   `operation="fill-area"`; NaN, infinity, and uncertified midpoints are
   forbidden.

# 2. Exact scope and API

## 2.1 Required properties

```python
class CubicPHSplineClosed(CubicPHSpline):
    @property
    def fill_area(self) -> float: ...

class PHBSplineClosed(PHBSpline):
    @property
    def fill_area(self) -> float: ...

class PHBSplineClosedSnapshot(PHBSplineSnapshot):
    @property
    def fill_area(self) -> float: ...

class ClosedNURBSHandle(NURBSHandle):
    @property
    def fill_area(self) -> float: ...
```

Open sources, open snapshots, the family bases, and the base
`NURBSHandle` SHALL NOT expose `fill_area`. Runtime conformance mirrors
the area specification's `hasattr` battery.

## 2.2 Relationship to the algebraic area

`fill_area` MAY read `signed_area`/`area` (and thereby trigger and share
their caches). When the simplicity certificate of section 5 holds, the
published value SHALL be bitwise `abs(signed_area)`. The two caches are
otherwise independent: computing one SHALL NOT compute the other.

## 2.3 Authoritative data

The authoritative inputs are exactly those of the area specification: the
committed normalized Bernstein position controls, the exact PH preimages,
span widths, the normalization scale $H$, and for offsets the accepted
signed distance $d$ and the verified offset-metric certificate (cells,
constant-sign factors, exact preimage restrictions). A `d == 0` offset
handle SHALL be treated as its captured polynomial source.

# 3. Mathematical definitions

## 3.1 Winding decomposition

Let $r$ traverse the closed piecewise-regular exact-reference locus once.
For a point $q$ off the locus, $\operatorname{wind}(r,q)\in\mathbb Z$.
The fill area is decision 1's integral. If the locus has only finitely
many transversal self-intersections $\{X_1,\dots,X_k\}$, splitting the
parameter circle at the crossing parameters and rejoining arcs at each
crossing in traversal-preserving order (Seifert smoothing) yields simple
closed sub-loops $C_1,\dots,C_{k+1}$ whose interiors pairwise do not
cross: any two are disjoint or nested (their boundaries meet at most in
crossing points). The interiors therefore form a **laminar forest** under
containment, and for every point $q$ off the locus

$$
\operatorname{wind}(r,q)=\sum_{i:\,q\in\operatorname{int}(C_i)}s_i,
\qquad s_i=\operatorname{sign}(A(C_i)),
$$

with $A(C_i)$ the algebraic sub-loop area. Consequently, with
$\mathcal{ch}(i)$ the forest children of loop $i$ and
$W_i=\sum_{j\in\text{ancestors}(i)\cup\{i\}}s_j$ the constant winding on
the face $\operatorname{int}(C_i)\setminus\bigcup_{c}\operatorname{int}(C_c)$,

$$
\boxed{
F=\sum_{i\,:\,W_i\neq 0}
\Big(|A(C_i)|-\sum_{c\in\mathcal{ch}(i)}|A(C_c)|\Big).
}
\tag{3.1}
$$

This is the normative computation. A loop with exactly zero algebraic
area is degenerate input and SHALL raise the typed failure.

## 3.2 Sub-arc line integrals

Sub-loop areas are sums of Green line integrals over parameter arcs plus
the same residual connector convention as the area specification. For a
polynomial arc on one span, restricted to local $[a,b]$:

$$
\int_a^b [\,\hat r,\hat r'\,]\,dt = F(b)-F(a),
$$

where $F$ is the exact rational power-basis antiderivative of
$x y' - y x'$ built from the stored position controls.

For an exact PH offset arc $\hat z_d=\hat z+\hat d\,N$ with
$N=i\,w^2/|w|^2$, $\hat z'=h\,w^2$, and $\hat\sigma=h|w|^2$, the exact
identities $[N,\hat z']=-\hat\sigma$,
$[\hat z,N']=([\hat z,N])'-\hat\sigma$, and $[N,N']=2\varphi'$ (with
$\varphi$ a continuous argument of $w$) give the boxed sub-arc formula

$$
\boxed{
\int_a^b[\hat z_d,\hat z_d']\,dt
=\big(F(b)-F(a)\big)
+\hat d\Big([\hat z,N]\Big|_a^b-2\,\Delta\hat s\Big)
+2\hat d^{\,2}\,\Delta\varphi,
}
\tag{3.2}
$$

where $\Delta\hat s=h\,(R(b)-R(a))$ from the exact antiderivative $R$ of
$|w|^2$, and $\Delta\varphi$ is the continuous preimage-phase increment
assembled from the verified metric phase cells exactly as in the
turning-number procedure of the area specification (partial cells at the
arc ends use the same principal-`atan2` enclosures). Summed over a full
loop, (3.2) reproduces $A_d=A_0-dL_0+\pi\nu d^2$. Sub-loop areas in user
coordinates carry the $H^2$ factor; the origin never occurs.

Decomposed loops close at crossing points; the connector chord terms
$\tfrac12[\,p_{\text{end}},p_{\text{start}}\,]$ between consecutive arc
endpoint enclosures SHALL be included, exactly as in the area
specification's join convention.

# 4. Monotone pieces

The engine SHALL first partition the locus into **certified monotone
pieces**. A piece is a span (for offsets: metric cell) subinterval with
dyadic bounds together with an exact rational direction $D$ and the
certificate that every Bernstein coefficient of the restricted tangent
generator $\eta\,w^2$ satisfies $g\cdot D>0$ (for offsets additionally:
the restricted $|w|^2$ coefficients are strictly positive, so rational
position hulls are well defined). Here $\eta$ is the cell's certified
constant sign of the cusp polynomial $G$ for offsets and $+1$ for
sources; since the offset tangent is $G\cdot w^2$-directed, the
certificate makes the piece a graph over $D$ and therefore **injective**.
Subdivision at dyadic midpoints SHALL continue until every piece is
certified; a depth cap raises the typed failure.

Two immediate consequences the implementation SHALL exploit:

- a piece never self-intersects, so same-piece pairs are skipped; and
- the convex hull property bounds every piece tangent and every piece
  chord inside the convex cone spanned by its $\eta w^2$ coefficients.

# 5. Certified crossing enumeration

All unordered piece pairs are processed. The engine SHALL be complete:
every pair region is either excluded, contains exactly one certified
transversal crossing, is a certified-injective corner block at a shared
joint, or is absorbed as a micro block under section 5.4. Anything else
escalates subdivision up to a hard depth cap and then raises the typed
failure.

## 5.1 Exclusion and cones

Pairs prune when outward-rounded position boxes are disjoint (rational
hulls for exactness; a conservative float pre-filter MAY run first).
For surviving pairs, the **cone-separation certificate** requires every
pairwise determinant $\det(g_P,g_Q)$ of tangent generators to carry one
strict sign. Because piece chords lie in their tangent cones, separated
cones imply *at most one* intersection of the two arcs (two crossings
would produce a common chord direction in both cones) and imply that any
crossing is transversal. Pairs whose cones cannot be separated after
subdivision fall through to 5.4.

## 5.2 Existence and localization

With separated cones, choose $A=D_Q^\perp$ and $B=D_P^\perp$; then
$A\cdot P(s)$ is strictly monotone in $s$ and $B\cdot Q(t)$ in $t$ on the
block. The separable Miranda test certifies existence: $A\cdot(P-Q)$ has
uniform opposite strict signs on the two $s$-edges and $B\cdot(P-Q)$ on
the two $t$-edges, each decided by exact univariate Bernstein hull
comparisons (rational hull division for offset loci, whose restricted
weights are positive). Existence plus at-most-one yields exactly one
transversal crossing in the block.

Localization SHALL refine by exact dyadic bisection that preserves the
Miranda certificate, with deterministic jittered split fractions when a
crossing lies on a trial split line; an exact dyadic hit is stored as an
exact crossing parameter. Refinement SHALL reach parameter enclosures
tight enough for the acceptance gate of section 8 (the reference target
is $2^{-90}$ or better).

## 5.3 Shared joints and the seam

Adjacent pieces (including the cyclic seam pair) share one endpoint that
is not a crossing. The engine SHALL shrink corner blocks toward the
shared joint until either the two blocks' tangent generators admit one
common strict half-plane (the combined arc is then a graph, hence
injective through the joint) or the corner block qualifies as a micro
block (5.4); the remaining L-shaped region is processed as ordinary far
rectangles. Offset joints at certified cusps, where the tangent reverses,
are expected to terminate through the micro-block branch.

## 5.4 Micro blocks

A pair block whose two position enclosures fit in one box of diameter
$\delta$ MAY be **absorbed**: there exists a curve inside the
$\delta$-tube, identical outside the box, with no crossing in the box,
and the fill areas of the two curves differ by at most the box area. The
engine SHALL add $H^2\delta^2$ (outward rounded) to the fill enclosure
slack and otherwise ignore the block. The summed slack budget SHALL be
tracked and SHALL enter the acceptance gate; absorption is valid
regardless of how many unresolved contacts the box hides.

## 5.5 Deduplication

Certified crossings whose parameter enclosures overlap on the parameter
circle (a crossing found on both sides of a block boundary) SHALL merge.
A configuration in which three or more arcs pass through one point fails
cone separation and terminates through 5.4 or the typed failure.

# 6. Decomposition and loop areas

Crossing parameters split the parameter circle into arcs. Following the
traversal and swapping to the partner parameter at every crossing yields
the sub-loops of section 3.1. Each loop's algebraic area is evaluated as
a rational interval: exact antiderivative differences (3.2) at the
crossing-parameter enclosures, phase-cell `atan2` balls for offset arcs,
connector chords between consecutive endpoint enclosures, and the final
$H^2$ scaling. Loop orientation is the certified sign of the loop-area
enclosure; an enclosure straddling zero forces refinement and ultimately
the typed failure.

# 7. Containment forest and winding

For each loop pair, containment is decided by the winding number of one
exact rational test point of the inner candidate (taken at a dyadic
parameter away from all crossing enclosures) with respect to the outer
candidate. The winding is computed by certified horizontal ray casting:
per arc, exact univariate root isolation of $y(t)-p_y$ (offsets:
$Y - p_y W$), sign-certified $x$-comparison and crossing direction at
each root, with roots refined until every sign decides. A ray that
passes through a crossing enclosure or an arc endpoint enclosure SHALL
be discarded and a different deterministic test point chosen. The parent
of a loop is its innermost container; face windings and the fill sum
follow (3.1).

# 8. Arithmetic and acceptance

All quantities are exact rationals or rational intervals; transcendental
phase terms use the existing `atan2_ball` enclosures on the precision
ladder of the area specification. The final fill enclosure is the
interval sum of face terms widened by the absorbed micro-block slack.
The result is published only when both enclosure endpoints round to the
same finite binary64 value; otherwise crossing enclosures and phase
precisions are refined and the sum is rebuilt, up to the documented caps,
after which the typed failure is raised. For the simplicity fast path
(no crossings found), the published value is bitwise `abs(signed_area)`
with no enclosure work.

Determinism follows the area specification: same committed state, same
package version, same result or same typed failure.

# 9. Lazy state, caching, serialization

The contracts of area-specification sections 6.3, 7.2, 7.4, 7.5, 11 apply
verbatim to the fill cache, with these bindings:

- `CubicPHSplineClosed` and `ClosedNURBSHandle` hold one additional
  immutable empty-marker cache slot; pickle restoration clears it;
- `PHBSplineClosed` holds a `(version, fill_float)` tuple invalidated by
  version mismatch; per-span reuse is not required (the decomposition is
  global), but a conforming implementation MAY add it later;
- closed snapshots answer from their captured state forever;
- serialized payloads omit the fill cache; deserialization performs no
  fill computation.

# 10. Nonconforming shortcuts

Explicitly forbidden:

- polygonization, sampling, rasterization, Monte-Carlo, or an external
  clipping/Boolean library in the production path;
- float-only intersection finding without exact certification;
- returning $\sum_i |A(C_i)|$ (wrong whenever loops nest with mixed
  orientations) or conflating nonzero-winding with even-odd fill;
- assuming `fill_area >= area` or `fill_area <= area`;
- assuming a simple curve from `nu == +-1` or from the input polygon;
- reading the rounded public NURBS control points for offset fill;
- computing fill during construction, edit commit, or deserialization;
- returning an uncertified midpoint after precision exhaustion.

# 11. Test requirements

Tests SHALL prove at minimum:

- API and topology exposure exactly as section 2.1, including snapshots
  and the `hasattr` battery on open types;
- bitwise `fill_area == area` on certified-simple representatives of all
  four closed varieties, including cusp-free offsets and a fully
  reversed simple offset loop;
- figure-eight sources: `fill_area` equals the independent
  crossing-split oracle within its accuracy and exceeds `area`;
- limacon sources: `fill_area` equals the outer-loop area and is below
  `area`;
- invariance battery: traversal reversal, seam rotation, translation,
  rotation, reflection (tolerance-level on distinct constructions);
- offsets of self-intersecting sources and cusp-forming offsets whose
  loops decompose (crossing count and fill verified against an
  independent oracle);
- laziness and caching: no fill work in constructors, edits, offset
  construction, snapshot creation, pickling, or unpickling; version
  invalidation; snapshot retention; O(1) warm queries;
- serialization round trips before and after the first query.

# 12. Performance

Let $S$ be the span count, $C$ the metric cell count, and $X$ the
crossing count. The simplicity fast path SHOULD run in
$O((S{+}C)^2)$ worst-case pair pruning with near-linear observed cost;
the decomposition path adds certified localization per crossing and
$O(X^2)$ containment tests. Repeated queries are O(1). No complexity
term may depend on a sampling tolerance. Benchmarks SHOULD report the
fast-path fraction, certified crossing counts, absorbed micro-block
slack, and maximum phase precision.

# 13. Relationship to the area specification

`ClosedSpline_Area_Specification.md` section 2.3 excludes fill-rule areas
from the *algebraic* interface; this document adds the nonzero-winding
fill as a separate certified interface without altering any algebraic
contract. The exact-reference model, join-closure convention,
double-double/exact-rational layering, `atan2_ball` enclosures, phase
cells, and cache disciplines are shared, not duplicated.
