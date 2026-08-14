# Area of Closed PH Splines and Their Exact Offset NURBS

**Status:** normative implementation specification and implementation handoff<br>
**Applies to:** `CubicPHSplineClosed`, `PHBSplineClosed`, immutable snapshots of
closed PH B-splines, and exact closed offset NURBS returned from those types<br>
**Reviewed baseline:** package version 1.3.0 on 2026-08-14<br>
**Technical confidence:** high; the coefficient identities were checked in
exact rational arithmetic through the package's maximum position degree, and
the offset sign and cusp behavior were checked against independent numerical
line integration<br>
**Does not specify:** open curves, generic Bezier curves, generic B-splines,
generic NURBS, fill-rule or Boolean region area, offset trimming, or spline
construction changes unrelated to area

The words **SHALL**, **SHALL NOT**, **SHOULD**, and **MAY** are normative. An
implementation conforms only if every SHALL requirement is met. Mathematical
formulas define the result. They do not require one programming language or
one internal data layout.

# 1. Executive decisions

The implementation SHALL make these decisions explicit:

1. Only closed package types have an area interface. Open PH splines and open
   offset handles SHALL NOT have `area` or `signed_area` members. They SHALL
   not expose a member that raises a topology error when called.
2. `signed_area` is the normative primitive. It is the winding-number-weighted
   algebraic area

   $$
   A=\frac12\oint_C(x\,dy-y\,dx).
   $$

   Counterclockwise traversal is positive in the package's Cartesian
   coordinate convention. Clockwise traversal is negative.
3. `area` is exactly the nonnegative magnitude `abs(signed_area)`. It is not
   the sum of absolute lobe areas, an even-odd fill area, a nonzero-winding
   fill area, or the area of a trimmed offset.
4. For a simple closed curve, `area` is the ordinary enclosed area. For a
   self-intersecting curve, oppositely wound regions cancel in `signed_area`
   before `area` takes the magnitude.
5. Calculation is lazy. Construction, edit commit, snapshot creation, offset
   construction, copying, and deserialization SHALL NOT start an area
   calculation. The first area query for a geometry revision calculates and
   caches the result. Repeated queries are O(1).
6. Polynomial PH source area SHALL be calculated analytically from the stored
   Bernstein position coefficients. Numerical quadrature, sampling,
   tessellation, flattening, and polygon approximation are forbidden in the
   production result.
7. Exact PH offset area SHALL use the parallel-curve identity

   $$
   \boxed{A_d=A_0-dL_0+\pi\nu d^2,}
   $$

   where $d$ is the package's signed left-normal offset, $L_0$ is the source
   PH curve length, and $\nu$ is the source tangent turning number. The
   implementation SHALL NOT integrate the rational NURBS and SHALL NOT use
   the offset's unsigned `length` property in this formula.
8. All ordinary arithmetic SHALL run in the package normalization frame.
   A certified extended-precision fast path SHOULD handle normal cases. An
   exact-rational or certified adaptive-precision fallback SHALL resolve
   severe cancellation and rounding ambiguity.
9. A finite value SHALL be returned only when its binary floating-point
   rounding is certified. The implementation SHALL not return NaN, infinity,
   a saturated value, or an uncertified cancellation residue.

# 2. Exact scope

## 2.1 Included source variants

This specification covers every closed curve variant that exists in this
package:

- `CubicPHSplineClosed` constructed by the strictly convex cyclic solver;
- `CubicPHSplineClosed` constructed by the general cyclic path with straight
  runs, auxiliary inflections, curvature-sign changes, and documented G1
  joins;
- `PHBSplineClosed` at every supported preimage degree and continuity order;
- every committed version of an editable `PHBSplineClosed`;
- an immutable snapshot of a `PHBSplineClosed`; and
- the exact closed rational offset returned by `offset(d)` from either closed
  source family, including `d == 0`, cusp-free offsets, cusp-forming offsets,
  and offsets with reversals or self-intersections.

Source self-intersection is not a reason to reject an area query. Offset cusps
and self-intersections are not reasons to reject an area query. They change
the geometric interpretation, but the algebraic line integral remains
defined.

## 2.2 Excluded types

The following types SHALL have no area interface:

- `PHSpline`;
- `CubicPHSpline`;
- `CubicPHSplineOpen`;
- `PHBSpline`;
- `PHBSplineOpen`;
- an open PH B-spline snapshot; and
- an open `NURBSHandle` returned by an open source.

This document does not define area for an arbitrary Bezier control net, an
arbitrary polynomial B-spline, or an arbitrary NURBS. The Bernstein formulas
in Section 5 are internal kernels for the package types listed in Section
2.1, not a new generic public geometry API.

## 2.3 Explicitly excluded meanings of area

This version does not build a planar arrangement and does not classify
faces. Therefore, it does not calculate:

- union area of all regions touched by a self-intersecting curve;
- even-odd raster fill area;
- nonzero-winding raster fill area with absolute face weights;
- material area after trimming offset loops or cusps;
- the area between an open spline and an implicit closing chord; or
- swept area between two unrelated curves.

Those operations require intersection isolation, face construction, and an
explicit fill rule. Calling the algebraic result a fill-rule area would be
incorrect.

Amendment: the separate addendum `ClosedSpline_FillArea_Specification.md`
adds the nonzero-winding fill area as its own certified `fill_area`
interface on the same closed types. It does not change any algebraic
`signed_area`/`area` contract of this document.

# 3. Public API and structural topology

## 3.1 Required properties

Closed polynomial types SHALL expose:

```python
class CubicPHSplineClosed(CubicPHSpline):
    @property
    def signed_area(self) -> float: ...

    @property
    def area(self) -> float: ...


class PHBSplineClosed(PHBSpline):
    @property
    def signed_area(self) -> float: ...

    @property
    def area(self) -> float: ...
```

The corresponding closed PH B-spline snapshot SHALL expose the same two
read-only properties.

`area` SHALL call or reuse `signed_area`; it SHALL NOT run an independent
calculation. Its definition is

```text
area = abs(signed_area)
```

An exact zero SHALL be published as positive zero. If a nonzero exact result
correctly rounds to a signed zero, `signed_area` MAY preserve the IEEE 754
sign; `area` SHALL return positive zero.

## 3.2 Closed offset handle type

The current `NURBSHandle` class represents both open and closed offsets. A
property placed on that class would also create an area interface on open
handles. That design is nonconforming.

The implementation SHALL add a topology-specific closed subtype, or an
equivalent static type split:

```python
class NURBSHandle:
    # Existing common API. No area properties.
    ...


class ClosedNURBSHandle(NURBSHandle):
    @property
    def signed_area(self) -> float: ...

    @property
    def area(self) -> float: ...
```

`build_offset_handle(...)` SHALL instantiate `ClosedNURBSHandle` when its
verified source topology is closed and the base `NURBSHandle` when it is
open. Public return annotations and type stubs SHALL preserve this fact:

- `CubicPHSplineClosed.offset(...) -> ClosedNURBSHandle`;
- `PHBSplineClosed.offset(...) -> ClosedNURBSHandle`; and
- open-family `offset(...) -> NURBSHandle` with no area member.

`ClosedNURBSHandle` SHOULD be exported from `ph_spline.__init__`. Existing
code that accepts `NURBSHandle` remains compatible because the closed type is
a subtype.

Runtime conformance includes:

```python
assert hasattr(closed_curve, "area")
assert hasattr(closed_curve.offset(0.0), "area")
assert not hasattr(open_curve, "area")
assert not hasattr(open_curve.offset(0.0), "area")
```

No `OpenCurveAreaError` or equivalent exception SHALL be added.

## 3.3 Closed snapshot type

The same structural rule applies to snapshots. The preferred design is:

```python
class PHBSplineSnapshot:
    # Common query surface. No area properties.
    ...


class PHBSplineClosedSnapshot(PHBSplineSnapshot):
    @property
    def signed_area(self) -> float: ...

    @property
    def area(self) -> float: ...
```

`PHBSplineClosed.snapshot()` SHALL return the closed snapshot type. An
equivalent generated protocol or static wrapper is conforming only if open
snapshots do not advertise either area member at runtime or to static type
checkers.

## 3.4 Units and orientation

Both properties return a scalar in squared user-coordinate units. They are
independent of parameter speed and of the location selected as the closed
seam.

For a simple curve:

- counterclockwise traversal gives `signed_area > 0`;
- clockwise traversal gives `signed_area < 0`; and
- reversing traversal negates `signed_area` without changing `area`.

# 4. Mathematical definitions

## 4.1 Determinant notation

For planar vectors $a=(a_x,a_y)$ and $b=(b_x,b_y)$, define

$$
[a,b]=a_xb_y-a_yb_x.
$$

For complex values $a=a_x+ia_y$ and $b=b_x+ib_y$,

$$
[a,b]=\operatorname{Im}(\overline a b).
$$

The implementation SHOULD use a determinant or complex cross-product helper
whose intermediate products are evaluated with error-free transforms or a
certified extended-precision equivalent.

## 4.2 Signed algebraic area

Let $r(u)=(x(u),y(u))$ traverse a closed piecewise regular curve once in
package parameter order. The required result is

$$
\boxed{
A(r)=\frac12\int_0^1 [r(u),r'(u)]\,du
=\frac12\operatorname{Im}\int_0^1\overline{z(u)}z'(u)\,du.
}
$$

This definition is translation invariant for a closed chain. It is also
valid for immersed curves. Equivalently,

$$
A(r)=\iint_{\mathbb R^2}\operatorname{wind}(r,q)\,dq,
$$

where `wind` is the integer winding number of the oriented curve about a
point outside the curve. This equivalence explains why opposite lobes can
cancel.

## 4.3 Exact-reference state

The authoritative inputs to an area query are the binary floating-point
values in one committed, verified curve state, each interpreted as an exact
real value. The query SHALL not reconstruct geometry from sampled public
points.

For a polynomial source, the authoritative area data are its normalized
Bernstein position controls, the normalization scale $H$, and the ordered
span topology. For an exact offset, the authoritative data are a frozen
snapshot of the generating source position controls and PH preimages, its
scale $H$, its accepted binary floating-point offset $d$, and its source
span order.

This follows the exact-reference model already used by
`OffsetNURBS_Distance_Specification.md`: a PH offset handle represents the
verified exact parallel of captured PH source state. Area SHALL not be
inferred later from the rounded public NURBS control points.

## 4.4 Numerical join closure

In exact mathematics, adjacent source spans share endpoints. In finite
storage, a verified interpolation or reconstruction residual can leave two
one-sided Bernstein endpoints that differ by a few ulps. A sum of open-span
integrals alone would then acquire an anchor-dependent error.

The reference calculation SHALL make the stored chain exactly closed by
including the line-integral contribution of the residual connector from the
end of each span to the start of the next span, cyclically:

$$
A_{\mathrm{join}}
=\frac12\sum_{s=0}^{S-1}
[C_{s,p_s},C_{(s+1)\bmod S,0}].
$$

For an exact shared endpoint, its term is identically zero. These connectors
are a numerical closure convention, not new public geometry. They SHALL be
formed from the same committed normalized control arrays and included in the
same certified accumulator. This convention gives all of these properties
even when stored endpoints differ slightly:

- exact translation invariance of the complete stored chain;
- exact cyclic-seam invariance;
- no hidden dependence on the chosen anchor; and
- bitwise agreement between a source and the captured source term of
  `source.offset(0)`.

An implementation that stores bitwise shared endpoints MAY omit the zero
connector operations, but it SHALL verify the bitwise equality before doing
so.

# 5. Exact polynomial Bernstein area kernel

## 5.1 One span

Let one normalized polynomial span of degree $p\ge1$ be

$$
c(t)=\sum_{i=0}^{p} C_i B_i^p(t),\qquad 0\le t\le1,
$$

where $C_i\in\mathbb R^2$. Its derivative is

$$
c'(t)=p\sum_{j=0}^{p-1}(C_{j+1}-C_j)B_j^{p-1}(t).
$$

The Bernstein product identity and basis integral are

$$
B_i^p B_j^{p-1}
=\frac{\binom p i\binom{p-1}j}
       {\binom{2p-1}{i+j}}B_{i+j}^{2p-1},
\qquad
\int_0^1B_k^{2p-1}(t)\,dt=\frac1{2p}.
$$

Substitution into Green's line integral gives the direct implementation
formula

$$
\boxed{
A_{\mathrm{span}}
=\frac14\sum_{i=0}^{p}\sum_{j=0}^{p-1}
\frac{\binom p i\binom{p-1}j}
     {\binom{2p-1}{i+j}}
[C_i,C_{j+1}-C_j].
}
\tag{5.1}
$$

There is no numerical integration and no division by curve speed. All
denominators in (5.1) are positive integers known from the degree.

## 5.2 Pairwise determinant form

The pairwise form avoids repeated control differences and uses about half as
many determinants. For $0\le a<b\le p$, define the strictly positive
rational coefficient

$$
\boxed{
K_{ab}^{(p)}=
\frac{(b-a)\binom p a\binom p b}
     {2(2p-1)\binom{2p-2}{a+b-1}}.
}
\tag{5.2}
$$

where a fraction whose denominator has an out-of-range lower index is zero.
Then

$$
\boxed{
A_{\mathrm{span}}=\sum_{0\le a<b\le p}K_{ab}^{(p)}[C_a,C_b].
}
\tag{5.3}
$$

The implementation SHOULD use (5.3). It SHALL generate
$K_{ab}^{(p)}$ from exact integer binomial coefficients and exact rational
division. It SHALL NOT generate the table through numerical quadrature or
through inversion of a sampled Vandermonde system.

The table depends only on $p$. It SHOULD be cached once per degree on first
area use. Table generation is part of lazy area work; it SHALL not occur
during spline construction merely because a curve of degree $p$ was built.

## 5.3 Composite closed chain

For $S$ ordered normalized spans with degrees $p_s$, the normalized area is

$$
\boxed{
\widehat A
=\sum_{s=0}^{S-1}
  \sum_{0\le a<b\le p_s}K_{ab}^{(p_s)}[C_{s,a},C_{s,b}]
+\frac12\sum_{s=0}^{S-1}
  [C_{s,p_s},C_{(s+1)\bmod S,0}].
}
\tag{5.4}
$$

If the package normalization is

$$
r=O+H\widehat r,
$$

the user-coordinate area is

$$
\boxed{A=H^2\widehat A.}
\tag{5.5}
$$

The origin $O$ does not occur. This is the primary defense against
catastrophic cancellation for small shapes translated to large user
coordinates.

The multiplication by $H^2$ SHALL use exponent-scaled or exact arithmetic.
It SHALL not form `H * H` first when that product can overflow even though
`H * (H * A_hat)` or the exact final result is finite.

## 5.4 Derivation of the pairwise coefficient

For implementation review, the coefficient in (5.2) follows without an
unstated identity. The coefficient of $[C_a,C_b]$, $a<b$, is

$$
\frac12\int_0^1
\left(B_a^p(B_b^p)'-B_b^p(B_a^p)'\right)dt.
$$

The logarithmic derivative of a Bernstein basis function is

$$
\frac{(B_i^p)'(t)}{B_i^p(t)}
=\frac{i}{t}-\frac{p-i}{1-t}.
$$

Therefore,

$$
B_a^p(B_b^p)'-B_b^p(B_a^p)'
=\frac{b-a}{t(1-t)}B_a^pB_b^p.
$$

The right side is positive on $(0,1)$ because $b>a$. Direct beta integration
gives

$$
\begin{aligned}
K_{ab}^{(p)}
&=\frac{b-a}{2}\binom pa\binom pb
  \int_0^1t^{a+b-1}(1-t)^{2p-a-b-1}\,dt\\
&=\frac{b-a}{2}\binom pa\binom pb
  \frac{(a+b-1)!(2p-a-b-1)!}{(2p-1)!},
\end{aligned}
$$

which is exactly (5.2). The endpoint exponents are nonnegative for every
$0\le a<b\le p$. This derivation proves strict positivity and provides an
independent coefficient-table test against the difference formula obtained
by substituting $(B_i^p)'=p(B_{i-1}^{p-1}-B_i^{p-1})$ into Section 5.1.

# 6. Cubic PH spline procedure

## 6.1 Authoritative data

Each `PHSegment` already stores the normalized cubic Bezier control net
`ctrl`. `CubicPHSplineClosed.signed_area` SHALL use those position controls.
It SHALL not call `point(u)`, reconstruct controls from sampled values, or
re-solve PH preimages.

The strictly convex and general cyclic construction paths use the same area
kernel. Straight spans and auxiliary-inflection spans need no special case.

## 6.2 Specialized cubic expression

For $C_0,C_1,C_2,C_3$, (5.3) reduces to

$$
\boxed{
\begin{aligned}
A_{\mathrm{cubic}}={}&
\frac3{10}\big([C_0,C_1]+[C_2,C_3]\big)\\
&+\frac3{20}\big([C_0,C_2]+[C_1,C_2]+[C_1,C_3]\big)
+\frac1{20}[C_0,C_3].
\end{aligned}
}
\tag{6.1}
$$

The reference implementation SHOULD use this fixed expression or a
precomputed exact degree-3 table. It requires six determinants per cubic
span. The join correction from (5.4) remains required when adjacent stored
endpoints are not bitwise equal.

## 6.3 Lazy cache

`CubicPHSplineClosed` is immutable. It SHALL contain an initially empty
derived cache, for example `_signed_area_cache = None`. Construction SHALL
only initialize the empty marker. The first query computes and atomically
publishes the finite float. Later queries return that value directly.

The cache is not authoritative state. Pickle restoration SHALL clear it or
shall verify it by full recomputation before accepting it. Clearing is the
preferred behavior.

# 7. Editable PH B-spline procedure

## 7.1 Authoritative data

Each `PHBSplineSpan` stores normalized Bernstein position controls in
`position`. If the preimage degree is $m$, the position degree is

$$
p=2m+1.
$$

`PHBSplineClosed.signed_area` SHALL apply (5.3) to each committed span's
`position` controls, add the cyclic join terms in (5.4), and scale by $H^2$.
It SHALL not convert the complete curve to a global power basis. It SHALL not
perform quadrature.

The local parameter width does not appear in the final position-control
formula. Reparameterizing a span without changing its oriented locus does not
change its area.

## 7.2 Versioned whole-result cache

The mutable curve SHALL cache the public result with its committed version:

```text
(version, signed_area_float)
```

The cache is valid only when its version equals `curve.version`. A successful
edit SHALL invalidate the whole-result cache without calculating a new area.
A failed edit SHALL leave both geometry and the existing area cache
unchanged.

## 7.3 Reuse after local edits

The existing local edit builder retains unchanged immutable span kernels and
keeps the normalization frame $(O,H)$ stable. The area implementation SHOULD
exploit that state.

It SHOULD keep a private per-span cache keyed by immutable position-control
identity plus normalization-frame generation. Each entry contains a
certified normalized span contribution from (5.3), not a public rounded
whole-curve area. On the first query after a local edit:

1. reuse entries for bitwise retained position arrays in the same frame;
2. calculate entries only for new or changed position arrays;
3. recompute all cheap join terms because they depend on neighboring spans;
4. aggregate all contributions with the certified accumulator; and
5. publish the new whole-result cache only for the captured version.

No new span contribution SHALL be calculated during edit commit. Cache
pruning, identity comparison, and empty-marker publication are permitted
because they do not evaluate area.

A global rebuild normally changes $(O,H)$ and all span kernels. It SHALL use
a new frame generation and SHALL not reuse normalized contributions from the
old frame.

## 7.4 Snapshot isolation

A closed snapshot retains one immutable compiled state and SHALL return the
area of that state after later source edits. Any mutable cache dictionary
copied into a snapshot SHALL be cloned, or cache updates SHALL use immutable
copy-on-write maps. A shallow-shared mutable area cache between an editable
source and its snapshot is nonconforming.

Snapshot construction MAY copy an already computed valid scalar result. It
SHALL not trigger a calculation when the source result is absent.

## 7.5 Concurrent edit and query

An area query on a mutable curve SHALL capture one immutable build state,
normalization frame, and version before work starts. It SHALL calculate only
from that capture. If an edit commits concurrently, the old result is
linearizable at capture time and MAY be returned, but it SHALL not be
published into the new version's cache. An implementation MAY instead retry
against the new version.

No reader may observe a partially accumulated result or a cache tagged with
the wrong version.

# 8. Exact PH offset area

## 8.1 Parallel-curve model and sign

Let the regular closed source be parameterized by source arc length $s$.
Write

$$
T=\frac{dr}{ds},\qquad N_L=J T,\qquad
J(x,y)=(-y,x),
$$

and let signed curvature satisfy

$$
\frac{dT}{ds}=\kappa N_L,
\qquad
\frac{dN_L}{ds}=-\kappa T.
$$

The package offset is

$$
r_d=r+dN_L.
$$

Positive $d$ is left-normal. Thus, a positive offset of a counterclockwise
simple curve goes inward. This sign is easy to reverse accidentally and SHALL
be tested with both orientations.

## 8.2 Derivation

Differentiate the offset:

$$
\frac{dr_d}{ds}=(1-d\kappa)T.
$$

Since $[N_L,T]=-1$,

$$
\begin{aligned}
2A_d
&=\oint [r+dN_L,(1-d\kappa)T],ds\\
&=\oint\left([r,T]-d\kappa[r,T]-d+d^2\kappa\right)ds.
\end{aligned}
$$

Also,

$$
\frac d{ds}[r,N_L]
=[T,N_L]+[r,-\kappa T]
=1-\kappa[r,T].
$$

Its integral around the closed curve is zero, so

$$
\oint\kappa[r,T],ds=L_0.
$$

The total signed curvature is

$$
\oint\kappa\,ds=2\pi\nu,
$$

where $\nu\in\mathbb Z$ is the tangent turning number. Therefore,

$$
\boxed{A_d=A_0-dL_0+\pi\nu d^2.}
\tag{8.1}
$$

The derivation is valid for a regular piecewise-C1 source with a continuous
tangent and integrable piecewise curvature. That includes the closed cubic
PH family's documented G1 joins and the closed PH B-spline family.

The offset itself need not be regular. A zero of $1-d\kappa$ creates a cusp,
but it does not invalidate the polynomial identity used above. Equation
(8.1) therefore remains the required algebraic area through cusp creation,
reversal, looping, and self-intersection. No absolute value may be inserted
inside the line integral.

## 8.3 Important length distinction

$L_0$ in (8.1) is the source's signed-parameter speed integral

$$
L_0=\oint ds>0.
$$

It is not `ClosedNURBSHandle.length`. The handle property is the unsigned
offset traversal length

$$
L_d=\oint|1-d\kappa|\,ds,
$$

which differs after the offset reverses and can already differ at cusp
boundaries. Substituting $L_d$ in (8.1) is a critical implementation defect.

## 8.4 Source length from PH coefficients

For a source span with local coordinate $t$, positive parameter-width factor
$h_s$, and complex Bernstein preimage

$$
w_s(t)=\sum_{i=0}^{m}b_{s,i}B_i^m(t),
$$

the normalized derivative and speed are

$$
\widehat z_s'(t)=h_s w_s(t)^2,
\qquad
\widehat\sigma_s(t)=h_s|w_s(t)|^2.
$$

Let

$$
|w_s(t)|^2=\sum_{k=0}^{2m}\rho_{s,k}B_k^{2m}(t),
$$

with exact Bernstein product coefficients

$$
\boxed{
\rho_{s,k}=
\sum_{i+j=k}
\frac{\binom mi\binom mj}{\binom{2m}k}
b_{s,i}\overline{b_{s,j}}.
}
\tag{8.2}
$$

The verified imaginary residual is zero in the exact-reference model. The
exact source length is

$$
\boxed{
L_0=H\sum_s\frac{h_s}{2m+1}
\sum_{k=0}^{2m}\rho_{s,k}.
}
\tag{8.3}
$$

For a cubic PH segment, $m=1$ and $h_s=1$ in its local coordinate. For a PH
B-spline span, `PHBSplineSpan.parameter_width` supplies $h_s$.

The reference offset metric already compiles the $\rho_{s,k}$, $h_s$, and
$H$ as exact rational values in `_SpanExact`. The area query SHALL reuse that
captured certificate. It SHALL not use the rounded public source `length`
when cancellation in (8.1) could expose its rounding error.

## 8.5 Offset area provenance

A closed offset handle needs source data that generic NURBS controls do not
contain. Offset construction SHALL attach an immutable private area
provenance record containing, or sharing immutable ownership of:

- the ordered normalized source position control arrays before positivity
  refinement;
- the exact-reference source PH preimages;
- source span width factors;
- the source normalization scale $H$;
- the accepted signed offset $d$;
- closed topology and cyclic span order; and
- a reference to the verified offset metric certificate when it already
  owns equivalent exact source data.

Capturing references or read-only copies is not area calculation. The record
SHALL be created without evaluating (5.4), (8.1), a turning number, or any
area cache.

The record SHALL own immutable snapshots, not a reference to the mutable
source curve. An offset from PH B-spline version $v$ remains unchanged after
version $v+1$ commits.

Source preimages SHOULD be shared with the existing metric certificate
instead of duplicated. Source position arrays MAY share immutable span
storage. Serialization SHALL preserve enough raw source state to rebuild and
verify the provenance.

## 8.6 Offset query procedure

On the first `ClosedNURBSHandle.signed_area` query:

1. calculate the exact or certified $A_0$ from the captured source position
   controls with (5.4) and (5.5);
2. calculate exact-reference $L_0$ with (8.3), reusing metric coefficients;
3. if `d == 0`, return the correctly rounded $A_0$ without calculating
   $\nu$ or $\pi$;
4. otherwise calculate the certified integer $\nu$ by Section 9;
5. evaluate (8.1) with the certified arithmetic of Section 10;
6. atomically cache the final finite float; and
7. make `area` return its absolute value.

The NURBS degree, public knot multiplicities, and any positive-weight
subdivision do not enter this procedure. Those operations change only the
rational representation of the same exact-reference offset.

# 9. Certified tangent turning number

## 9.1 PH phase identity

Because

$$
z_s'(t)=h_sw_s(t)^2,\qquad h_s>0,
$$

the unit tangent is

$$
T_s(t)=\frac{w_s(t)^2}{|w_s(t)|^2}.
$$

If $\phi$ is a continuous argument of $w_s$, the tangent angle inside one
span is $2\phi$. In exact mathematics, adjacent verified spans have the same
unit tangent and their join-angle contribution is zero. In the committed
binary exact-reference state, however, two one-sided preimage endpoints can
satisfy the continuity bound without producing bitwise equal squared
tangents. Omitting this small stored join angle would prevent an exact
integer certificate.

Let $q_s^-=w_s(1)^2$ and $q_s^+=w_s(0)^2$ denote nonzero, unnormalized
one-sided tangent vectors. At the join from span $s$ to span $s+1$, including
the cyclic seam, define the canonical shortest tangent correction

$$
\delta_s=\operatorname{atan2}
\left([q_s^-,q_{s+1}^+],q_s^-\cdot q_{s+1}^+\right).
$$

The existing verified G1 bound SHALL imply
$q_s^-\cdot q_{s+1}^+>0$, so $|\delta_s|<\pi/2$ and the correction is unique.
This is a finite-storage closure convention analogous to the position join
correction in Section 4.4. It does not add a geometric corner to the public
curve. It recovers exact zero when the squared tangent vectors are positively
proportional.

For a closed regular source, the required turning number is

$$
\boxed{
\nu=\frac1{2\pi}
\left(2\sum_{\mathrm{phase\ cells}}\Delta\arg w
+\sum_{\mathrm{cyclic\ joins}}\delta_s\right).
}
\tag{9.1}
$$

A sign change of preimage gauge between adjacent spans does not change
$w^2$. The squared-tangent join formula therefore needs no gauge special
case.

## 9.2 Reuse of the offset metric phase cells

For every nonzero offset distance, the existing offset-metric construction
already subdivides each nonconstant-phase preimage span until its exact
rational Bernstein hull lies strictly in the open half-plane of each endpoint
preimage. This proves that the continuous preimage phase change on a cell is
less than $\pi/2$ in magnitude.

The area query SHALL reuse those verified cell boundaries. It SHALL not infer
a turning number from sampled tangents, accumulated `atan2` calls on an
unverified coarse partition, signed curvature samples, the sign of area, or
the input polygon orientation.

For a metric cell with exact nonzero endpoint preimages $w_a,w_b$, define

$$
X=\operatorname{Re}(w_b\overline{w_a})=w_a\cdot w_b,
\qquad
Y=\operatorname{Im}(w_b\overline{w_a})=[w_a,w_b].
$$

The certified cell increment is

$$
\boxed{\Delta\phi=\operatorname{atan2}(Y,X)\in(-\pi/2,\pi/2).}
\tag{9.2}
$$

The endpoints SHALL be evaluated from the exact rational Bernstein
preimage and the cell's exact local bounds. Rounded cached endpoint values
are permitted only as a fast seed, not as the certification input.

A span whose phase polynomial is identically constant contributes exact
zero. For `d == 0`, the $d^2$ term vanishes, so the query SHALL skip the
turning calculation even if the source turns.

After the within-span cells, form each $q_s^-$ and $q_{s+1}^+$ by exact
complex squaring of the exact rational endpoint preimages. Verify their dot
product is strictly positive and evaluate the join correction with the same
`atan2_ball` kernel. A nonpositive dot product contradicts the source's
already-verified G1 contract and SHALL fail with the existing structured
numerical or continuity error; it SHALL not choose an arbitrary long arc.

## 9.3 Adaptive integer certification

At precision $P$ bits, use the existing exact-integer `atan2_ball` routine to
obtain for every preimage phase cell

$$
\Delta\phi_j\in
\left[\frac{v_j-e_j}{2^P},
      \frac{v_j+e_j}{2^P}\right].
$$

Multiply every cell ball by two because tangent phase is twice preimage
phase. Obtain one additional `atan2_ball` for every cyclic squared-tangent
join correction $\delta_s$. Sum the scaled cell balls and join balls without
floating-point arithmetic:

$$
V_\Theta=2\sum_jv_j+\sum_s v_s^{\mathrm{join}},
\qquad
E_\Theta=2\sum_je_j+\sum_s e_s^{\mathrm{join}}.
$$

Obtain a simultaneous `pi_ball(P)` enclosure

$$
\pi\in
\left[\frac{V_\pi-E_\pi}{2^P},
      \frac{V_\pi+E_\pi}{2^P}\right].
$$

Form the rational interval quotient

$$
Q=\frac{[V_\Theta-E_\Theta,V_\Theta+E_\Theta]}
        {[2(V_\pi-E_\pi),2(V_\pi+E_\pi)]}.
$$

The denominator interval is positive. Endpoint order SHALL account for the
sign of the numerator interval. The within-span tangent paths plus the
canonical join arcs form an exactly closed path in nonzero tangent-vector
space, so its winding is an integer. Accept only when $Q$ contains exactly
one integer; that integer is $\nu$ by (9.1).

Use the repository precision ladder

```text
256, 512, 1024, 2048, 4096 bits
```

or a proved stronger ladder. If the interval still does not isolate one
integer at the documented cap, raise the existing `NumericalPrecisionError`
with `operation="area"` and turning-number enclosure diagnostics. Do not
round an uncertified approximate quotient.

The implementation SHOULD also verify as an internal invariant that the
isolated integer is consistent with the complete tangent phase enclosure and
$2\pi\nu$.

## 9.4 Why the turning number is not assumed to be plus or minus one

A simple regular closed curve has $\nu=+1$ or $-1$. Package inputs can create
self-intersecting and multiply turning curves. Their turning number can be
zero or have magnitude greater than one. Hard-coding the sign of source area,
the sign of the input polygon, or `±1` is nonconforming.

# 10. Numerical arithmetic and certification

## 10.1 Required accuracy contract

For source polynomial area, a conforming binary64 implementation SHALL return
the correctly rounded nearest-even binary64 value of (5.4)-(5.5), with every
stored binary64 coefficient interpreted exactly.

For offset area, it SHALL return a binary64 value only when a certified
enclosure of (8.1) maps to one nearest-even binary64 value. Because $\pi$ is
transcendental, an adaptive enclosure is required in the ambiguous cases.

If the finite exact result is outside binary64 range, the implementation
SHALL raise the existing `NumericalPrecisionError` with
`operation="area"`, the quantity name, and the representable bound. It SHALL
not return infinity. Correctly rounded gradual underflow, including a signed
zero for a sufficiently small nonzero result, is permitted.

## 10.2 Normalized-coordinate rule

All determinant sums SHALL be performed on normalized position controls.
The implementation SHALL not first denormalize controls to
$O+HC_i$. The latter can lose all low-order shape bits when $O$ is large and
can make products overflow although the translated area is representable.

For local PH B-spline edits, the retained normalization frame remains the
frame for cached span contributions. A frame change invalidates them.

## 10.3 Fast path

The reference Python implementation SHOULD use the existing double-double
module and FMA-based error-free transforms. Other languages MAY use a wider
native format, floating-point expansions, an exact dot-product facility, or
interval arithmetic.

Each fast-path scalar SHALL be represented as a center plus a nonnegative
outward-rounded radius. The center SHOULD be a double-double pair. Error
bounds SHALL be propagated per operation; a fixed comparison such as
`abs(area) > 1e-12` is forbidden.

For a double-double center $c=(c_h,c_l)$, define an upward-rounded magnitude
bound

$$
M(c)=\operatorname{nextUp}(|c_h|+|c_l|).
$$

Using the operation bounds already documented by `ph_spline.ddouble`, a
conforming ball layer MAY use these conservative recurrences, with every
right-hand side rounded upward:

$$
\begin{aligned}
r_{a+b}&=r_a+r_b
 +4\epsilon_{dd}\big(M(a)+M(b)\big),\\
r_{ab}&=M(a)r_b+M(b)r_a+r_ar_b
 +8\epsilon_{dd}\big(M(a)+r_a\big)\big(M(b)+r_b\big),
\end{aligned}
\tag{10.1}
$$

where $\epsilon_{dd}=2^{-104}$. Exact `two_sum`/`two_prod` results start with
zero radius when no overflow or underflow occurred. Exact rational
$K_{ab}^{(p)}$ values SHALL be converted to a double-double enclosure, not
to one unchecked binary64 coefficient.

For each determinant, evaluate both products with `two_prod` and subtract
them with the ball operation. Accumulate weighted determinants, join terms,
and span totals in a balanced or magnitude-aware order. Track the sum of
absolute term bounds as a condition diagnostic.

After multiplying by $H$ twice through ball multiplication, expand the final
interval outward by one binary64 `nextafter` step at each end. The fast result
is accepted only if both interval endpoints round to the same finite
binary64 value. Otherwise, use the fallback in Section 10.4.

If any fast intermediate becomes nonfinite, do not rescale by an empirical
constant and do not discard terms. Go directly to the exact fallback.

## 10.4 Exact rational fallback for polynomial area

Every finite binary floating-point value is an exact dyadic rational. The
fallback SHALL decode controls and $H$ exactly, construct the rational
coefficients (5.2) from integers, and evaluate (5.4)-(5.5) in exact rational
arithmetic.

The implementation SHOULD avoid a separate rational object allocation for
every elementary product when performance matters. One conforming approach
is:

1. decode binary floats into integer significands and powers of two;
2. precompute a common integer denominator for the degree-$p$ coefficient
   table;
3. accumulate determinant numerators with arbitrary-size integers;
4. include connector and $H^2$ factors exactly; and
5. perform one correctly rounded rational-to-binary64 conversion.

The existing Python `fractions.Fraction` type is a simpler conforming first
implementation because the package degree cap is modest. A later optimized
integer accumulator is permitted if it produces identical results.

Exact zero detection SHALL happen in this path. No absolute or relative
tolerance may replace it.

## 10.5 Certified offset evaluation

Write (8.1) as

$$
A_d=R+C\pi,
$$

where

$$
R=A_0-dL_0\in\mathbb Q,
\qquad
C=\nu d^2\in\mathbb Q.
$$

Both are exact rationals under the exact-reference convention. If $C=0$,
convert $R$ directly with correct rounding.

Otherwise, at each precision on the adaptive ladder, get
$\pi\in[\pi_-,\pi_+]$ from `pi_ball`. Form exact rational bounds

$$
[A_-,A_+]=
\begin{cases}
[R+C\pi_-,R+C\pi_+],&C>0,\\
[R+C\pi_+,R+C\pi_-],&C<0.
\end{cases}
\tag{10.2}
$$

Return only when $A_-$ and $A_+$ round to the same finite binary64 number.
Escalate through 4096 bits otherwise. At the cap, raise the existing
`NumericalPrecisionError`; never choose the midpoint of an unresolved
enclosure.

This procedure directly resolves catastrophic cancellation among $A_0$,
$dL_0$, and $\pi\nu d^2$. It also avoids overflow in `d*d` and `H*H` until
the final representability decision.

## 10.6 Division-by-zero and singularity audit

The source area path divides only by positive degree-dependent integers.
The source-length path divides only by $2m+1>0$. The turning path calls
`atan2(Y,X)` only for nonzero regular preimage endpoints, so $(X,Y)\ne(0,0)$.
The offset formula contains no division.

Therefore an area query SHALL not evaluate:

- $1/|w|$;
- $1/(1-d\kappa)$;
- a NURBS weight reciprocal;
- a rational antiderivative at a NURBS pole; or
- a numerical curvature integral.

Offset cusps cannot cause division by zero in this design.

## 10.7 Determinism

For the same serialized committed state, package version, and supported
arithmetic profile, the result and accept-or-fail decision SHALL be
deterministic. Thread scheduling may cause duplicate private computation, but
not a different published float.

# 11. Lazy state, lifetime, and serialization

## 11.1 Empty state

Every closed-area-capable object SHALL initialize a cache marker only. It
SHALL not initialize coefficient tables, exact rationals, phase cells,
turning numbers, or area values during construction.

An offset handle MAY reuse metric cells that its existing distance contract
already requires at offset construction. Reusing that state does not permit
eager summation into an area or turning number.

## 11.2 Atomic publication

The cache SHALL have only two observable states:

- empty; or
- one complete finite binary64 result associated with an immutable state or
  exact version.

Two concurrent first readers MAY calculate the same deterministic value.
Publication SHALL be atomic. A lock, compare-and-swap, or runtime-specific
single-reference assignment is conforming. A partially populated mutable
record is not.

## 11.3 Editing

A successful PH B-spline edit SHALL:

- publish the new verified geometry first as the existing transaction
  contract requires;
- increment the version;
- leave the new whole-area cache empty;
- preserve only safe per-span cache entries from bitwise retained spans in
  the unchanged frame; and
- perform no determinant or global area sum.

The first later query performs the work. A failed edit changes nothing.

## 11.4 Copy and pickle

Area caches are derived, nonauthoritative data. The preferred persistence
contract is:

- omit scalar and per-span area caches from serialized state;
- serialize immutable offset area provenance with the closed handle;
- on restore, validate provenance dimensions, finiteness, degree agreement,
  span order, positive widths, scale, offset, and closed topology;
- rebuild or share the verified metric certificate as already required; and
- restore an empty area cache.

Copying an immutable closed cubic or closed offset handle MAY copy an already
computed scalar cache. Copying an editable PH B-spline or making a snapshot
must preserve version isolation as specified in Section 7.

# 12. Platform-specific implementation notes

This section is specific to the current Python/NumPy stack. It does not alter
the language-neutral mathematical contract.

## 12.1 Suggested module boundary

Add a private module such as `ph_spline/area.py` that owns:

- exact degree-table generation;
- cubic and general Bernstein span kernels;
- join correction and normalized composite reduction;
- double-double ball evaluation;
- exact-rational fallback and float publication;
- exact source-length extraction from offset metric state;
- turning-number certification from exact phase cells; and
- evaluation of $R+C\pi$.

This avoids copying formulas into `cubic.py`, `bspline.py`, and `nurbs.py`.
Those modules should contain only topology-specific properties, cache
capture/publication, and provenance wiring.

The area module MAY reuse private helpers from `exact_real.py`,
`ddouble.py`, and a factored phase-cell helper from `offset_metric.py`.
Circular imports SHALL be avoided. A small immutable internal protocol for
area provenance is preferable to importing spline classes into `area.py`.

## 12.2 Class changes

Expected surgical changes are:

- `ph_spline/cubic.py`: empty immutable cache slot and properties only on
  `CubicPHSplineClosed`;
- `ph_spline/bspline.py`: versioned cache, optional per-span reuse, properties
  only on `PHBSplineClosed`, and a closed snapshot return type;
- `ph_spline/nurbs.py`: `ClosedNURBSHandle`, immutable area provenance,
  topology-selected construction, and lazy cache;
- `ph_spline/offset_metric.py`: expose or factor an internal read-only phase
  cell view and exact source-speed state without duplicating algorithms;
- `ph_spline/exact_real.py`: reuse `atan2_ball`, `pi_ball`, and
  `ball_to_fraction_bounds`; no second transcendental implementation;
- `ph_spline/ddouble.py`: reuse documented operations through an area ball
  wrapper; do not weaken the existing bounds;
- `ph_spline/__init__.py`: export `ClosedNURBSHandle` and the closed snapshot
  type if it is public; and
- `ph_spline/typing.py` or type stubs: closed-only return types and
  properties.

Do not add area to `PHSpline`, `CubicPHSpline`, `PHBSpline`, or the common
`NURBSHandle` base.

## 12.3 Python cache details

`CubicPHSpline` and `NURBSHandle` use frozen `__slots__`. Their closed
subclasses may add cache slots, or the common storage may contain a private
empty slot that has no public property on open instances. Internal cache
mutation SHALL use `object.__setattr__` only after a complete result exists;
public geometric immutability remains unchanged.

`PHBSpline` has a `__dict__`. A versioned tuple is sufficient for the scalar
cache. If a per-span dictionary is used, `PHBSplineSnapshot.__init__` SHALL
clone it rather than shallow-share it.

NumPy arrays used as cache keys are unhashable. A Python implementation MAY
key by `id(position_array)` only if each cache entry also holds a strong
reference and confirms object identity, preventing object-ID reuse. A stable
immutable span-kernel token is cleaner if one is added without changing
public behavior.

## 12.4 Correctly rounded conversion

Python's arbitrary-size integers and `Fraction` are suitable for the
fallback. Conversion code SHALL explicitly detect overflow instead of
accepting `inf`. Use the existing Python 3.14 interpreter configured for this
repository. Test subnormal and tie-adjacent conversions on the supported
platform.

All new repository text files SHALL use CRLF line endings on Windows.

# 13. Reference pseudocode

## 13.1 Exact coefficient table

```text
function area_coefficients(p):
    require p >= 1
    table = []
    for a in 0 .. p:
        for b in a+1 .. p:
            numerator = (b-a) * C(p,a) * C(p,b)
            denominator = 2 * (2p-1) * C(2p-2,a+b-1)
            K = exact_rational(numerator, denominator)
            require K > 0
            table.append(a, b, K)
    return immutable(table)
```

## 13.2 Normalized composite area

```text
function normalized_area_exact(spans):
    total = exact_rational(0)
    for span in spans:
        C = exact_decode(span.position_controls)
        for (a, b, K) in area_coefficients(span.degree):
            total += K * det(C[a], C[b])

    for s in 0 .. span_count-1:
        e = exact_decode(spans[s].position_controls[-1])
        q = exact_decode(spans[(s+1) mod span_count].position_controls[0])
        total += det(e, q) / 2

    return total

function source_signed_area_exact(spans, H):
    return exact(H) * exact(H) * normalized_area_exact(spans)
```

The production wrapper first attempts the certified fast equivalent and
calls this exact function only when its rounding interval is ambiguous.

## 13.3 Turning number

```text
function turning_number(metric):
    for precision in [256, 512, 1024, 2048, 4096]:
        V_theta = 0
        E_theta = 0
        for cell in metric.phase_cells:
            if cell.constant_phase:
                continue
            wa = exact_bernstein_endpoint(cell, left)
            wb = exact_bernstein_endpoint(cell, right)
            X = dot(wa, wb)
            Y = det(wa, wb)
            require X != 0 or Y != 0
            (v, e) = atan2_ball(Y, X, precision)
            V_theta += 2*v
            E_theta += 2*e

        for cyclic join (left_span, right_span):
            ql = exact_square(left_span.preimage_at_right)
            qr = exact_square(right_span.preimage_at_left)
            X = dot(ql, qr)
            Y = det(ql, qr)
            require X > 0
            (v, e) = atan2_ball(Y, X, precision)
            V_theta += v
            E_theta += e

        (v_pi, e_pi) = pi_ball(precision)
        Q = interval_divide(
              [V_theta-E_theta, V_theta+E_theta],
              [2*(v_pi-e_pi), 2*(v_pi+e_pi)])
        integers = integers_contained_in(Q)
        if count(integers) == 1:
            return integers[0]

    raise NumericalPrecisionError(operation="area",
                                  quantity="turning number enclosure")
```

## 13.4 Closed offset area

```text
function closed_offset_signed_area(handle):
    cached = handle.area_cache
    if cached exists:
        return cached

    P = handle.area_provenance
    A0 = source_signed_area_exact(P.position_spans, P.scale)
    if P.distance == 0:
        result = correctly_rounded_finite_float(A0)
        publish_atomically_if_empty(result)
        return result

    L0 = exact_source_length(P.metric_source_state)
    nu = turning_number(P.metric)
    R = A0 - exact(P.distance) * L0
    C = exact(nu) * exact(P.distance) * exact(P.distance)
    result = round_rational_plus_pi(R, C,
               precision_ladder=[256, 512, 1024, 2048, 4096])
    publish_atomically_if_empty(result)
    return result
```

# 14. Required invariants and verification gates

## 14.1 Structural invariants

Before an area-capable object is published, existing construction already
verifies geometry. Area-specific wiring SHALL additionally ensure:

- only a closed topology type exposes the properties;
- source position span count is positive;
- each position array has shape `(p + 1, 2)` or an equivalent complex shape;
- all captured coefficients, $H$, widths, and $d$ are finite;
- $H>0$ and every source span width is positive;
- position and preimage span counts agree in offset provenance;
- preimage degree $m$ agrees with position degree $2m+1$;
- metric and area provenance refer to the same distance, scale, breakpoints,
  source span order, and closed topology; and
- the scalar area cache is empty or belongs to the exact immutable state or
  committed version that owns it.

These checks SHALL not calculate area.

## 14.2 Algebraic implementation checks

Tests SHALL independently verify for every supported degree:

- (5.1) equals (5.3) in exact rational arithmetic;
- every generated $K_{ab}^{(p)}$ is positive;
- degree 1 gives the shoelace segment coefficient $1/2$;
- degree 2 gives coefficients $1/3,1/6,1/3$;
- degree 3 gives (6.1);
- direct differentiation and exact power-basis integration agree with the
  Bernstein result; and
- subdividing a span by exact de Casteljau leaves its summed area unchanged.

## 14.3 Offset identity checks

For captured source state, an independent high-precision test oracle SHALL
differentiate and integrate the exact-reference offset expression

$$
z_d=z+dN_L
$$

on ordinary test cases and compare it with (8.1). This oracle is test-only;
it SHALL not become the production algorithm.

For public rounded NURBS arrays, high-precision rational-function integration
or sufficiently rigorous interval quadrature MAY be used as a secondary
representation check within the existing offset construction error budget.
It is not the normative exact-reference area.

# 15. Test specification

## 15.1 API and topology

Tests SHALL prove:

- both closed source families expose `signed_area` and `area`;
- a closed PH B-spline snapshot exposes both and preserves its version;
- closed offsets from both families return `ClosedNURBSHandle` and expose
  both;
- all open source types, common family bases, open snapshots, and open offset
  handles do not expose either member;
- `area == abs(signed_area)` including exact and underflowed zero; and
- static return annotations distinguish closed and open offset handles.

## 15.2 Exact elementary shapes and identities

Use stored coefficient fixtures whose results are known exactly:

- an oriented polygon represented by degree-1 internal fixtures to recover
  the shoelace formula;
- a cubic Bezier loop with rational controls and exact rational area;
- the same span before and after one or more exact midpoint subdivisions;
- translated, rotated by a quarter turn, uniformly scaled, reflected, and
  traversal-reversed versions; and
- cyclic seam rotations of the same composite chain.

Expected metamorphic laws are

$$
\begin{aligned}
A(r+a)&=A(r),\\
A(Rr)&=\det(R)A(r)\quad\text{for orthogonal }R,\\
A(\lambda r)&=\lambda^2A(r),\\
A(r\text{ reversed})&=-A(r).
\end{aligned}
$$

## 15.3 Cubic PH closed cases

Cover:

- strictly convex counterclockwise and clockwise cycles;
- general nonconvex cycles;
- straight/curved transitions;
- auxiliary inflections and curvature-sign changes;
- self-intersecting and opposite-lobe cancellation cases;
- the smallest accepted point count;
- many-segment closed examples; and
- seam locations at ordinary and auxiliary joins.

Compare with an independent 100-or-more-decimal-digit polynomial integral,
not with a sampled polygon as the acceptance oracle.

## 15.4 PH B-spline closed cases

Cover every supported preimage degree boundary, including the default and
the configured maximum. Cover multiple parameterizations and continuity
requests. Test:

- cold first query;
- warm repeated query;
- move, insert, delete, append/prepend as applicable to closed topology, and
  multi-operation transaction;
- strict-local, expanding, and global repair paths;
- edits of the seam point;
- unchanged span-cache reuse after a local edit;
- no reuse after a frame-changing global rebuild;
- failed edit retains the prior cached result; and
- a pre-edit snapshot retains the pre-edit area.

Instrumentation SHALL prove that construction and edit commit call no area
kernel. After a local edit following one prior area query, instrumentation
SHOULD show that only replaced span contributions are recalculated.

## 15.5 Offset cases

For each source family, test:

- `d == 0`, with bitwise equality to source `signed_area`;
- positive and negative cusp-free distances;
- distance exactly at a certified minimum curvature radius where feasible;
- distances beyond cusp creation;
- offsets with loops, reversals, and self-intersections;
- clockwise and counterclockwise sources;
- a source with turning number zero;
- a source with `abs(turning_number) > 1` if constructible by the package;
- a source with straight spans and zero-phase cells;
- exact preimage gauge sign changes and verified ulp-scale one-sided tangent
  mismatches, proving that cyclic join corrections recover the intended
  integer;
- positive and negative $d$ under traversal reversal; and
- offset handles retained after later source edits.

For every case, verify

$$
A(d_2)-A(d_1)
=-(d_2-d_1)L_0+\pi\nu(d_2^2-d_1^2)
$$

with an independent high-precision oracle.

For a counterclockwise circle-like source of radius $R$, positive left-normal
distance is inward and the limiting analytic check is

$$
A_d\approx\pi(R-d)^2.
$$

For clockwise traversal it is

$$
A_d\approx-\pi(R+d)^2.
$$

These two tests catch both common sign errors.

## 15.6 Numerical stress

The suite SHALL include:

- user-coordinate translations near the largest magnitude accepted by
  existing construction while the shape remains small;
- scales spanning at least `1e-150` through the largest constructible scale;
- subnormal exact areas;
- an area whose correctly rounded value is zero;
- nearly equal positive and negative lobes;
- offset distances chosen so $A_0$, $dL_0$, and $\pi\nu d^2$ nearly cancel;
- coefficient products that overflow binary64 on a naive unnormalized or
  unscaled path while the final area is finite;
- a finite curve whose exact area exceeds binary64 range and must raise
  `NumericalPrecisionError` rather than return infinity;
- rounding-boundary cases immediately above and below a binary64 midpoint;
- degree-table cache races; and
- concurrent first area queries, plus a concurrent PH B-spline edit/query.

Tests SHALL run with warnings promoted to errors, consistent with the current
pytest configuration.

## 15.7 Serialization

Round-trip each area-capable type before and after its first area query.
Verify:

- geometry and area are unchanged;
- deserialization does not execute an area calculation;
- an omitted cache is rebuilt only on the next query;
- corrupted offset provenance is rejected by existing structured numerical
  or construction errors; and
- restored read-only arrays remain read-only.

# 16. Performance and complexity

Let $S$ be the compiled source span count and $p$ the position degree.

| Operation | Required or target complexity |
|---|---:|
| First cubic closed source area | $O(S)$ time, $O(1)$ work memory apart from certification fallback |
| First PH B-spline closed source area | $O(Sp^2)$ time, $O(p^2)$ degree-table memory |
| Repeated source query | $O(1)$ |
| First query after local edit with $K$ changed spans | $O(Kp^2+S)$ reference target, with unchanged span reuse |
| First closed offset area | $O(Sp^2+C)$ plus certified transcendental work, where $C$ is the existing phase-cell count |
| Repeated closed offset query | $O(1)$ |

No complexity term may depend on a sampling tolerance or tessellation count.

Benchmarks SHALL report cold, warm, and post-local-edit timings separately for
both families and closed offsets. They SHALL include span counts of at least
10, 100, and 1,000 and several supported PH B-spline degrees. They SHALL also
report:

- percentage of queries accepted by the fast path;
- exact fallback count;
- maximum phase precision used;
- number of reused and recomputed span contributions after edits; and
- peak additional memory for offset area provenance.

Performance optimization SHALL not weaken the certified publication gate.
The exact fallback is expected to be rare for ordinary simple curves and
deliberately common in adversarial cancellation tests.

# 17. Nonconforming shortcuts

The following implementations are explicitly forbidden:

- adding area properties to open types and raising an error when called;
- closing an open spline implicitly;
- flattening the curve and applying the polygon shoelace formula;
- adaptive or fixed Gaussian quadrature in the production source-area path;
- generic numerical integration of the rational offset NURBS;
- using public rounded offset control points to rediscover source PH data;
- using `handle.length` instead of $L_0$ in (8.1);
- assuming $\nu=\operatorname{sign}(A_0)$ or $\nu=\pm1$;
- estimating $\nu$ from tangent samples;
- dropping the $\pi\nu d^2$ term after offset cusps appear;
- taking absolute values per span or per lobe before summation;
- accumulating determinants formed in large translated user coordinates;
- calculating area during construction or edit commit;
- retaining a whole-result cache after a version change;
- sharing a mutable cache between a live editable spline and a snapshot;
- accepting a result by a fixed epsilon test; or
- returning NaN, infinity, or an uncertified midpoint after precision
  exhaustion.

# 18. Developer handoff sequence

Implementation SHOULD proceed in this order so each stage has an independent
acceptance gate:

1. Add exact coefficient-table tests and the private source-area kernel.
   Verify (5.1), (5.2), cubic formula (6.1), subdivision invariance, and exact
   rational fixtures.
2. Add the certified double-double ball fast path and exact fallback. Verify
   correct rounding, cancellation, exponent extremes, and fallback forcing.
3. Add closed-only properties and lazy caches to `CubicPHSplineClosed` and
   `PHBSplineClosed`. Verify no open type has the interface.
4. Add version-safe per-span reuse and closed snapshot isolation. Verify edit
   and concurrency behavior.
5. Factor the exact preimage phase-cell view from `offset_metric.py` and add
   turning-number certification. Verify integer isolation for several
   rotation indices.
6. Add immutable offset area provenance and `ClosedNURBSHandle`. Verify that
   offset construction performs no area calculation.
7. Implement exact $L_0$ extraction and certified $R+C\pi$ evaluation. Verify
   `d == 0`, both offset signs, cusp and beyond-cusp cases.
8. Add persistence, typing, documentation, full regression tests, and
   benchmarks.

At every stage, all pre-existing tests SHALL continue to pass. The change
SHALL not alter point, derivative, curvature, distance, offset geometry,
editing, or existing serialization results except for the intentional new
closed-handle runtime subtype and its serialized type tag.

# 19. Acceptance checklist

The implementation is complete only when all answers below are yes:

1. Do only closed package types expose `signed_area` and `area`?
2. Is `signed_area` the algebraic Green line integral and `area` its absolute
   value?
3. Are source areas analytic Bernstein coefficient sums with residual join
   closure and no quadrature?
4. Does the cubic implementation use the exact six-determinant formula or an
   identical degree-3 table?
5. Does the PH B-spline implementation support every package degree and
   reuse retained span work only after demand?
6. Is every whole result lazy and version-safe?
7. Does a closed snapshot retain its captured result after source edits?
8. Does a closed offset use $A_0-dL_0+\pi\nu d^2$ with the captured source
   length, never the offset length?
9. Is $\nu$ a certified integer derived from exact PH phase cells rather than
   an assumption or sample?
10. Do cusp-forming and self-intersecting offsets retain a defined algebraic
    area?
11. Are determinant sums performed in normalized coordinates?
12. Does a certified fast path fall back to exact rational/adaptive arithmetic
    whenever cancellation or rounding is unresolved?
13. Are overflow, underflow, zero, and rounding-boundary cases handled by a
    documented representability rule?
14. Are cache publication, edit races, snapshots, copies, and pickle restore
    safe?
15. Do tests independently verify the equations and prove that no constructor
    or edit computes area?

# 20. References and relationship to repository specifications

This document is an area addendum to:

- `CubicPHSpline_Technical_Specification.md`, especially its normalized
  cubic PH representation and exact-offset sections;
- `PHBSpline_Technical_Specification.md`, especially its Bernstein preimage,
  analytic speed, editing, snapshot, and exact-offset sections; and
- `OffsetNURBS_Distance_Specification.md`, especially its exact-reference
  model, exact rational PH metric state, phase-lift proof, double-double fast
  path, and adaptive certified transcendental arithmetic.

Mathematical background:

1. R. T. Farouki, *Pythagorean-Hodograph Curves: Algebra and Geometry
   Inseparable*, Springer, 2008, especially the planar PH, polynomial speed,
   arc-length, and rational-offset chapters.
2. G. Albrecht, C. V. Beccari, J.-C. Canonne, and L. Romani,
   “Pythagorean-Hodograph B-Spline Curves,” 2016,
   <https://arxiv.org/abs/1609.07888>.
3. S. Lanzat and M. Polyak, “Integrating curvature: from Umlaufsatz to
   $J^+$ invariant,” 2011, <https://arxiv.org/abs/1108.4288>, for the total
   signed curvature and rotation-number relation used in (8.1).
4. R. T. Farouki, “Introduction to Pythagorean-hodograph curves,” UC Davis
   lecture notes, for the exact polynomial-speed and rational-offset
   properties,
   <https://faculty.engineering.ucdavis.edu/farouki/wp-content/uploads/sites/51/2021/07/Introduction-to-PH-curves.pdf>.

The exact Bernstein area formulas and the left-normal sign form of (8.1) are
fully derived in this document. No external formula adaptation is left to the
implementing developer.
