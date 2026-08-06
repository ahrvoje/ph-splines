# PH B-spline examples

- `generate_referenced_examples.py` renders PHBSpline versions of all 224 referenced input-gallery cases.
- `generate_features.py` renders 128 additional examples in eight feature families: prescribed continuity, closed distance stations, local move/insert/delete, parameter derivatives, tangent/curvature-vector geometry, and curvature-vector derivatives.

Run either generator from the repository root with the project’s `examples`
dependencies installed. The referenced cases are written directly to `base/`,
`nonconvex/`, `pathological/`, and `near_break/`; B-spline-only cases remain in
`features/`.
