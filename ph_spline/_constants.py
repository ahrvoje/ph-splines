"""Shared numerical constants (private).

All tolerances follow the technical specification.  Every constant is a
plain module-level float so the whole package is deterministic and free of
runtime configuration.
"""

from __future__ import annotations

import math
import sys

#: binary64 machine epsilon (2^-52).
EPS: float = sys.float_info.epsilon

#: Smallest positive normal double, used as a clip floor inside guarded
#: solver arithmetic (never in strict verification arithmetic).
TINY: float = sys.float_info.min

#: Uniqueness bound for adjacent turn-angle sums:  pi + arccos(1/sqrt(3)).
THETA_UNIQUE: float = math.pi + math.acos(1.0 / math.sqrt(3.0))

#: Angular safety margin for the boundary-angle clamp (spec section 7.2).
DELTA_THETA: float = max(1024.0 * EPS, 1e-12)

#: Normalized-cross threshold below which a turn is numerically zero
#: (collinearity classification, spec section 6.2).
COLLINEAR_EPS: float = 1024.0 * EPS

#: Minimum admissible chord-length ratio min|dP| / max|dP| (spec section 5).
CHORD_RATIO_MIN: float = 1024.0 * EPS

#: Tangent-variable box bound: x in [X_MIN, 1 - X_MIN] (spec section 9.1).
X_MIN: float = 64.0 * EPS

#: Solver acceptance bound on the logarithmic curvature residual (9.6).
F_TOL: float = 1e-11

#: Relative regularity margin sigma_min / max(sigma(0), sigma(1)) (10.1).
RHO_MIN: float = 1e-12

#: G2 verification tolerances (spec section 10.3).
EPS_TANGENT: float = 1e-12
EPS_KAPPA: float = 1e-10

#: Normalized PH reconstruction tolerance (spec section 4.2).
RECON_TOL: float = 1e-10

#: Below this threshold the depressed-cubic scaled parameter G is treated as
#: too small for the hyperbolic arc-length inverse and the stable scaled
#: Cardano form is used instead (spec section 13.4).
G_HYPERBOLIC_MIN: float = 1e-90

#: Ulp slack applied when clamping parameters that fall marginally outside
#: their domain because of prior floating-point arithmetic (spec 15.1).
ULP_SLACK: float = 4.0
