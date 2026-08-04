"""Bounded tridiagonal G2 solve (specification section 9).

The unknowns are the internal tangent fractions ``x_i`` in ``(0, 1)``.  The
residual is the dimensionless logarithmic curvature mismatch at every
internal point.  The Jacobian is tridiagonal and is computed by
complex-step differentiation of the branch-free smooth residual core using
three-coloring, which yields derivatives accurate to machine precision with
three residual evaluations.

The solver is SciPy's deterministic trust-region reflective least-squares
with box bounds; its success flag is never trusted -- the caller performs an
independent strict acceptance pass.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import least_squares
from scipy.sparse import diags_array

from cubic_ph_spline._constants import F_TOL, TINY, X_MIN
from cubic_ph_spline.construction import edge_quantities, segment_alpha_beta
from cubic_ph_spline.exceptions import SplineConvergenceError

__all__ = ["solve_internal_tangents"]

#: Complex-step size.  Far below the square root of the smallest relative
#: feature of the residual, so the imaginary part carries a pure derivative
#: with no subtractive cancellation.
_CS_H = 1e-200

#: Above this system size the Jacobian is assembled as a sparse banded
#: matrix and the trust-region subproblem uses LSMR.
_SPARSE_THRESHOLD = 256


def _guarded_log(z: np.ndarray) -> np.ndarray:
    """Logarithm clipped at the smallest positive normal double.

    Where the clip activates, the input was outside the admissible region;
    the derivative is deliberately killed there so that complex-step rows
    remain finite.
    """
    if np.iscomplexobj(z):
        return np.where(
            z.real > TINY, np.log(np.where(z.real > TINY, z, 1.0)), np.log(TINY)
        )
    return np.log(np.maximum(z, TINY))


def _residual_core(
    x: np.ndarray, phi: np.ndarray, phi0: float, phim: float, lhat: np.ndarray
) -> np.ndarray:
    """Guarded logarithmic curvature residuals; real or complex ``x``."""
    alpha, beta = segment_alpha_beta(x, phi, phi0, phim)
    _, _, lam0, lam1, beta1, _ = edge_quantities(alpha, beta, lhat, guarded=True)
    log_sin = _guarded_log(np.sin(beta1))
    log_l0 = _guarded_log(lam0)
    log_l1 = _guarded_log(lam1)
    log_k_end = log_sin + 0.5 * log_l0 - 1.5 * log_l1
    log_k_start = log_sin + 0.5 * log_l1 - 1.5 * log_l0
    return log_k_end[:-1] - log_k_start[1:]


def _jacobian_bands(
    x: np.ndarray,
    phi: np.ndarray,
    phi0: float,
    phim: float,
    lhat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal Jacobian bands via three-color complex step."""
    n = x.shape[0]
    main = np.zeros(n)
    sub = np.zeros(max(n - 1, 0))
    sup = np.zeros(max(n - 1, 0))
    for color in range(3):
        idx = np.arange(color, n, 3)
        if idx.size == 0:
            continue
        xc = x.astype(np.complex128)
        xc[idx] += 1j * _CS_H
        im = np.imag(_residual_core(xc, phi, phi0, phim, lhat)) / _CS_H
        main[idx] = im[idx]
        left = idx[idx >= 1]
        sup[left - 1] = im[left - 1]  # J[j-1, j]
        right = idx[idx <= n - 2]
        sub[right] = im[right + 1]  # J[j+1, j]
    return main, sub, sup


def _newton_polish(
    x: np.ndarray,
    phi: np.ndarray,
    phi0: float,
    phim: float,
    lhat: np.ndarray,
) -> np.ndarray:
    """Damped, box-projected Newton refinement of the TRF iterate.

    The trust-region stage globalizes; its vector-norm-scaled stopping
    tests cannot resolve solution components of ~1e-8 to the residual
    gate.  This terminal stage solves the exact tridiagonal Newton system
    (banded LU) with backtracking and step rejection; every accepted step
    must strictly reduce the residual norm.  Deterministic; at most a few
    residual evaluations.
    """
    n = x.shape[0]

    def resid(v: np.ndarray) -> np.ndarray:
        r = _residual_core(v, phi, phi0, phim, lhat)
        return np.asarray(r.real if np.iscomplexobj(r) else r, dtype=np.float64)

    f = resid(x)
    f_norm = float(np.max(np.abs(f))) if f.size else 0.0
    for _ in range(30):
        if not np.all(np.isfinite(f)) or f_norm <= 0.25 * F_TOL:
            break
        main, sub, sup = _jacobian_bands(x, phi, phi0, phim, lhat)
        ab = np.zeros((3, n))
        ab[1, :] = main
        if n > 1:
            ab[0, 1:] = sup
            ab[2, :-1] = sub
        try:
            with np.errstate(all="ignore"):
                step = solve_banded((1, 1), ab, -f)
        except (ValueError, np.linalg.LinAlgError):
            break
        if not np.all(np.isfinite(step)):
            break
        accepted = False
        damping = 1.0
        for _ in range(12):
            x_new = np.clip(x + damping * step, X_MIN, 1.0 - X_MIN)
            f_new = resid(x_new)
            if np.all(np.isfinite(f_new)):
                f_new_norm = float(np.max(np.abs(f_new)))
                if f_new_norm < f_norm:
                    x, f, f_norm = x_new, f_new, f_new_norm
                    accepted = True
                    break
            damping *= 0.25
        if not accepted:
            break
    return x


def solve_internal_tangents(
    lhat: np.ndarray,
    phi: np.ndarray,
    phi0: float,
    phim: float,
    x0: np.ndarray,
) -> np.ndarray:
    """Solve for the internal tangent fractions inside their box.

    Returns the solver iterate; strict acceptance is performed by the
    caller (spec section 9.6).  Raises :class:`SplineConvergenceError` only
    if the optimizer itself fails to produce an evaluable iterate.
    """
    n = x0.shape[0]

    def fun(x: np.ndarray) -> np.ndarray:
        r = _residual_core(x, phi, phi0, phim, lhat)
        return np.asarray(r.real if np.iscomplexobj(r) else r, dtype=np.float64)

    def jac(x: np.ndarray):
        main, sub, sup = _jacobian_bands(x, phi, phi0, phim, lhat)
        if n > _SPARSE_THRESHOLD:
            return diags_array([sub, main, sup], offsets=[-1, 0, 1], format="csr")
        J = np.diag(main)
        if n > 1:
            J += np.diag(sub, -1) + np.diag(sup, 1)
        return J

    try:
        result = least_squares(
            fun,
            x0,
            jac=jac,
            bounds=(X_MIN, 1.0 - X_MIN),
            method="trf",
            x_scale="jac",
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
            max_nfev=500,
            tr_solver="lsmr" if n > _SPARSE_THRESHOLD else "exact",
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise SplineConvergenceError(
            f"Nonlinear G2 solver failed to iterate: {exc}",
            quantity="least_squares",
        ) from exc
    x = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise SplineConvergenceError(
            "Nonlinear G2 solver produced a nonfinite iterate",
            quantity="x",
        )
    return _newton_polish(x, phi, phi0, phim, lhat)
