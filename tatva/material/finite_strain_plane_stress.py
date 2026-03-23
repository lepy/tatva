from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from tatva.operator import Operator
from tatva.utils import virtual_work_to_residual


class HSPlaneStressState(eqx.Module):
    """Internal state for finite-strain plane-stress J2 plasticity."""

    Fp: Array
    epbar: Array
    Epl: Array
    lam3: Array


class HSPlaneStressMaterial(eqx.Module):
    """Finite-strain J2 material with Hockett-Sherby hardening."""

    mu: float
    kappa: float
    sigma0: float
    sigma_inf: float
    m: float
    n: float
    newton_tol: float = 1e-10
    newton_maxiter: int = 25
    plane_stress_eps: float = 1e-6


class HSPlaneStressResponse(eqx.Module):
    """Local material response for one quadrature point."""

    P: Array
    P2: Array
    tau: Array
    F: Array
    state: HSPlaneStressState
    delta_gamma: Array
    yielded: Array


def make_initial_state(dtype: jnp.dtype = jnp.float64) -> HSPlaneStressState:
    """Return an undeformed initial state for one quadrature point."""

    I = jnp.eye(3, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)
    return HSPlaneStressState(Fp=I, epbar=zero, Epl=jnp.zeros((3, 3), dtype=dtype), lam3=jnp.array(1.0, dtype=dtype))


def make_initial_state_field(
    op: Operator, dtype: jnp.dtype = jnp.float64
) -> HSPlaneStressState:
    """Return an undeformed state field matching an operator's element/quad layout."""

    n_el = op.mesh.elements.shape[0]
    n_qp = op.element.quad_points.shape[0]
    single = make_initial_state(dtype=dtype)
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_el, n_qp, *x.shape)).copy(),
        single,
    )


def hockett_sherby_yield_stress(epbar: Array, material: HSPlaneStressMaterial) -> Array:
    ep = jnp.maximum(epbar, 0.0)
    saturation = jnp.exp(-material.m * ep**material.n)
    return material.sigma_inf - (material.sigma_inf - material.sigma0) * saturation


def hockett_sherby_yield_stress_slope(
    epbar: Array, material: HSPlaneStressMaterial
) -> Array:
    ep = jnp.maximum(epbar, 1e-12)
    prefactor = (material.sigma_inf - material.sigma0) * jnp.exp(
        -material.m * ep**material.n
    )
    return prefactor * material.m * material.n * ep ** (material.n - 1.0)


def plastic_log_strain_components(state: HSPlaneStressState) -> dict[str, Array]:
    """Return the most relevant plane-stress plastic logarithmic strain components."""

    return {
        "Ep11": state.Epl[0, 0],
        "Ep22": state.Epl[1, 1],
        "Ep33": state.Epl[2, 2],
        "Ep12": state.Epl[0, 1],
    }


def deformation_gradient_from_displacement_gradient(grad_u: Array) -> Array:
    """Return the in-plane deformation gradient from the in-plane displacement gradient."""

    return jnp.eye(2, dtype=grad_u.dtype) + grad_u


def evaluate_plane_stress_responses(
    op: Operator,
    u: Array,
    state_n: HSPlaneStressState,
    material: HSPlaneStressMaterial,
) -> HSPlaneStressResponse:
    """Evaluate local plane-stress responses at all elements and quadrature points.

    `state_n` is the committed state from the previous increment; the returned
    ``response.state`` is the trial state for the current displacement field.
    """

    grad_u = op.grad(u)
    F2 = jax.vmap(
        jax.vmap(deformation_gradient_from_displacement_gradient, in_axes=0),
        in_axes=0,
    )(grad_u)
    update_qp = lambda F2_qp, state_qp: update_plane_stress(F2_qp, state_qp, material)
    update_fn = jax.vmap(jax.vmap(update_qp, in_axes=(0, 0)), in_axes=(0, 0))
    return update_fn(F2, state_n)


def plane_stress_internal_virtual_work(
    op: Operator,
    test_u: Array,
    u: Array,
    state_n: HSPlaneStressState,
    material: HSPlaneStressMaterial,
) -> Array:
    """Return the internal virtual work for a plane-stress finite-strain material."""

    responses = evaluate_plane_stress_responses(op, u, state_n, material)
    grad_test = op.grad(test_u)
    density = jnp.einsum("eqij,eqij->eq", responses.P2, grad_test)
    return op.integrate(density)


def make_plane_stress_residual(
    op: Operator, u_shape: tuple[int, ...], *, jit: bool = False
):
    """Return a residual function ``r(u, state_n, material)`` for global Newton solves."""

    @virtual_work_to_residual(test_shape=u_shape, jit=jit)
    def _virtual_work(
        test_u: Array,
        u: Array,
        state_n: HSPlaneStressState,
        material: HSPlaneStressMaterial,
    ) -> Array:
        return plane_stress_internal_virtual_work(op, test_u, u, state_n, material)

    return _virtual_work


def make_plane_stress_residual_flat(
    op: Operator, *, n_dim: int = 2, jit: bool = False
):
    """Return a flattened residual function for Newton/Krylov solvers."""

    u_shape = (op.mesh.coords.shape[0], n_dim)
    residual = make_plane_stress_residual(op, u_shape, jit=jit)

    def _residual_flat(
        u_flat: Array,
        state_n: HSPlaneStressState,
        material: HSPlaneStressMaterial,
    ) -> Array:
        u = u_flat.reshape(u_shape)
        return residual(u, state_n, material).reshape(-1)

    if jit:
        _residual_flat_jit = jax.jit(_residual_flat)
        return _residual_flat_jit
    return _residual_flat


def make_plane_stress_tangent_flat(
    op: Operator, *, n_dim: int = 2, jit: bool = False
):
    """Return a tangent function ``K(u_flat, state_n, material)`` via forward AD."""

    residual_flat = make_plane_stress_residual_flat(op, n_dim=n_dim, jit=jit)
    tangent = jax.jacfwd(residual_flat, argnums=0)
    if jit:
        tangent = jax.jit(tangent)
    return tangent


def update_plane_stress(
    F2: Array, state_n: HSPlaneStressState, material: HSPlaneStressMaterial
) -> HSPlaneStressResponse:
    """Update one quadrature point for finite-strain plane-stress J2 plasticity.

    Args:
        F2: In-plane deformation gradient shaped ``(2, 2)``.
        state_n: State at the end of the previous converged increment.
        material: Material parameters.
    """

    lam3 = _solve_plane_stress_lambda(F2, state_n, material)
    return _update_fixed_lambda(F2, state_n, lam3, material)


def _solve_plane_stress_lambda(
    F2: Array, state_n: HSPlaneStressState, material: HSPlaneStressMaterial
) -> Array:
    def residual(lam3: Array) -> Array:
        return _update_fixed_lambda(F2, state_n, lam3, material).tau[2, 2]

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        i, lam3, _ = carry
        r = residual(lam3)
        dr = _finite_difference_derivative(residual, lam3, material.plane_stress_eps)
        lam3_new = jnp.maximum(lam3 - r / (dr + 1e-14), 1e-8)
        err = jnp.abs(lam3_new - lam3)
        return i + 1, lam3_new, err

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        i, _, err = carry
        return (i < material.newton_maxiter) & (err > material.newton_tol)

    init = (
        jnp.array(0, dtype=jnp.int32),
        jnp.asarray(state_n.lam3, dtype=F2.dtype),
        jnp.asarray(jnp.inf, dtype=F2.dtype),
    )
    _, lam3, _ = jax.lax.while_loop(cond, body, init)
    return lam3


def _update_fixed_lambda(
    F2: Array, state_n: HSPlaneStressState, lam3: Array, material: HSPlaneStressMaterial
) -> HSPlaneStressResponse:
    F = _embed_deformation_gradient(F2, lam3)
    Fp_n = state_n.Fp
    Fe_trial = F @ jnp.linalg.inv(Fp_n)
    Ce_trial = _sym(Fe_trial.T @ Fe_trial)
    ee_trial = 0.5 * _spd_log(Ce_trial)

    tau_trial = _kirchhoff_stress_from_log_strain(ee_trial, material)
    s_trial = _dev(tau_trial)
    q_trial = jnp.sqrt(1.5) * _tensor_norm(s_trial)
    f_trial = q_trial - hockett_sherby_yield_stress(state_n.epbar, material)

    def elastic_branch(_: None) -> HSPlaneStressResponse:
        return _build_response(
            F=F,
            tau=tau_trial,
            Fp=Fp_n,
            epbar=state_n.epbar,
            lam3=lam3,
            delta_gamma=jnp.array(0.0, dtype=F.dtype),
            yielded=jnp.array(False),
        )

    def plastic_branch(_: None) -> HSPlaneStressResponse:
        delta_gamma = _solve_delta_gamma(q_trial, state_n.epbar, material)
        factor = jnp.maximum(
            1.0 - 3.0 * material.mu * delta_gamma / (q_trial + 1e-30), 0.0
        )
        s_new = factor * s_trial
        mean_tau = jnp.trace(tau_trial) / 3.0
        tau_new = s_new + mean_tau * _eye3(F.dtype)

        ee_dev_new = s_new / (2.0 * material.mu)
        ee_vol_new = (jnp.trace(ee_trial) / 3.0) * _eye3(F.dtype)
        ee_new = ee_dev_new + ee_vol_new

        Ue_new = _spd_exp(ee_new)
        Re_trial = _polar_rotation(Fe_trial)
        Fe_new = Re_trial @ Ue_new
        Fp_new = jnp.linalg.inv(Fe_new) @ F
        Fp_new = _isochoric_project(Fp_new)
        epbar_new = state_n.epbar + delta_gamma

        return _build_response(
            F=F,
            tau=tau_new,
            Fp=Fp_new,
            epbar=epbar_new,
            lam3=lam3,
            delta_gamma=delta_gamma,
            yielded=jnp.array(True),
        )

    return jax.lax.cond(f_trial <= 0.0, elastic_branch, plastic_branch, operand=None)


def _build_response(
    *,
    F: Array,
    tau: Array,
    Fp: Array,
    epbar: Array,
    lam3: Array,
    delta_gamma: Array,
    yielded: Array,
) -> HSPlaneStressResponse:
    Cp = _sym(Fp.T @ Fp)
    Epl = 0.5 * _spd_log(Cp)
    P = tau @ jnp.linalg.inv(F).T
    state = HSPlaneStressState(Fp=Fp, epbar=epbar, Epl=Epl, lam3=lam3)
    return HSPlaneStressResponse(
        P=P,
        P2=P[:2, :2],
        tau=tau,
        F=F,
        state=state,
        delta_gamma=delta_gamma,
        yielded=yielded,
    )


def _solve_delta_gamma(
    q_trial: Array, epbar_n: Array, material: HSPlaneStressMaterial
) -> Array:
    f_trial = q_trial - hockett_sherby_yield_stress(epbar_n, material)
    slope0 = hockett_sherby_yield_stress_slope(epbar_n, material)
    dg0 = jnp.maximum(f_trial / (3.0 * material.mu + slope0 + 1e-12), 0.0)

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        i, _, err = carry
        return (i < material.newton_maxiter) & (err > material.newton_tol)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        i, dg, _ = carry
        yield_value = hockett_sherby_yield_stress(epbar_n + dg, material)
        slope = hockett_sherby_yield_stress_slope(epbar_n + dg, material)
        g = q_trial - 3.0 * material.mu * dg - yield_value
        gp = -3.0 * material.mu - slope
        dg_new = jnp.maximum(dg - g / (gp + 1e-14), 0.0)
        err = jnp.abs(dg_new - dg)
        return i + 1, dg_new, err

    init = (
        jnp.array(0, dtype=jnp.int32),
        jnp.asarray(dg0, dtype=jnp.result_type(q_trial, epbar_n)),
        jnp.asarray(jnp.inf, dtype=jnp.result_type(q_trial, epbar_n)),
    )
    _, delta_gamma, _ = jax.lax.while_loop(cond, body, init)
    return delta_gamma


def _kirchhoff_stress_from_log_strain(
    ee: Array, material: HSPlaneStressMaterial
) -> Array:
    return (
        2.0 * material.mu * _dev(ee)
        + material.kappa * jnp.trace(ee) * _eye3(ee.dtype)
    )


def _embed_deformation_gradient(F2: Array, lam3: Array) -> Array:
    return jnp.array(
        [
            [F2[0, 0], F2[0, 1], 0.0],
            [F2[1, 0], F2[1, 1], 0.0],
            [0.0, 0.0, lam3],
        ],
        dtype=F2.dtype,
    )


def _finite_difference_derivative(fun, x: Array, eps: float) -> Array:
    step = jnp.asarray(eps, dtype=x.dtype)
    return (fun(x + step) - fun(x - step)) / (2.0 * step)


def _sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


def _eye3(dtype: jnp.dtype) -> Array:
    return jnp.eye(3, dtype=dtype)


def _dev(A: Array) -> Array:
    return A - jnp.trace(A) / 3.0 * _eye3(A.dtype)


def _tensor_norm(A: Array) -> Array:
    return jnp.sqrt(jnp.sum(A * A) + 1e-30)


def _polar_rotation(F: Array) -> Array:
    U, _, Vh = jnp.linalg.svd(F, full_matrices=False)
    R = U @ Vh
    detR = jnp.linalg.det(R)
    fix = jnp.diag(jnp.array([1.0, 1.0, jnp.sign(detR)], dtype=F.dtype))
    return U @ fix @ Vh


def _spd_log(A: Array) -> Array:
    vals, vecs = jnp.linalg.eigh(_sym(A))
    vals = jnp.clip(vals, 1e-30, None)
    return vecs @ jnp.diag(jnp.log(vals)) @ vecs.T


def _spd_exp(A: Array) -> Array:
    vals, vecs = jnp.linalg.eigh(_sym(A))
    return vecs @ jnp.diag(jnp.exp(vals)) @ vecs.T


def _isochoric_project(Fp: Array) -> Array:
    det_fp = jnp.linalg.det(Fp)
    return Fp / jnp.cbrt(jnp.abs(det_fp) + 1e-30)
