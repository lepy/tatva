import jax
import jax.numpy as jnp
import numpy as np

from tatva import Mesh, Operator, element
from tatva.material import (
    HSPlaneStressMaterial,
    evaluate_plane_stress_responses,
    hockett_sherby_yield_stress,
    hockett_sherby_yield_stress_slope,
    make_initial_state,
    make_initial_state_field,
    make_plane_stress_residual,
    make_plane_stress_residual_flat,
    make_plane_stress_tangent_flat,
    plane_stress_internal_virtual_work,
    plastic_log_strain_components,
    update_plane_stress,
)

jax.config.update("jax_enable_x64", True)


def test_hockett_sherby_is_monotone_and_starts_at_sigma0():
    material = HSPlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=50.0,
        sigma_inf=120.0,
        m=12.0,
        n=0.6,
    )
    ep = jnp.linspace(0.0, 0.5, 6)
    sigma_y = hockett_sherby_yield_stress(ep, material)
    slope = hockett_sherby_yield_stress_slope(ep[1:], material)

    np.testing.assert_allclose(sigma_y[0], material.sigma0, rtol=1e-8, atol=1e-8)
    assert np.all(np.diff(np.asarray(sigma_y)) > 0.0)
    assert np.all(np.asarray(slope) > 0.0)


def test_plane_stress_elastic_step_keeps_plastic_state_zero():
    material = HSPlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=1_000.0,
        sigma_inf=1_100.0,
        m=5.0,
        n=1.0,
    )
    state0 = make_initial_state()
    F2 = jnp.array([[1.01, 0.0], [0.0, 0.99]], dtype=jnp.float64)

    response = update_plane_stress(F2, state0, material)

    np.testing.assert_allclose(response.tau[2, 2], 0.0, atol=1e-7)
    np.testing.assert_allclose(response.state.epbar, 0.0, atol=1e-12)
    np.testing.assert_allclose(response.state.Epl, np.zeros((3, 3)), atol=1e-12)
    assert not bool(response.yielded)


def test_plane_stress_plastic_step_updates_epbar_and_Epl_components():
    material = HSPlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=5.0,
        sigma_inf=120.0,
        m=18.0,
        n=0.7,
    )
    state0 = make_initial_state()
    F2 = jnp.array([[1.35, 0.0], [0.0, 1.0]], dtype=jnp.float64)

    response = update_plane_stress(F2, state0, material)
    comps = plastic_log_strain_components(response.state)

    np.testing.assert_allclose(response.tau[2, 2], 0.0, atol=1e-6)
    assert bool(response.yielded)
    assert float(response.delta_gamma) > 0.0
    assert float(response.state.epbar) > 0.0
    assert abs(float(comps["Ep33"])) > 1e-8
    np.testing.assert_allclose(
        response.state.Epl,
        np.asarray(response.state.Epl).T,
        atol=1e-10,
    )
    np.testing.assert_allclose(jnp.linalg.det(response.state.Fp), 1.0, atol=1e-6)


def test_global_plane_stress_response_and_residual_pipeline():
    coords = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    elements_ = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    op = Operator(Mesh(coords=coords, elements=elements_), element.Tri3())

    material = HSPlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=15.0,
        sigma_inf=120.0,
        m=18.0,
        n=0.7,
    )
    state0 = make_initial_state_field(op)
    u = jnp.zeros((coords.shape[0], 2), dtype=jnp.float64)
    test_u = jnp.ones_like(u)

    responses = evaluate_plane_stress_responses(op, u, state0, material)
    virtual_work = plane_stress_internal_virtual_work(op, test_u, u, state0, material)
    residual = make_plane_stress_residual(op, u.shape)(u, state0, material)
    residual_flat = make_plane_stress_residual_flat(op)(u.reshape(-1), state0, material)
    tangent = make_plane_stress_tangent_flat(op)(u.reshape(-1), state0, material)

    np.testing.assert_allclose(responses.tau[..., 2, 2], 0.0, atol=1e-7)
    np.testing.assert_allclose(virtual_work, 0.0, atol=1e-12)
    np.testing.assert_allclose(residual, np.zeros_like(u), atol=1e-12)
    np.testing.assert_allclose(
        residual_flat, np.zeros(coords.shape[0] * 2, dtype=np.float64), atol=1e-12
    )
    assert tangent.shape == (coords.shape[0] * 2, coords.shape[0] * 2)
