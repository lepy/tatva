import jax
import jax.numpy as jnp
import numpy as np

from tatva import Mesh, Operator, element
from tatva.material import (
    Hill48PlaneStressMaterial,
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
    update_plane_stress_batch,
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


def test_update_plane_stress_batch_matches_scalar_update():
    material = HSPlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=5.0,
        sigma_inf=120.0,
        m=18.0,
        n=0.7,
    )
    state0 = make_initial_state()
    F2_single = jnp.array([[1.12, 0.01], [0.0, 0.97]], dtype=jnp.float64)
    F2_batch = F2_single[None, None, :, :]
    state_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], state0)

    scalar = update_plane_stress(F2_single, state0, material)
    batch = update_plane_stress_batch(F2_batch, state_batch, material)

    np.testing.assert_allclose(batch.tau[0, 0], scalar.tau, atol=1e-8)
    np.testing.assert_allclose(batch.state.epbar[0, 0], scalar.state.epbar, atol=1e-10)
    np.testing.assert_allclose(batch.state.Epl[0, 0], scalar.state.Epl, atol=1e-10)


def test_hill48_isotropic_limit_matches_plane_stress_von_mises_measure():
    material = Hill48PlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=5.0,
        sigma_inf=120.0,
        m=12.0,
        n=0.7,
        hill_f=0.5,
        hill_g=0.5,
        hill_h=0.5,
        hill_n=1.5,
    )
    tau = jnp.array(
        [
            [3.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )

    sigma_vm_plane_stress = np.sqrt(3.0**2 - 3.0 * 1.0 + 1.0**2 + 3.0 * 0.5**2)
    sigma_hill = float(
        jnp.sqrt(
            material.hill_g * tau[0, 0] ** 2
            + material.hill_f * tau[1, 1] ** 2
            + material.hill_h * (tau[0, 0] - tau[1, 1]) ** 2
            + 2.0 * material.hill_n * tau[0, 1] ** 2
        )
    )
    np.testing.assert_allclose(sigma_hill, sigma_vm_plane_stress, rtol=1e-10, atol=1e-10)


def test_hill48_plane_stress_step_is_finite_and_updates_state():
    material = Hill48PlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=5.0,
        sigma_inf=120.0,
        m=18.0,
        n=0.7,
        hill_f=0.6,
        hill_g=1.1,
        hill_h=0.4,
        hill_n=1.8,
    )
    state0 = make_initial_state()
    F2 = jnp.array([[1.25, 0.02], [0.0, 0.96]], dtype=jnp.float64)

    response = update_plane_stress(F2, state0, material)

    assert np.isfinite(np.asarray(response.tau)).all()
    assert np.isfinite(np.asarray(response.state.Epl)).all()
    np.testing.assert_allclose(response.tau[2, 2], 0.0, atol=1e-5)
    assert float(response.state.epbar) >= 0.0
