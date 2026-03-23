import jax
import jax.numpy as jnp
import numpy as np

from tatva.material import (
    DogboneGeometry,
    HSPlaneStressMaterial,
    NewtonSettings,
    build_dogbone_lifter,
    build_dogbone_mesh,
    simulate_dogbone_tension,
)

jax.config.update("jax_enable_x64", True)


def test_build_dogbone_mesh_and_lifter():
    geometry = DogboneGeometry(
        grip_length=6.0,
        transition_length=4.0,
        gauge_length=8.0,
        grip_width=8.0,
        gauge_width=4.0,
        imperfection_depth=0.02,
        imperfection_length=2.0,
    )
    specimen = build_dogbone_mesh(
        geometry, n_x_grip=2, n_x_transition=2, n_x_gauge=4, n_y=3
    )
    lifter = build_dogbone_lifter(specimen)

    assert specimen.mesh.coords.ndim == 2
    assert specimen.mesh.elements.ndim == 2
    assert specimen.left_nodes.size == specimen.right_nodes.size == 4
    assert lifter.size == specimen.mesh.coords.shape[0] * 2
    assert float(specimen.gauge_width0) < geometry.grip_width


def test_simulate_dogbone_tension_tracks_width_reduction():
    material = HSPlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=10.0,
        sigma_inf=120.0,
        m=14.0,
        n=0.7,
        newton_maxiter=20,
    )
    geometry = DogboneGeometry(
        grip_length=4.0,
        transition_length=2.5,
        gauge_length=6.0,
        grip_width=6.0,
        gauge_width=3.0,
        imperfection_depth=0.03,
        imperfection_length=1.5,
    )
    result = simulate_dogbone_tension(
        material,
        geometry=geometry,
        n_x_grip=1,
        n_x_transition=1,
        n_x_gauge=3,
        n_y=2,
        n_increments=1,
        final_displacement=0.15,
        newton=NewtonSettings(max_iters=0, tol=1e-6, line_search_steps=1),
    )

    assert result.history.displacement.shape == (2,)
    assert result.history.accepted_increment.shape == (2,)
    np.testing.assert_allclose(result.history.displacement[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.history.reaction_force[0], 0.0, atol=1e-12)
    assert np.all(np.asarray(result.history.iterations) == np.array([0, 0]))
    assert np.all(np.asarray(result.history.min_gauge_width_ratio) <= 1.0 + 1e-8)
    assert np.all(np.asarray(result.history.center_width_ratio) <= 1.0 + 1e-8)
    assert np.isfinite(np.asarray(result.history.reaction_force)).all()
    assert np.all(np.asarray(result.history.reaction_force) >= -1e-12)
    assert float(result.history.max_epbar[-1]) >= 0.0
    assert np.all(np.isfinite(np.asarray(result.u)))
