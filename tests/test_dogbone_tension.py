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
from tatva.material.dogbone_tension import _gmres_solve

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
    assert result.history.arc_radius.shape == result.history.displacement.shape
    assert result.history.cutbacks.shape == result.history.displacement.shape
    assert result.history.initial_residual_norm.shape == result.history.displacement.shape
    assert result.history.final_residual_norm.shape == result.history.displacement.shape
    assert result.history.linear_converged.shape == result.history.displacement.shape
    assert result.history.linear_iterations.shape == result.history.displacement.shape
    assert result.history.linear_relative_residual.shape == result.history.displacement.shape
    assert float(result.history.max_epbar[-1]) >= 0.0
    assert np.all(np.isfinite(np.asarray(result.u)))


def test_simulate_dogbone_tension_prepeak_matrix_free_smoke():
    material = HSPlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=8.0,
        sigma_inf=40.0,
        m=10.0,
        n=0.7,
        newton_maxiter=20,
    )
    geometry = DogboneGeometry(
        grip_length=4.0,
        transition_length=1.0,
        gauge_length=4.0,
        grip_width=6.0,
        gauge_width=3.0,
        imperfection_depth=0.03,
        imperfection_length=1.0,
    )
    result = simulate_dogbone_tension(
        material,
        geometry=geometry,
        n_x_grip=1,
        n_x_transition=1,
        n_x_gauge=2,
        n_y=1,
        n_increments=1,
        final_displacement=0.05,
        newton=NewtonSettings(
            max_iters=4,
            tol=1e-5,
            line_search_steps=4,
            use_arc_length=False,
            matrix_free=True,
            fast_prepeak=True,
            krylov_tol=1e-6,
            krylov_maxiter=12,
        ),
    )

    assert np.all(np.asarray(result.history.converged))
    assert int(result.history.iterations[-1]) > 0
    assert float(result.history.final_residual_norm[-1]) <= 1e-5
    assert float(result.history.reaction_force[-1]) > 0.0


def test_gmres_solve_handles_small_nonsymmetric_system():
    A = jnp.array(
        [
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float64)

    x, converged, iterations, relres = _gmres_solve(
        matvec=lambda v: A @ v,
        rhs=rhs,
        tol=1e-10,
        maxiter=8,
    )

    np.testing.assert_allclose(np.asarray(A @ x), np.asarray(rhs), rtol=1e-8, atol=1e-8)
    assert converged
    assert iterations > 0
    assert float(relres) <= 1e-10


def test_simulate_dogbone_tension_arc_length_smoke():
    material = HSPlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=2.0,
        sigma_inf=8.0,
        m=8.0,
        n=0.7,
        newton_maxiter=20,
    )
    geometry = DogboneGeometry(
        grip_length=4.0,
        transition_length=0.0,
        gauge_length=6.0,
        grip_width=6.0,
        gauge_width=3.0,
        imperfection_depth=0.05,
        imperfection_length=1.5,
    )
    result = simulate_dogbone_tension(
        material,
        geometry=geometry,
        n_x_grip=1,
        n_x_transition=0,
        n_x_gauge=2,
        n_y=1,
        n_increments=4,
        final_displacement=0.4,
        newton=NewtonSettings(
            max_iters=6,
            tol=1e-5,
            line_search_steps=4,
            use_arc_length=True,
            arc_length_radius=0.1,
            matrix_free=True,
            krylov_tol=1e-6,
            krylov_maxiter=12,
            min_displacement_increment=1e-3,
            max_cutbacks=6,
        ),
    )

    disp = np.asarray(result.history.displacement)
    assert disp.shape[0] >= 2
    assert np.all(np.diff(disp) >= -1e-10)
    assert np.all(np.isfinite(np.asarray(result.history.reaction_force)))
    assert np.all(np.asarray(result.history.converged))
    assert np.all(np.asarray(result.history.arc_radius[1:]) > 0.0)
    assert np.all(np.asarray(result.history.final_residual_norm) >= -1e-12)
    assert np.all(
        np.asarray(result.history.final_residual_norm[1:])
        <= np.asarray(result.history.initial_residual_norm[1:]) + 1e-10
    )
