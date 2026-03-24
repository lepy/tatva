from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import tatva.element as element
from tatva.lifter import Fixed, Lifter, RuntimeValue
from tatva.material.finite_strain_plane_stress import (
    HSPlaneStressMaterial,
    HSPlaneStressState,
    evaluate_plane_stress_responses,
    make_initial_state_field,
    make_plane_stress_residual_flat,
)
from tatva.mesh import Mesh
from tatva.operator import Operator


class DogboneGeometry(eqx.Module):
    """Reference geometry for a 2D dogbone specimen."""

    grip_length: float = 20.0
    transition_length: float = 12.0
    gauge_length: float = 30.0
    grip_width: float = 20.0
    gauge_width: float = 12.5
    imperfection_depth: float = 0.01
    imperfection_length: float = 6.0


class DogboneMesh(eqx.Module):
    """Structured dogbone mesh plus useful node sets."""

    mesh: Mesh
    node_grid: Array
    x_columns: Array
    left_nodes: Array
    right_nodes: Array
    gauge_column_mask: Array
    center_column_mask: Array
    gauge_width0: Array
    center_width0: Array


class NewtonSettings(eqx.Module):
    """Settings for the reduced Newton solve per load increment."""

    max_iters: int = 20
    tol: float = 1e-8
    line_search_steps: int = 8
    line_search_factor: float = 0.5
    min_displacement_increment: float = 1e-3
    max_cutbacks: int = 12
    use_arc_length: bool = False
    arc_length_radius: float = 0.0
    arc_length_scale: float = 1.0
    arc_length_growth_factor: float = 1.25
    arc_length_shrink_factor: float = 0.5
    arc_length_max_steps_factor: int = 8
    fd_eps: float = 1e-6
    tangent_regularization: float = 1e-8
    tangent_rebuild_interval: int = 6
    matrix_free: bool = True
    krylov_tol: float = 1e-6
    krylov_maxiter: int = 40
    eisenstat_walker: bool = True
    fast_prepeak: bool = True
    fast_prepeak_line_search_steps: int = 2
    allow_inexact_steps: bool = False
    accepted_residual_ratio: float = 0.25


class DogboneHistory(eqx.Module):
    """Load-step history for the dogbone simulation."""

    displacement: Array
    reaction_force: Array
    min_gauge_width: Array
    min_gauge_width_ratio: Array
    center_width: Array
    center_width_ratio: Array
    max_epbar: Array
    accepted_increment: Array
    arc_radius: Array
    cutbacks: Array
    initial_residual_norm: Array
    final_residual_norm: Array
    linear_converged: Array
    linear_iterations: Array
    linear_relative_residual: Array
    converged: Array
    iterations: Array


class DogboneSimulation(eqx.Module):
    """Full result of a displacement-controlled dogbone tension simulation."""

    specimen: DogboneMesh
    op: Operator
    lifter: Lifter
    material: HSPlaneStressMaterial
    u: Array
    u_history: Array
    state: HSPlaneStressState
    history: DogboneHistory


class JitDogboneResult(eqx.Module):
    """Per-step results from a JIT-compiled dogbone simulation.

    Each field has shape ``(n_increments,)``.  When the solver is
    ``jax.vmap``-ed over batched materials the leading batch axis is
    prepended, giving ``(n_batch, n_increments)``.
    """

    displacement: Array
    reaction_force: Array
    converged: Array
    iterations: Array
    final_residual_norm: Array
    max_epbar: Array
    min_gauge_width: Array


def build_dogbone_mesh(
    geometry: DogboneGeometry,
    *,
    n_x_grip: int = 6,
    n_x_transition: int = 6,
    n_x_gauge: int = 14,
    n_y: int = 10,
) -> DogboneMesh:
    """Return a structured quadrilateral mesh of a symmetric dogbone specimen."""

    x0 = -0.5 * _total_length(geometry)
    x1 = -0.5 * geometry.gauge_length - geometry.transition_length
    x2 = -0.5 * geometry.gauge_length
    x3 = 0.5 * geometry.gauge_length
    x4 = 0.5 * geometry.gauge_length + geometry.transition_length
    x5 = 0.5 * _total_length(geometry)

    x_values = jnp.concatenate(
        [
            jnp.linspace(x0, x1, n_x_grip + 1),
            jnp.linspace(x1, x2, n_x_transition + 1)[1:],
            jnp.linspace(x2, x3, n_x_gauge + 1)[1:],
            jnp.linspace(x3, x4, n_x_transition + 1)[1:],
            jnp.linspace(x4, x5, n_x_grip + 1)[1:],
        ]
    )
    eta = jnp.linspace(-1.0, 1.0, n_y + 1)
    half_width = jax.vmap(lambda x: _half_width(x, geometry))(x_values)

    xv = jnp.repeat(x_values[:, None], n_y + 1, axis=1)
    yv = half_width[:, None] * eta[None, :]
    coords = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)

    def node_id(i: int, j: int) -> int:
        return i * (n_y + 1) + j

    elements = []
    for i in range(x_values.shape[0] - 1):
        for j in range(n_y):
            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n01 = node_id(i, j + 1)
            n11 = node_id(i + 1, j + 1)
            elements.append([n00, n10, n11, n01])

    node_grid = jnp.arange(coords.shape[0], dtype=jnp.int32).reshape(
        x_values.shape[0], n_y + 1
    )
    left_nodes = node_grid[0]
    right_nodes = node_grid[-1]

    gauge_column_mask = jnp.abs(x_values) <= (0.5 * geometry.gauge_length + 1e-12)
    center_idx = jnp.argmin(jnp.abs(x_values))
    center_column_mask = jnp.arange(x_values.shape[0]) == center_idx

    widths = 2.0 * half_width
    gauge_width0 = jnp.min(widths[gauge_column_mask])
    center_width0 = widths[center_idx]

    return DogboneMesh(
        mesh=Mesh(coords=coords, elements=jnp.array(elements, dtype=jnp.int32)),
        node_grid=node_grid,
        x_columns=x_values,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        gauge_column_mask=gauge_column_mask,
        center_column_mask=center_column_mask,
        gauge_width0=gauge_width0,
        center_width0=center_width0,
    )


def build_dogbone_lifter(specimen: DogboneMesh) -> Lifter:
    """Prescribe axial displacement while keeping lateral contraction essentially free."""

    left_x_dofs = _node_component_dofs(specimen.left_nodes, 0)
    right_x_dofs = _node_component_dofs(specimen.right_nodes, 0)
    anchor_node = specimen.left_nodes[specimen.left_nodes.shape[0] // 2]
    anchor_y_dof = _node_component_dofs(jnp.asarray([anchor_node]), 1)

    return Lifter(
        specimen.mesh.coords.shape[0] * 2,
        Fixed(left_x_dofs, 0.0),
        Fixed(anchor_y_dof, 0.0),
        Fixed(right_x_dofs, RuntimeValue("right_ux", default=0.0)),
    )


def simulate_dogbone_tension(
    material: HSPlaneStressMaterial,
    *,
    geometry: DogboneGeometry | None = None,
    n_x_grip: int = 6,
    n_x_transition: int = 6,
    n_x_gauge: int = 14,
    n_y: int = 10,
    n_increments: int = 20,
    final_displacement: float = 6.0,
    newton: NewtonSettings | None = None,
) -> DogboneSimulation:
    """Run a displacement-controlled dogbone tension simulation."""

    if geometry is None:
        geometry = DogboneGeometry()
    if newton is None:
        newton = NewtonSettings()

    specimen = build_dogbone_mesh(
        geometry,
        n_x_grip=n_x_grip,
        n_x_transition=n_x_transition,
        n_x_gauge=n_x_gauge,
        n_y=n_y,
    )
    op = Operator(specimen.mesh, element.Quad4())
    lifter = build_dogbone_lifter(specimen)
    state = make_initial_state_field(op, dtype=specimen.mesh.coords.dtype)
    u_full = jnp.zeros((specimen.mesh.coords.shape[0] * 2,), dtype=specimen.mesh.coords.dtype)

    residual_full = make_plane_stress_residual_flat(op, jit=True)
    responses_full = jax.jit(
        lambda u_flat, state_n, material: evaluate_plane_stress_responses(
            op, u_flat.reshape(-1, 2), state_n, material
        )
    )

    # Warm up the compiled global kernels once for the current mesh/layout.
    _ = residual_full(u_full, state, material).block_until_ready()
    _ = responses_full(u_full, state, material).state.epbar.block_until_ready()
    if newton.use_arc_length:
        return _simulate_dogbone_tension_arc_length(
            specimen=specimen,
            op=op,
            lifter=lifter,
            material=material,
            state=state,
            u_full=u_full,
            residual_full=residual_full,
            responses_full=responses_full,
            n_increments=n_increments,
            final_displacement=final_displacement,
            newton=newton,
        )

    target_displacements = list(
        np.asarray(jnp.linspace(0.0, final_displacement, n_increments + 1)[1:])
    )
    current_displacement = 0.0
    reaction_force = []
    min_gauge_width = []
    min_gauge_width_ratio = []
    center_width = []
    center_width_ratio = []
    max_epbar = []
    accepted_increment = []
    accepted_radius = []
    accepted_cutbacks = []
    initial_residual_norm = []
    final_residual_norm = []
    linear_converged = []
    linear_iterations = []
    linear_relative_residual = []
    converged = []
    iterations = []
    u_history = [u_full.reshape(-1, 2)]

    cutbacks = 0
    while target_displacements:
        target_disp = float(target_displacements.pop(0))
        step_size = target_disp - current_displacement
        if step_size <= 0.0:
            continue

        state_prev = state
        u_guess = _scaled_displacement_guess(
            u_full=u_full,
            current_displacement=current_displacement,
            target_displacement=target_disp,
            lifter=lifter,
        )
        lifter_step = lifter.at["right_ux"].set(
            jnp.asarray(target_disp, dtype=u_full.dtype)
        )
        u_trial, step_info = _solve_dogbone_increment(
            u_full=u_guess,
            state_n=state_prev,
            material=material,
            lifter=lifter_step,
            residual_full=residual_full,
            newton=newton,
        )
        responses = responses_full(u_trial, state_prev, material)
        residual_at_step = residual_full(u_trial, state_prev, material)
        valid = _step_is_valid(
            newton=newton,
            step_info=step_info,
            u_full=u_trial,
            residual=residual_at_step,
            state=responses.state,
        )

        if not valid:
            cutbacks += 1
            if (
                cutbacks > newton.max_cutbacks
                or step_size <= newton.min_displacement_increment
            ):
                break
            midpoint = current_displacement + 0.5 * step_size
            target_displacements.insert(0, target_disp)
            target_displacements.insert(0, midpoint)
            continue

        step_cutbacks = cutbacks
        cutbacks = 0
        current_displacement = target_disp
        u_full = u_trial
        state = responses.state
        u_history.append(u_full.reshape(-1, 2))

        reaction_force.append(
            -jnp.sum(
                residual_at_step[_node_component_dofs(specimen.right_nodes, 0)]
            )
        )

        min_width, center = _measure_current_widths(specimen, u_full.reshape(-1, 2))
        min_gauge_width.append(min_width)
        min_gauge_width_ratio.append(min_width / specimen.gauge_width0)
        center_width.append(center)
        center_width_ratio.append(center / specimen.center_width0)
        max_epbar.append(jnp.max(state.epbar))
        accepted_increment.append(step_size)
        accepted_radius.append(jnp.nan)
        accepted_cutbacks.append(step_cutbacks)
        initial_residual_norm.append(_step_info_value(step_info, "initial_residual_norm"))
        final_residual_norm.append(_step_info_value(step_info, "final_residual_norm"))
        linear_converged.append(_step_info_value(step_info, "linear_converged"))
        linear_iterations.append(_step_info_value(step_info, "linear_iterations"))
        linear_relative_residual.append(
            _step_info_value(step_info, "linear_relative_residual")
        )
        converged.append(step_info["converged"])
        iterations.append(step_info["iterations"])

    history = DogboneHistory(
        displacement=jnp.asarray(
            np.concatenate(([0.0], np.cumsum(np.asarray(accepted_increment))))
        ),
        reaction_force=jnp.asarray(
            np.concatenate(([0.0], np.abs(np.asarray(reaction_force))))
        ),
        min_gauge_width=jnp.asarray(
            np.concatenate(([specimen.gauge_width0], np.asarray(min_gauge_width)))
        ),
        min_gauge_width_ratio=jnp.asarray(
            np.concatenate(([1.0], np.asarray(min_gauge_width_ratio)))
        ),
        center_width=jnp.asarray(
            np.concatenate(([specimen.center_width0], np.asarray(center_width)))
        ),
        center_width_ratio=jnp.asarray(
            np.concatenate(([1.0], np.asarray(center_width_ratio)))
        ),
        max_epbar=jnp.asarray(np.concatenate(([0.0], np.asarray(max_epbar)))),
        accepted_increment=jnp.asarray(np.concatenate(([0.0], np.asarray(accepted_increment)))),
        arc_radius=jnp.asarray(np.concatenate(([jnp.nan], np.asarray(accepted_radius)))),
        cutbacks=jnp.asarray(np.concatenate(([0], np.asarray(accepted_cutbacks)))),
        initial_residual_norm=jnp.asarray(
            np.concatenate(([0.0], np.asarray(initial_residual_norm)))
        ),
        final_residual_norm=jnp.asarray(
            np.concatenate(([0.0], np.asarray(final_residual_norm)))
        ),
        linear_converged=jnp.asarray(
            np.concatenate(([True], np.asarray(linear_converged, dtype=bool)))
        ),
        linear_iterations=jnp.asarray(
            np.concatenate(([0], np.asarray(linear_iterations)))
        ),
        linear_relative_residual=jnp.asarray(
            np.concatenate(([0.0], np.asarray(linear_relative_residual)))
        ),
        converged=jnp.asarray(np.concatenate(([True], np.asarray(converged)))),
        iterations=jnp.asarray(np.concatenate(([0], np.asarray(iterations)))),
    )
    return DogboneSimulation(
        specimen=specimen,
        op=op,
        lifter=lifter,
        material=material,
        u=u_full.reshape(-1, 2),
        u_history=jnp.stack(u_history, axis=0),
        state=state,
        history=history,
    )


def make_jit_dogbone_solver(
    geometry: DogboneGeometry | None = None,
    *,
    n_x_grip: int = 6,
    n_x_transition: int = 6,
    n_x_gauge: int = 14,
    n_y: int = 10,
    n_increments: int = 20,
    final_displacement: float = 6.0,
    newton: NewtonSettings | None = None,
):
    """Return ``(solver, specimen)`` where ``solver(material) → JitDogboneResult``.

    The solver is fully JAX-traceable and can be composed with ``jax.jit``
    and ``jax.vmap`` for efficient batched parameter sweeps::

        solver, specimen = make_jit_dogbone_solver(geometry, n_increments=30, ...)
        solve_one = jax.jit(solver)
        result = solve_one(material)

        # 10 000 simulations with varying Hockett-Sherby parameters:
        solve_batch = jax.jit(jax.vmap(solver))
        results = solve_batch(batched_materials)

    All fields of the *material* must be broadcastable to the batch size
    when using ``jax.vmap``.
    """

    if geometry is None:
        geometry = DogboneGeometry()
    if newton is None:
        newton = NewtonSettings()

    specimen = build_dogbone_mesh(
        geometry,
        n_x_grip=n_x_grip,
        n_x_transition=n_x_transition,
        n_x_gauge=n_x_gauge,
        n_y=n_y,
    )
    op = Operator(specimen.mesh, element.Quad4())
    lifter = build_dogbone_lifter(specimen)

    residual_fn = make_plane_stress_residual_flat(op, jit=False)
    responses_fn = lambda u_flat, s, m: evaluate_plane_stress_responses(
        op, u_flat.reshape(-1, 2), s, m
    )

    targets = jnp.linspace(0.0, final_displacement, n_increments + 1)[1:]
    state0 = make_initial_state_field(op, dtype=specimen.mesh.coords.dtype)
    u0 = jnp.zeros(
        specimen.mesh.coords.shape[0] * 2, dtype=specimen.mesh.coords.dtype
    )
    right_x_dofs = _node_component_dofs(specimen.right_nodes, 0)

    def solve(material: HSPlaneStressMaterial) -> JitDogboneResult:
        def step(carry, target_disp):
            u_full, state, prev_disp = carry

            scale = jnp.where(
                jnp.abs(prev_disp) > 1e-14, target_disp / prev_disp, 1.0
            )
            lifter_step = lifter.at["right_ux"].set(target_disp)
            u_guess = lifter_step.lift_from_zeros(
                lifter_step.reduce(u_full * scale)
            )

            u_new, iters, converged, final_rnorm = _jit_newton_solve(
                u_guess, state, material, lifter_step, residual_fn, newton
            )

            responses = responses_fn(u_new, state, material)
            res = residual_fn(u_new, state, material)
            force = -jnp.sum(res[right_x_dofs])
            min_width, _ = _measure_current_widths(
                specimen, u_new.reshape(-1, 2)
            )

            out = JitDogboneResult(
                displacement=target_disp,
                reaction_force=jnp.abs(force),
                converged=converged,
                iterations=iters,
                final_residual_norm=final_rnorm,
                max_epbar=jnp.max(responses.state.epbar),
                min_gauge_width=min_width,
            )
            return (u_new, responses.state, target_disp), out

        _, history = jax.lax.scan(
            step,
            (u0, state0, jnp.asarray(0.0, dtype=u0.dtype)),
            targets,
        )
        return history

    return solve, specimen


def _jit_newton_solve(u_full, state_n, material, lifter_step, residual_fn, newton):
    """Fully JAX-traceable Newton solve for one displacement increment.

    Uses a frozen tangent (modified Newton) for efficiency and robustness,
    matching the ``fast_prepeak`` strategy of the main solver.
    """

    u_red = lifter_step.reduce(u_full)

    def reduced_residual(u_red_local):
        return lifter_step.reduce(
            residual_fn(lifter_step.lift_from_zeros(u_red_local), state_n, material)
        )

    # Freeze tangent at initial guess (modified Newton).
    _, tangent_vec = jax.linearize(reduced_residual, u_red)

    def matvec(v):
        return tangent_vec(v) + newton.tangent_regularization * v

    r = reduced_residual(u_red)
    initial_rnorm = jnp.linalg.norm(r)

    def cond(carry):
        it, _, _, converged, terminated, _ = carry
        return (it < newton.max_iters) & (~converged) & (~terminated)

    def body(carry):
        it, u_red_local, r_local, _, _, prev_rnorm = carry
        rnorm = jnp.linalg.norm(r_local)
        if newton.eisenstat_walker:
            eta = _eisenstat_walker_eta(rnorm, prev_rnorm, newton)
        else:
            eta = newton.krylov_tol

        du, linear_ok, _, _ = _gmres_solve(
            matvec=matvec, rhs=-r_local, tol=eta, maxiter=newton.krylov_maxiter,
        )
        u_candidate, r_candidate, accepted = _line_search_update(
            u_red=u_red_local,
            du=du,
            residual=reduced_residual,
            line_search_steps=newton.line_search_steps,
            reduction=newton.line_search_factor,
        )

        def fallback(_):
            K = _finite_difference_tangent(
                residual=reduced_residual,
                u_red=u_red_local,
                fd_eps=newton.fd_eps,
                regularization=newton.tangent_regularization,
            )
            du_dense = _solve_linear_system(K, -r_local)
            return _line_search_update(
                u_red=u_red_local,
                du=du_dense,
                residual=reduced_residual,
                line_search_steps=newton.line_search_steps,
                reduction=newton.line_search_factor,
            )

        u_new, r_new, accepted = jax.lax.cond(
            (~linear_ok) | (~accepted),
            fallback,
            lambda _: (u_candidate, r_candidate, accepted),
            operand=None,
        )
        residual_norm = jnp.linalg.norm(r_new)
        converged = accepted & (residual_norm <= newton.tol)
        terminated = ~accepted
        return (it + 1, u_new, r_new, converged, terminated, rnorm)

    it, u_red_final, r_final, converged, _, _ = jax.lax.while_loop(
        cond,
        body,
        (
            jnp.asarray(0, jnp.int32),
            u_red,
            r,
            initial_rnorm <= newton.tol,
            jnp.asarray(False),
            initial_rnorm,
        ),
    )

    return (
        lifter_step.lift_from_zeros(u_red_final),
        it,
        converged,
        jnp.linalg.norm(r_final),
    )


def _simulate_dogbone_tension_arc_length(
    *,
    specimen: DogboneMesh,
    op: Operator,
    lifter: Lifter,
    material: HSPlaneStressMaterial,
    state: HSPlaneStressState,
    u_full: Array,
    residual_full,
    responses_full,
    n_increments: int,
    final_displacement: float,
    newton: NewtonSettings,
) -> DogboneSimulation:
    current_displacement = 0.0
    reaction_force = []
    min_gauge_width = []
    min_gauge_width_ratio = []
    center_width = []
    center_width_ratio = []
    max_epbar = []
    accepted_increment = []
    accepted_radius = []
    accepted_cutbacks = []
    initial_residual_norm = []
    final_residual_norm = []
    linear_converged = []
    linear_iterations = []
    linear_relative_residual = []
    converged = []
    iterations = []
    u_history = [u_full.reshape(-1, 2)]

    radius0 = (
        newton.arc_length_radius
        if newton.arc_length_radius > 0.0
        else final_displacement / max(n_increments, 1)
    )
    radius = radius0
    direction = 1.0
    cutbacks = 0
    accepted_steps = 0
    prev_delta_u_red = None
    prev_delta_lambda = None
    min_step = max(float(newton.min_displacement_increment), 1e-12)
    max_steps = max(
        newton.arc_length_max_steps_factor * max(n_increments, 1),
        int(np.ceil(final_displacement / min_step)) + 1,
        16,
    )

    while (current_displacement < final_displacement - 1e-12) and (
        accepted_steps < max_steps
    ):
        state_prev = state
        u_red_prev = lifter.reduce(u_full)
        trial_radius = min(radius, max(final_displacement - current_displacement, min_step))
        u_red_guess, lam_guess, predictor_direction = _predict_arc_length_step(
            u_red_prev=u_red_prev,
            lambda_prev=current_displacement,
            radius=trial_radius,
            direction=direction,
            previous_delta_u_red=prev_delta_u_red,
            previous_delta_lambda=prev_delta_lambda,
            lifter=lifter,
            residual_full=residual_full,
            state_n=state_prev,
            material=material,
            newton=newton,
        )
        u_trial, lambda_trial, step_info = _solve_dogbone_increment_arc_length(
            u_red_prev=u_red_prev,
            lambda_prev=current_displacement,
            u_red_guess=u_red_guess,
            lambda_guess=lam_guess,
            arc_radius=trial_radius,
            lifter=lifter,
            residual_full=residual_full,
            state_n=state_prev,
            material=material,
            newton=newton,
        )

        lifter_step = lifter.at["right_ux"].set(jnp.asarray(lambda_trial, dtype=u_full.dtype))
        responses = responses_full(u_trial, state_prev, material)
        residual_at_step = residual_full(u_trial, state_prev, material)
        valid = _step_is_valid(
            newton=newton,
            step_info=step_info,
            u_full=u_trial,
            residual=residual_at_step,
            state=responses.state,
        )
        dlam = float(lambda_trial - current_displacement)
        if dlam <= 1e-12:
            valid = False

        if not valid:
            cutbacks += 1
            radius = max(
                trial_radius * newton.arc_length_shrink_factor,
                newton.min_displacement_increment,
            )
            if (
                cutbacks > newton.max_cutbacks
                or radius <= newton.min_displacement_increment
            ):
                break
            prev_delta_u_red = None
            prev_delta_lambda = None
            continue

        del lifter_step
        step_cutbacks = cutbacks
        cutbacks = 0
        accepted_steps += 1
        direction = predictor_direction if abs(dlam) < 1e-12 else np.sign(dlam)
        current_displacement = float(lambda_trial)
        prev_delta_u_red = lifter.reduce(u_trial) - u_red_prev
        prev_delta_lambda = dlam
        u_full = u_trial
        state = responses.state
        u_history.append(u_full.reshape(-1, 2))

        reaction_force.append(
            -jnp.sum(
                residual_at_step[_node_component_dofs(specimen.right_nodes, 0)]
            )
        )

        min_width, center = _measure_current_widths(specimen, u_full.reshape(-1, 2))
        min_gauge_width.append(min_width)
        min_gauge_width_ratio.append(min_width / specimen.gauge_width0)
        center_width.append(center)
        center_width_ratio.append(center / specimen.center_width0)
        max_epbar.append(jnp.max(state.epbar))
        accepted_increment.append(dlam)
        accepted_radius.append(trial_radius)
        accepted_cutbacks.append(step_cutbacks)
        initial_residual_norm.append(_step_info_value(step_info, "initial_residual_norm"))
        final_residual_norm.append(_step_info_value(step_info, "final_residual_norm"))
        linear_converged.append(_step_info_value(step_info, "linear_converged"))
        linear_iterations.append(_step_info_value(step_info, "linear_iterations"))
        linear_relative_residual.append(
            _step_info_value(step_info, "linear_relative_residual")
        )
        converged.append(step_info["converged"])
        iterations.append(step_info["iterations"])

        linear_ok = bool(_step_info_value(step_info, "linear_converged"))
        linear_relres = float(_step_info_value(step_info, "linear_relative_residual"))
        nonlinear_iters = int(step_info["iterations"])
        if linear_ok and linear_relres <= max(newton.krylov_tol * 10.0, 1e-8) and (
            nonlinear_iters <= max(2, newton.max_iters // 3)
        ):
            radius = min(trial_radius * newton.arc_length_growth_factor, radius0)
        elif (not linear_ok) or linear_relres > max(newton.krylov_tol * 100.0, 1e-6):
            radius = max(
                trial_radius * newton.arc_length_shrink_factor,
                newton.min_displacement_increment,
            )
        elif nonlinear_iters >= max(3, int(0.75 * newton.max_iters)):
            radius = max(
                trial_radius * np.sqrt(newton.arc_length_shrink_factor),
                newton.min_displacement_increment,
            )
        else:
            radius = trial_radius

    history = DogboneHistory(
        displacement=jnp.asarray(
            np.concatenate(([0.0], np.cumsum(np.asarray(accepted_increment))))
        ),
        reaction_force=jnp.asarray(
            np.concatenate(([0.0], np.abs(np.asarray(reaction_force))))
        ),
        min_gauge_width=jnp.asarray(
            np.concatenate(([specimen.gauge_width0], np.asarray(min_gauge_width)))
        ),
        min_gauge_width_ratio=jnp.asarray(
            np.concatenate(([1.0], np.asarray(min_gauge_width_ratio)))
        ),
        center_width=jnp.asarray(
            np.concatenate(([specimen.center_width0], np.asarray(center_width)))
        ),
        center_width_ratio=jnp.asarray(
            np.concatenate(([1.0], np.asarray(center_width_ratio)))
        ),
        max_epbar=jnp.asarray(np.concatenate(([0.0], np.asarray(max_epbar)))),
        accepted_increment=jnp.asarray(
            np.concatenate(([0.0], np.asarray(accepted_increment)))
        ),
        arc_radius=jnp.asarray(np.concatenate(([0.0], np.asarray(accepted_radius)))),
        cutbacks=jnp.asarray(np.concatenate(([0], np.asarray(accepted_cutbacks)))),
        initial_residual_norm=jnp.asarray(
            np.concatenate(([0.0], np.asarray(initial_residual_norm)))
        ),
        final_residual_norm=jnp.asarray(
            np.concatenate(([0.0], np.asarray(final_residual_norm)))
        ),
        linear_converged=jnp.asarray(
            np.concatenate(([True], np.asarray(linear_converged, dtype=bool)))
        ),
        linear_iterations=jnp.asarray(
            np.concatenate(([0], np.asarray(linear_iterations)))
        ),
        linear_relative_residual=jnp.asarray(
            np.concatenate(([0.0], np.asarray(linear_relative_residual)))
        ),
        converged=jnp.asarray(np.concatenate(([True], np.asarray(converged)))),
        iterations=jnp.asarray(np.concatenate(([0], np.asarray(iterations)))),
    )
    return DogboneSimulation(
        specimen=specimen,
        op=op,
        lifter=lifter,
        material=material,
        u=u_full.reshape(-1, 2),
        u_history=jnp.stack(u_history, axis=0),
        state=state,
        history=history,
    )


def _solve_dogbone_increment(
    *,
    u_full: Array,
    state_n: HSPlaneStressState,
    material: HSPlaneStressMaterial,
    lifter: Lifter,
    residual_full,
    newton: NewtonSettings,
) -> tuple[Array, dict[str, Array]]:
    """Solve one displacement increment on the reduced space."""

    if newton.fast_prepeak:
        return _solve_dogbone_increment_prepeak(
            u_full=u_full,
            state_n=state_n,
            material=material,
            lifter=lifter,
            residual_full=residual_full,
            newton=newton,
        )

    u_red = lifter.reduce(u_full)

    if newton.max_iters <= 0:
        return lifter.lift_from_zeros(u_red), {
            "converged": jnp.asarray(False),
            "iterations": jnp.asarray(0),
            "initial_residual_norm": jnp.asarray(0.0, dtype=u_red.dtype),
            "final_residual_norm": jnp.asarray(0.0, dtype=u_red.dtype),
            "linear_converged": jnp.asarray(True),
            "linear_iterations": jnp.asarray(0),
            "linear_relative_residual": jnp.asarray(0.0, dtype=u_red.dtype),
        }

    def reduced_residual(u_red_local: Array) -> Array:
        u_trial = lifter.lift_from_zeros(u_red_local)
        return lifter.reduce(residual_full(u_trial, state_n, material))

    def reduced_tangent(u_red_local: Array) -> Array:
        return _finite_difference_tangent(
            residual=reduced_residual,
            u_red=u_red_local,
            fd_eps=newton.fd_eps,
            regularization=newton.tangent_regularization,
        )

    def reduced_linear_operator(u_red_local: Array):
        _, tangent_vec = jax.linearize(reduced_residual, u_red_local)

        def matvec(v: Array) -> Array:
            return tangent_vec(v) + newton.tangent_regularization * v

        return matvec

    r = reduced_residual(u_red)
    initial_residual_norm = jnp.linalg.norm(r)
    converged0 = initial_residual_norm <= newton.tol

    if newton.matrix_free:
        def cond(carry):
            it, _, _, converged, terminated, _, _, _, _ = carry
            return (it < newton.max_iters) & (~converged) & (~terminated)

        def body(carry):
            it, u_red_local, r_local, _, _, _, _, _, prev_rnorm = carry
            rnorm = jnp.linalg.norm(r_local)
            if newton.eisenstat_walker:
                eta = _eisenstat_walker_eta(rnorm, prev_rnorm, newton)
            else:
                eta = newton.krylov_tol
            matvec = reduced_linear_operator(u_red_local)
            du, linear_ok, linear_iters, linear_relres = _gmres_solve(
                matvec=matvec,
                rhs=-r_local,
                tol=eta,
                maxiter=newton.krylov_maxiter,
            )
            u_candidate, r_candidate, accepted = _line_search_update(
                u_red=u_red_local,
                du=du,
                residual=reduced_residual,
                line_search_steps=newton.line_search_steps,
                reduction=newton.line_search_factor,
            )

            def fallback(_):
                K = reduced_tangent(u_red_local)
                du_dense = _solve_linear_system(K, -r_local)
                return _line_search_update(
                    u_red=u_red_local,
                    du=du_dense,
                    residual=reduced_residual,
                    line_search_steps=newton.line_search_steps,
                    reduction=newton.line_search_factor,
                )

            u_new, r_new, accepted = jax.lax.cond(
                (~linear_ok) | (~accepted),
                fallback,
                lambda _: (u_candidate, r_candidate, accepted),
                operand=None,
            )
            residual_norm = jnp.linalg.norm(r_new)
            converged = accepted & (residual_norm <= newton.tol)
            terminated = ~accepted
            return (
                it + 1,
                u_new,
                r_new,
                converged,
                terminated,
                linear_ok,
                linear_iters,
                linear_relres,
                rnorm,
            )

        carry = jax.lax.while_loop(
            cond,
            body,
            (
                jnp.asarray(0, dtype=jnp.int32),
                u_red,
                r,
                converged0,
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=u_red.dtype),
                initial_residual_norm,
            ),
        )
        (
            last_iter,
            u_red,
            r,
            converged,
            _terminated,
            last_linear_ok,
            last_linear_iters,
            last_linear_relres,
            _,
        ) = carry
    else:
        K0 = reduced_tangent(u_red)

        def cond(carry):
            it, _, _, _, converged, terminated, _, _, _ = carry
            return (it < newton.max_iters) & (~converged) & (~terminated)

        def body(carry):
            it, u_red_local, r_local, K_local, _, _, _, _, _ = carry
            du = _solve_linear_system(K_local, -r_local)
            u_new, r_new, accepted = _line_search_update(
                u_red=u_red_local,
                du=du,
                residual=reduced_residual,
                line_search_steps=newton.line_search_steps,
                reduction=newton.line_search_factor,
            )
            s = u_new - u_red_local
            y = r_new - r_local
            residual_norm = jnp.linalg.norm(r_new)
            converged = accepted & (residual_norm <= newton.tol)
            terminated = ~accepted
            K_new = jax.lax.cond(
                accepted & (~converged),
                lambda _: _broyden_update_tangent(
                    K_local,
                    s,
                    y,
                    regularization=newton.tangent_regularization,
                ),
                lambda _: K_local,
                operand=None,
            )
            return (
                it + 1,
                u_new,
                r_new,
                K_new,
                converged,
                terminated,
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=u_red.dtype),
            )

        carry = jax.lax.while_loop(
            cond,
            body,
            (
                jnp.asarray(0, dtype=jnp.int32),
                u_red,
                r,
                K0,
                converged0,
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=u_red.dtype),
            ),
        )
        (
            last_iter,
            u_red,
            r,
            _K,
            converged,
            _terminated,
            last_linear_ok,
            last_linear_iters,
            last_linear_relres,
        ) = carry

    u_full_new = lifter.lift_from_zeros(u_red)
    return u_full_new, {
        "converged": converged,
        "iterations": last_iter,
        "initial_residual_norm": initial_residual_norm,
        "final_residual_norm": jnp.linalg.norm(r),
        "linear_converged": last_linear_ok,
        "linear_iterations": last_linear_iters,
        "linear_relative_residual": jnp.asarray(
            last_linear_relres, dtype=u_red.dtype
        ),
    }


def _solve_dogbone_increment_prepeak(
    *,
    u_full: Array,
    state_n: HSPlaneStressState,
    material: HSPlaneStressMaterial,
    lifter: Lifter,
    residual_full,
    newton: NewtonSettings,
) -> tuple[Array, dict[str, Array]]:
    """Cheaper modified-Newton path for standard pre-peak displacement control."""

    u_red = lifter.reduce(u_full)

    if newton.max_iters <= 0:
        return lifter.lift_from_zeros(u_red), {
            "converged": jnp.asarray(False),
            "iterations": jnp.asarray(0),
            "initial_residual_norm": jnp.asarray(0.0, dtype=u_red.dtype),
            "final_residual_norm": jnp.asarray(0.0, dtype=u_red.dtype),
            "linear_converged": jnp.asarray(True),
            "linear_iterations": jnp.asarray(0),
            "linear_relative_residual": jnp.asarray(0.0, dtype=u_red.dtype),
        }

    def reduced_residual(u_red_local: Array) -> Array:
        u_trial = lifter.lift_from_zeros(u_red_local)
        return lifter.reduce(residual_full(u_trial, state_n, material))

    def reduced_tangent(u_red_local: Array) -> Array:
        return _finite_difference_tangent(
            residual=reduced_residual,
            u_red=u_red_local,
            fd_eps=newton.fd_eps,
            regularization=newton.tangent_regularization,
        )

    r0 = reduced_residual(u_red)
    initial_residual_norm = jnp.linalg.norm(r0)
    if bool(initial_residual_norm <= newton.tol):
        return lifter.lift_from_zeros(u_red), {
            "converged": jnp.asarray(True),
            "iterations": jnp.asarray(0),
            "initial_residual_norm": initial_residual_norm,
            "final_residual_norm": initial_residual_norm,
            "linear_converged": jnp.asarray(True),
            "linear_iterations": jnp.asarray(0),
            "linear_relative_residual": jnp.asarray(0.0, dtype=u_red.dtype),
        }

    if newton.matrix_free:
        _, tangent_vec = jax.linearize(reduced_residual, u_red)

        def matvec(v: Array) -> Array:
            return tangent_vec(v) + newton.tangent_regularization * v

        def line_search(u_red_local: Array, du: Array):
            return _line_search_update(
                u_red=u_red_local,
                du=du,
                residual=reduced_residual,
                line_search_steps=min(
                    newton.line_search_steps, newton.fast_prepeak_line_search_steps
                ),
                reduction=newton.line_search_factor,
            )

        def cond(carry):
            it, _, r_local, converged, terminated, _, _, _, _ = carry
            return (
                (it < newton.max_iters)
                & (~converged)
                & (~terminated)
                & (jnp.linalg.norm(r_local) > newton.tol)
            )

        def body(carry):
            it, u_red_local, r_local, _, _, _, _, _, prev_rnorm = carry
            rnorm = jnp.linalg.norm(r_local)
            if newton.eisenstat_walker:
                eta = _eisenstat_walker_eta(rnorm, prev_rnorm, newton)
            else:
                eta = newton.krylov_tol
            du, linear_ok, linear_iters, linear_relres = _gmres_solve(
                matvec=matvec,
                rhs=-r_local,
                tol=eta,
                maxiter=newton.krylov_maxiter,
            )
            u_new, r_new, accepted = line_search(u_red_local, du)
            res_norm = jnp.linalg.norm(r_new)
            converged = accepted & (res_norm <= newton.tol)
            terminated = (~accepted) | (~linear_ok)
            return (
                it + 1,
                u_new,
                r_new,
                converged,
                terminated,
                linear_ok,
                linear_iters,
                linear_relres,
                rnorm,
            )

        (
            last_iter,
            u_fast,
            r_fast,
            converged_fast,
            terminated_fast,
            linear_ok_fast,
            linear_iters_fast,
            linear_relres_fast,
            _,
        ) = jax.lax.while_loop(
            cond,
            body,
            (
                jnp.asarray(0, dtype=jnp.int32),
                u_red,
                r0,
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=u_red.dtype),
                initial_residual_norm,
            ),
        )

        if bool(converged_fast):
            return lifter.lift_from_zeros(u_fast), {
                "converged": converged_fast,
                "iterations": last_iter,
                "initial_residual_norm": initial_residual_norm,
                "final_residual_norm": jnp.linalg.norm(r_fast),
                "linear_converged": linear_ok_fast,
                "linear_iterations": linear_iters_fast,
                "linear_relative_residual": linear_relres_fast,
            }

        if bool(terminated_fast):
            full_newton = eqx.tree_at(lambda n: n.fast_prepeak, newton, False)
            return _solve_dogbone_increment(
                u_full=lifter.lift_from_zeros(u_red),
                state_n=state_n,
                material=material,
                lifter=lifter,
                residual_full=residual_full,
                newton=full_newton,
            )

        return lifter.lift_from_zeros(u_fast), {
            "converged": converged_fast,
            "iterations": last_iter,
            "initial_residual_norm": initial_residual_norm,
            "final_residual_norm": jnp.linalg.norm(r_fast),
            "linear_converged": linear_ok_fast,
            "linear_iterations": linear_iters_fast,
            "linear_relative_residual": linear_relres_fast,
        }

    K = reduced_tangent(u_red)

    def cond(carry):
        it, _, r_local, converged, terminated = carry
        return (
            (it < newton.max_iters)
            & (~converged)
            & (~terminated)
            & (jnp.linalg.norm(r_local) > newton.tol)
        )

    def body(carry):
        it, u_red_local, r_local, _, _ = carry
        du = _solve_linear_system(K, -r_local)
        u_new, r_new, accepted = _line_search_update(
            u_red=u_red_local,
            du=du,
            residual=reduced_residual,
            line_search_steps=min(
                newton.line_search_steps, newton.fast_prepeak_line_search_steps
            ),
            reduction=newton.line_search_factor,
        )
        res_norm = jnp.linalg.norm(r_new)
        converged = accepted & (res_norm <= newton.tol)
        terminated = ~accepted
        return it + 1, u_new, r_new, converged, terminated

    last_iter, u_fast, r_fast, converged_fast, terminated_fast = jax.lax.while_loop(
        cond,
        body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            u_red,
            r0,
            jnp.asarray(False),
            jnp.asarray(False),
        ),
    )

    if bool(terminated_fast) and (not bool(converged_fast)):
        full_newton = eqx.tree_at(lambda n: n.fast_prepeak, newton, False)
        return _solve_dogbone_increment(
            u_full=lifter.lift_from_zeros(u_red),
            state_n=state_n,
            material=material,
            lifter=lifter,
            residual_full=residual_full,
            newton=full_newton,
        )

    return lifter.lift_from_zeros(u_fast), {
        "converged": converged_fast,
        "iterations": last_iter,
        "initial_residual_norm": initial_residual_norm,
        "final_residual_norm": jnp.linalg.norm(r_fast),
        "linear_converged": jnp.asarray(True),
        "linear_iterations": jnp.asarray(0, dtype=jnp.int32),
        "linear_relative_residual": jnp.asarray(0.0, dtype=u_red.dtype),
    }


def _solve_dogbone_increment_arc_length(
    *,
    u_red_prev: Array,
    lambda_prev: float,
    u_red_guess: Array,
    lambda_guess: float,
    arc_radius: float,
    lifter: Lifter,
    residual_full,
    state_n: HSPlaneStressState,
    material: HSPlaneStressMaterial,
    newton: NewtonSettings,
) -> tuple[Array, float, dict[str, Array]]:
    if newton.max_iters <= 0:
        lifter_step = lifter.at["right_ux"].set(jnp.asarray(lambda_guess, dtype=u_red_guess.dtype))
        return lifter_step.lift_from_zeros(u_red_guess), float(lambda_guess), {
            "converged": jnp.asarray(False),
            "iterations": jnp.asarray(0),
            "initial_residual_norm": jnp.asarray(0.0, dtype=u_red_guess.dtype),
            "final_residual_norm": jnp.asarray(0.0, dtype=u_red_guess.dtype),
            "linear_converged": jnp.asarray(True),
            "linear_iterations": jnp.asarray(0),
            "linear_relative_residual": jnp.asarray(0.0, dtype=u_red_guess.dtype),
        }

    def reduced_residual(u_red_local: Array, lam_local: Array) -> Array:
        lifter_step = lifter.at["right_ux"].set(lam_local)
        u_trial = lifter_step.lift_from_zeros(u_red_local)
        return lifter_step.reduce(residual_full(u_trial, state_n, material))

    n_red = u_red_prev.shape[0]
    norm_factor = max(n_red, 1)

    def augmented_residual(z: Array) -> Array:
        u_red_local = z[:-1]
        lam_local = z[-1]
        r_eq = reduced_residual(u_red_local, lam_local)
        du = u_red_local - u_red_prev
        dlam = lam_local - lambda_prev
        g = jnp.dot(du, du) / norm_factor + (
            newton.arc_length_scale * dlam
        ) ** 2 - arc_radius**2
        return jnp.concatenate([r_eq, jnp.asarray([g], dtype=r_eq.dtype)])

    z = jnp.concatenate(
        [u_red_guess, jnp.asarray([lambda_guess], dtype=u_red_guess.dtype)]
    )
    r = augmented_residual(z)
    initial_residual_norm = jnp.linalg.norm(r)
    converged0 = initial_residual_norm <= newton.tol

    if newton.matrix_free:
        def cond(carry):
            it, _, _, converged, terminated, _, _, _, _ = carry
            return (it < newton.max_iters) & (~converged) & (~terminated)

        def body(carry):
            it, z_local, r_local, _, _, _, _, _, prev_rnorm = carry
            rnorm = jnp.linalg.norm(r_local)
            if newton.eisenstat_walker:
                eta = _eisenstat_walker_eta(rnorm, prev_rnorm, newton)
            else:
                eta = newton.krylov_tol
            _, tangent_vec = jax.linearize(augmented_residual, z_local)

            def matvec(v: Array) -> Array:
                return tangent_vec(v) + newton.tangent_regularization * v

            dz, linear_ok, linear_iters, linear_relres = _gmres_solve(
                matvec=matvec,
                rhs=-r_local,
                tol=eta,
                maxiter=newton.krylov_maxiter,
            )
            z_candidate, r_candidate, accepted = _line_search_update(
                u_red=z_local,
                du=dz,
                residual=augmented_residual,
                line_search_steps=newton.line_search_steps,
                reduction=newton.line_search_factor,
            )

            def fallback(_):
                K = _finite_difference_tangent(
                    residual=augmented_residual,
                    u_red=z_local,
                    fd_eps=newton.fd_eps,
                    regularization=newton.tangent_regularization,
                )
                dz_dense = _solve_linear_system(K, -r_local)
                return _line_search_update(
                    u_red=z_local,
                    du=dz_dense,
                    residual=augmented_residual,
                    line_search_steps=newton.line_search_steps,
                    reduction=newton.line_search_factor,
                )

            z_new, r_new, accepted = jax.lax.cond(
                (~linear_ok) | (~accepted),
                fallback,
                lambda _: (z_candidate, r_candidate, accepted),
                operand=None,
            )
            residual_norm = jnp.linalg.norm(r_new)
            converged = accepted & (residual_norm <= newton.tol)
            terminated = ~accepted
            return (
                it + 1,
                z_new,
                r_new,
                converged,
                terminated,
                linear_ok,
                linear_iters,
                linear_relres,
                rnorm,
            )

        carry = jax.lax.while_loop(
            cond,
            body,
            (
                jnp.asarray(0, dtype=jnp.int32),
                z,
                r,
                converged0,
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=u_red_guess.dtype),
                initial_residual_norm,
            ),
        )
        (
            last_iter,
            z,
            r,
            converged,
            _terminated,
            last_linear_ok,
            last_linear_iters,
            last_linear_relres,
            _,
        ) = carry
    else:
        def cond(carry):
            it, _, _, converged, terminated, _, _, _ = carry
            return (it < newton.max_iters) & (~converged) & (~terminated)

        def body(carry):
            it, z_local, r_local, _, _, _, _, _ = carry
            K = _finite_difference_tangent(
                residual=augmented_residual,
                u_red=z_local,
                fd_eps=newton.fd_eps,
                regularization=newton.tangent_regularization,
            )
            dz = _solve_linear_system(K, -r_local)
            z_new, r_new, accepted = _line_search_update(
                u_red=z_local,
                du=dz,
                residual=augmented_residual,
                line_search_steps=newton.line_search_steps,
                reduction=newton.line_search_factor,
            )
            residual_norm = jnp.linalg.norm(r_new)
            converged = accepted & (residual_norm <= newton.tol)
            terminated = ~accepted
            return (
                it + 1,
                z_new,
                r_new,
                converged,
                terminated,
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=u_red_guess.dtype),
            )

        carry = jax.lax.while_loop(
            cond,
            body,
            (
                jnp.asarray(0, dtype=jnp.int32),
                z,
                r,
                converged0,
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=u_red_guess.dtype),
            ),
        )
        (
            last_iter,
            z,
            r,
            converged,
            _terminated,
            last_linear_ok,
            last_linear_iters,
            last_linear_relres,
        ) = carry

    lifter_step = lifter.at["right_ux"].set(z[-1])
    return lifter_step.lift_from_zeros(z[:-1]), float(z[-1]), {
        "converged": converged,
        "iterations": last_iter,
        "initial_residual_norm": initial_residual_norm,
        "final_residual_norm": jnp.linalg.norm(r),
        "linear_converged": last_linear_ok,
        "linear_iterations": last_linear_iters,
        "linear_relative_residual": jnp.asarray(
            last_linear_relres, dtype=u_red_guess.dtype
        ),
    }


def _line_search_update(
    *,
    u_red: Array,
    du: Array,
    residual,
    line_search_steps: int,
    reduction: float,
) -> tuple[Array, Array, bool]:
    base_residual = residual(u_red)
    base_norm = jnp.linalg.norm(base_residual)
    finite_du = jnp.all(jnp.isfinite(du))

    def do_search(_):
        def cond(carry):
            i, _, _, accepted = carry
            return (i < line_search_steps) & (~accepted)

        def body(carry):
            i, candidate, candidate_residual, accepted = carry
            alpha = jnp.power(jnp.asarray(reduction, dtype=u_red.dtype), i)
            trial = u_red + alpha * du
            trial_residual = residual(trial)
            trial_norm = jnp.linalg.norm(trial_residual)
            good = jnp.all(jnp.isfinite(trial_residual)) & (trial_norm < base_norm)

            def accept(_):
                return trial, trial_residual, jnp.asarray(True)

            def reject(_):
                return candidate, candidate_residual, accepted

            candidate, candidate_residual, accepted = jax.lax.cond(
                good, accept, reject, operand=None
            )
            return i + 1, candidate, candidate_residual, accepted

        _, candidate, candidate_residual, accepted = jax.lax.while_loop(
            cond,
            body,
            (
                jnp.asarray(0, dtype=jnp.int32),
                u_red,
                base_residual,
                jnp.asarray(False),
            ),
        )
        return candidate, candidate_residual, accepted

    return jax.lax.cond(
        finite_du,
        do_search,
        lambda _: (u_red, base_residual, jnp.asarray(False)),
        operand=None,
    )


def _predict_arc_length_step(
    *,
    u_red_prev: Array,
    lambda_prev: float,
    radius: float,
    direction: float,
    previous_delta_u_red: Array | None,
    previous_delta_lambda: float | None,
    lifter: Lifter,
    residual_full,
    state_n: HSPlaneStressState,
    material: HSPlaneStressMaterial,
    newton: NewtonSettings,
) -> tuple[Array, float, float]:
    secant_predictor = _secant_arc_length_predictor(
        u_red_prev=u_red_prev,
        lambda_prev=lambda_prev,
        radius=radius,
        direction=direction,
        previous_delta_u_red=previous_delta_u_red,
        previous_delta_lambda=previous_delta_lambda,
        scale=newton.arc_length_scale,
    )
    if secant_predictor is not None:
        return secant_predictor

    def reduced_residual(u_red_local: Array, lam_local: Array) -> Array:
        lam_array = jnp.asarray(lam_local, dtype=u_red_prev.dtype)
        lifter_step = lifter.at["right_ux"].set(lam_array)
        u_trial = lifter_step.lift_from_zeros(u_red_local)
        return lifter_step.reduce(residual_full(u_trial, state_n, material))

    lam_prev_arr = jnp.asarray(lambda_prev, dtype=u_red_prev.dtype)

    def residual_u_only(u_red_local: Array) -> Array:
        return reduced_residual(u_red_local, lam_prev_arr)

    _, tangent_u = jax.linearize(residual_u_only, u_red_prev)
    lambda_prev_arr = jnp.asarray(lambda_prev, dtype=u_red_prev.dtype)
    radius_arr = jnp.asarray(radius, dtype=u_red_prev.dtype)
    direction_arr = jnp.asarray(direction, dtype=u_red_prev.dtype)
    step = newton.fd_eps * jnp.maximum(1.0, jnp.maximum(jnp.abs(lambda_prev_arr), radius_arr))
    r_lam = (
        reduced_residual(u_red_prev, lam_prev_arr + step)
        - reduced_residual(u_red_prev, lam_prev_arr - step)
    ) / (2.0 * step)

    def matvec(v: Array) -> Array:
        return tangent_u(v) + newton.tangent_regularization * v

    if newton.matrix_free:
        du_dlam, linear_ok, _, _ = _gmres_solve(
            matvec=matvec,
            rhs=-r_lam,
            tol=newton.krylov_tol,
            maxiter=newton.krylov_maxiter,
        )
    else:
        K = _finite_difference_tangent(
            residual=residual_u_only,
            u_red=u_red_prev,
            fd_eps=newton.fd_eps,
            regularization=newton.tangent_regularization,
        )
        du_dlam = _solve_linear_system(K, -r_lam)
        linear_ok = True

    du_dlam = jnp.where(
        linear_ok & jnp.all(jnp.isfinite(du_dlam)),
        du_dlam,
        jnp.zeros_like(u_red_prev),
    )

    norm_factor = jnp.asarray(max(u_red_prev.shape[0], 1), dtype=u_red_prev.dtype)
    denom = jnp.sqrt(
        jnp.dot(du_dlam, du_dlam) / norm_factor
        + jnp.asarray(newton.arc_length_scale**2, dtype=u_red_prev.dtype)
    )
    delta_lambda = jnp.where(
        denom <= 1e-14,
        direction_arr * radius_arr,
        direction_arr * radius_arr / denom,
    )
    delta_u = jnp.where(
        denom <= 1e-14,
        jnp.zeros_like(u_red_prev),
        delta_lambda * du_dlam,
    )
    direction_new = jnp.where(
        jnp.abs(delta_lambda) > 1e-14,
        jnp.sign(delta_lambda),
        direction_arr,
    )

    return (
        u_red_prev + delta_u,
        lambda_prev_arr + delta_lambda,
        direction_new,
    )


def _secant_arc_length_predictor(
    *,
    u_red_prev: Array,
    lambda_prev: float,
    radius: float,
    direction: float,
    previous_delta_u_red: Array | None,
    previous_delta_lambda: float | None,
    scale: float,
) -> tuple[Array, float, float] | None:
    if previous_delta_u_red is None or previous_delta_lambda is None:
        return None

    du = jnp.asarray(previous_delta_u_red, dtype=u_red_prev.dtype)
    dlam = jnp.asarray(previous_delta_lambda, dtype=u_red_prev.dtype)
    if (not bool(jnp.isfinite(dlam))) or float(jnp.abs(dlam)) <= 1e-14:
        return None
    if not bool(jnp.all(jnp.isfinite(du))):
        return None

    norm_factor = jnp.asarray(max(u_red_prev.shape[0], 1), dtype=u_red_prev.dtype)
    radius_arr = jnp.asarray(radius, dtype=u_red_prev.dtype)
    direction_arr = jnp.asarray(direction, dtype=u_red_prev.dtype)
    lambda_prev_arr = jnp.asarray(lambda_prev, dtype=u_red_prev.dtype)
    secant_norm = jnp.sqrt(
        jnp.dot(du, du) / norm_factor + (jnp.asarray(scale, dtype=u_red_prev.dtype) * dlam) ** 2
    )
    if float(secant_norm) <= 1e-14:
        return None

    sign = jnp.where(jnp.sign(dlam) == 0.0, direction_arr, jnp.sign(dlam))
    alpha = direction_arr * radius_arr / secant_norm
    return (
        u_red_prev + alpha * du,
        lambda_prev_arr + alpha * dlam,
        sign,
    )


def _eisenstat_walker_eta(
    rnorm: Array, prev_rnorm: Array, newton: NewtonSettings
) -> Array:
    """Eisenstat-Walker Choice 2 forcing term for adaptive Krylov tolerance."""
    ratio = rnorm / jnp.maximum(prev_rnorm, 1e-30)
    return jnp.clip(0.9 * ratio ** 2, newton.krylov_tol, 0.5)


def _finite_difference_tangent(
    *,
    residual,
    u_red: Array,
    fd_eps: float,
    regularization: float,
) -> Array:
    n = u_red.shape[0]
    K0 = jnp.zeros((n, n), dtype=u_red.dtype)

    def body(i, K):
        step = fd_eps * jnp.maximum(1.0, jnp.abs(u_red[i]))
        u_fwd = u_red.at[i].add(step)
        u_bwd = u_red.at[i].add(-step)
        return K.at[:, i].set((residual(u_fwd) - residual(u_bwd)) / (2.0 * step))

    K = jax.lax.fori_loop(0, n, body, K0)
    return K + regularization * jnp.eye(n, dtype=K.dtype)


def _solve_linear_system(K: Array, rhs: Array) -> Array:
    sol = jnp.linalg.lstsq(K, rhs, rcond=None)[0]
    return jnp.where(jnp.all(jnp.isfinite(sol)), sol, jnp.zeros_like(rhs))


def _gmres_solve(
    *,
    matvec,
    rhs: Array,
    tol: float | Array,
    maxiter: int,
) -> tuple[Array, bool, int, Array]:
    tol = jnp.asarray(tol, dtype=rhs.dtype)
    beta = jnp.linalg.norm(rhs)
    m = max(1, min(maxiter, rhs.shape[0]))
    n = rhs.shape[0]

    def solve(_):
        V0 = jnp.zeros((n, m + 1), dtype=rhs.dtype).at[:, 0].set(rhs / beta)
        H0 = jnp.zeros((m + 1, m), dtype=rhs.dtype)
        cs0 = jnp.zeros((m,), dtype=rhs.dtype)
        sn0 = jnp.zeros((m,), dtype=rhs.dtype)
        g0 = jnp.zeros((m + 1,), dtype=rhs.dtype).at[0].set(beta)

        def iteration(j, state):
            V, H, cs, sn, g, converged, terminated, last_iter, last_relres = state

            def active_step(_):
                vj = V[:, j]
                w = matvec(vj)
                finite_w = jnp.all(jnp.isfinite(w))

                def finite_branch(_):
                    def arnoldi(i, inner):
                        w_local, H_local = inner

                        def update(_):
                            h_ij = jnp.dot(V[:, i], w_local)
                            w_new = w_local - h_ij * V[:, i]
                            H_new = H_local.at[i, j].set(h_ij)
                            return w_new, H_new

                        return jax.lax.cond(
                            i <= j, update, lambda _: (w_local, H_local), operand=None
                        )

                    w_arnoldi, H_arnoldi = jax.lax.fori_loop(0, m, arnoldi, (w, H))

                    def reorth(i, inner):
                        w_local, H_local = inner

                        def update(_):
                            h_corr = jnp.dot(V[:, i], w_local)
                            w_new = w_local - h_corr * V[:, i]
                            H_new = H_local.at[i, j].add(h_corr)
                            return w_new, H_new

                        return jax.lax.cond(
                            i <= j, update, lambda _: (w_local, H_local), operand=None
                        )

                    w_arnoldi, H_arnoldi = jax.lax.fori_loop(
                        0, m, reorth, (w_arnoldi, H_arnoldi)
                    )
                    h_next = jnp.linalg.norm(w_arnoldi)
                    H_arnoldi = H_arnoldi.at[j + 1, j].set(h_next)
                    V_arnoldi = jax.lax.cond(
                        (h_next > 1e-14) & ((j + 1) < (m + 1)),
                        lambda _: V.at[:, j + 1].set(w_arnoldi / h_next),
                        lambda _: V,
                        operand=None,
                    )

                    def apply_givens(i, H_local):
                        def rotate(_):
                            h_i = H_local[i, j]
                            h_ip1 = H_local[i + 1, j]
                            h_i_new = cs[i] * h_i + sn[i] * h_ip1
                            h_ip1_new = -sn[i] * h_i + cs[i] * h_ip1
                            H_local_new = H_local.at[i, j].set(h_i_new)
                            H_local_new = H_local_new.at[i + 1, j].set(h_ip1_new)
                            return H_local_new

                        return jax.lax.cond(i < j, rotate, lambda _: H_local, operand=None)

                    H_rot = jax.lax.fori_loop(0, m, apply_givens, H_arnoldi)
                    a = H_rot[j, j]
                    b = H_rot[j + 1, j]
                    r = jnp.sqrt(a * a + b * b)
                    c = jnp.where(r > 1e-14, a / r, jnp.asarray(1.0, dtype=rhs.dtype))
                    s = jnp.where(r > 1e-14, b / r, jnp.asarray(0.0, dtype=rhs.dtype))

                    H_rot = H_rot.at[j, j].set(c * a + s * b)
                    H_rot = H_rot.at[j + 1, j].set(0.0)
                    cs_new = cs.at[j].set(c)
                    sn_new = sn.at[j].set(s)

                    g_j = g[j]
                    g_jp1 = g[j + 1]
                    g_new = g.at[j].set(c * g_j + s * g_jp1)
                    g_new = g_new.at[j + 1].set(-s * g_j + c * g_jp1)

                    relres = jnp.abs(g_new[j + 1]) / jnp.maximum(
                        jnp.asarray(1.0, dtype=rhs.dtype), beta
                    )
                    converged_new = relres <= tol
                    terminated_new = converged_new | (h_next <= 1e-14)
                    return (
                        V_arnoldi,
                        H_rot,
                        cs_new,
                        sn_new,
                        g_new,
                        converged_new,
                        terminated_new,
                        jnp.asarray(j + 1, dtype=jnp.int32),
                        relres,
                    )

                def nonfinite_branch(_):
                    return (
                        V,
                        H,
                        cs,
                        sn,
                        g,
                        jnp.asarray(False),
                        jnp.asarray(True),
                        jnp.asarray(j + 1, dtype=jnp.int32),
                        jnp.asarray(jnp.inf, dtype=rhs.dtype),
                    )

                return jax.lax.cond(
                    finite_w, finite_branch, nonfinite_branch, operand=None
                )

            return jax.lax.cond(
                converged | terminated,
                lambda _: (
                    V,
                    H,
                    cs,
                    sn,
                    g,
                    converged,
                    terminated,
                    last_iter,
                    last_relres,
                ),
                active_step,
                operand=None,
            )

        (
            _V,
            _H,
            _cs,
            _sn,
            _g,
            converged,
            terminated,
            last_iter,
            last_relres,
        ) = jax.lax.fori_loop(
            0,
            m,
            iteration,
            (
                V0,
                H0,
                cs0,
                sn0,
                g0,
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(jnp.inf, dtype=rhs.dtype),
            ),
        )

        def build_solution(_):
            j = jnp.maximum(last_iter - 1, 0)
            active_mask = jnp.arange(m) <= j
            active_matrix = active_mask[:, None] & active_mask[None, :]
            identity = jnp.eye(m, dtype=rhs.dtype)
            R_eff = jnp.where(active_matrix, _H[:m, :m], identity)
            g_eff = jnp.where(active_mask, _g[:m], 0.0)
            y = _solve_linear_system(R_eff, g_eff)
            x = _V[:, :m] @ y
            finite = jnp.all(jnp.isfinite(x))
            relres = jnp.where(finite, last_relres, jnp.asarray(jnp.inf, dtype=rhs.dtype))
            return x, converged & finite, last_iter, relres

        return jax.lax.cond(
            (last_iter > 0) & (~terminated | converged),
            build_solution,
            lambda _: (
                jnp.zeros_like(rhs),
                jnp.asarray(False),
                last_iter,
                jnp.asarray(jnp.inf, dtype=rhs.dtype),
            ),
            operand=None,
        )

    return jax.lax.cond(
        beta <= tol,
        lambda _: (
            jnp.zeros_like(rhs),
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=rhs.dtype),
        ),
        solve,
        operand=None,
    )


def _broyden_update_tangent(
    K: Array,
    s: Array,
    y: Array,
    *,
    regularization: float,
) -> Array:
    denom = jnp.dot(s, s)

    def update(_):
        correction = jnp.outer(y - K @ s, s) / denom
        return K + correction + regularization * jnp.eye(K.shape[0], dtype=K.dtype)

    return jax.lax.cond(denom <= 1e-20, lambda _: K, update, operand=None)


def _scaled_displacement_guess(
    *,
    u_full: Array,
    current_displacement: float,
    target_displacement: float,
    lifter: Lifter,
) -> Array:
    if abs(current_displacement) < 1e-14:
        return lifter.at["right_ux"].set(target_displacement).lift_from_zeros(
            lifter.reduce(u_full)
        )
    scale = target_displacement / current_displacement
    return lifter.at["right_ux"].set(target_displacement).lift_from_zeros(
        lifter.reduce(u_full * scale)
    )


def _step_is_valid(
    *,
    newton: NewtonSettings,
    step_info: dict[str, Array],
    u_full: Array,
    residual: Array,
    state: HSPlaneStressState,
) -> bool:
    finite = bool(
        jnp.all(jnp.isfinite(u_full))
        & jnp.all(jnp.isfinite(residual))
        & jnp.all(jnp.isfinite(state.epbar))
        & jnp.all(jnp.isfinite(state.Fp))
    )
    if int(step_info["iterations"]) == 0:
        return finite
    if bool(step_info["converged"]):
        return finite
    if not newton.allow_inexact_steps:
        return False
    initial = float(step_info["initial_residual_norm"])
    final = float(step_info["final_residual_norm"])
    if initial <= 1e-30:
        return finite
    return finite and (final / initial <= newton.accepted_residual_ratio)


def _step_info_value(step_info: dict[str, Array], key: str) -> Array:
    value = step_info.get(key)
    if value is None:
        return jnp.asarray(jnp.nan)
    return jnp.asarray(value)


def _measure_current_widths(specimen: DogboneMesh, u: Array) -> tuple[Array, Array]:
    current = specimen.mesh.coords + u
    y_columns = current[specimen.node_grid, 1]
    widths = jnp.max(y_columns, axis=1) - jnp.min(y_columns, axis=1)
    return (
        jnp.min(widths[specimen.gauge_column_mask]),
        jnp.min(widths[specimen.center_column_mask]),
    )


def _total_length(geometry: DogboneGeometry) -> float:
    return geometry.gauge_length + 2.0 * (
        geometry.transition_length + geometry.grip_length
    )


def _half_width(x: Array, geometry: DogboneGeometry) -> Array:
    x_abs = jnp.abs(x)
    x_gauge = 0.5 * geometry.gauge_length
    x_transition = x_gauge + geometry.transition_length

    blend = jnp.clip((x_abs - x_gauge) / geometry.transition_length, 0.0, 1.0)
    smooth = 0.5 - 0.5 * jnp.cos(jnp.pi * blend)
    width = jnp.where(
        x_abs <= x_gauge,
        geometry.gauge_width,
        jnp.where(
            x_abs <= x_transition,
            geometry.gauge_width + smooth * (geometry.grip_width - geometry.gauge_width),
            geometry.grip_width,
        ),
    )
    imperfection = 1.0 - geometry.imperfection_depth * jnp.exp(
        -(x / geometry.imperfection_length) ** 2
    )
    return 0.5 * width * imperfection


def _node_component_dofs(nodes: Array, component: int) -> Array:
    return 2 * nodes + component
