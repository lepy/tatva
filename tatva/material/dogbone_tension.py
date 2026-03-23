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
    fd_eps: float = 1e-6
    tangent_regularization: float = 1e-8
    tangent_rebuild_interval: int = 6
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

    u_red = lifter.reduce(u_full)
    last_iter = 0
    converged = False

    if newton.max_iters <= 0:
        return lifter.lift_from_zeros(u_red), {
            "converged": jnp.asarray(False),
            "iterations": jnp.asarray(0),
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

    r = reduced_residual(u_red)
    initial_residual_norm = jnp.linalg.norm(r)
    if float(jnp.linalg.norm(r)) < newton.tol:
        return lifter.lift_from_zeros(u_red), {
            "converged": jnp.asarray(True),
            "iterations": jnp.asarray(0),
            "initial_residual_norm": initial_residual_norm,
            "final_residual_norm": initial_residual_norm,
        }

    K = reduced_tangent(u_red)
    for it in range(newton.max_iters):
        last_iter = it + 1
        if (it > 0) and (newton.tangent_rebuild_interval > 0):
            if (it % newton.tangent_rebuild_interval) == 0:
                K = reduced_tangent(u_red)

        du = _solve_linear_system(K, -r)
        u_red_candidate, r_candidate, accepted = _line_search_update(
            u_red=u_red,
            du=du,
            residual=reduced_residual,
            line_search_steps=newton.line_search_steps,
            reduction=newton.line_search_factor,
        )
        if not accepted:
            K = reduced_tangent(u_red)
            du = _solve_linear_system(K, -r)
            u_red_candidate, r_candidate, accepted = _line_search_update(
                u_red=u_red,
                du=du,
                residual=reduced_residual,
                line_search_steps=newton.line_search_steps,
                reduction=newton.line_search_factor,
            )
            if not accepted:
                break

        s = u_red_candidate - u_red
        y = r_candidate - r
        u_red = u_red_candidate
        r = r_candidate
        if float(jnp.linalg.norm(r)) < newton.tol:
            converged = True
            break
        K = _broyden_update_tangent(
            K,
            s,
            y,
            regularization=newton.tangent_regularization,
        )

    u_full_new = lifter.lift_from_zeros(u_red)
    if last_iter > 0 and float(jnp.linalg.norm(r)) < newton.tol:
        converged = True
    return u_full_new, {
        "converged": jnp.asarray(converged),
        "iterations": jnp.asarray(last_iter),
        "initial_residual_norm": initial_residual_norm,
        "final_residual_norm": jnp.linalg.norm(r),
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
    candidate = u_red + du
    candidate_residual = residual(candidate)
    accepted = False

    for i in range(line_search_steps):
        alpha = reduction**i
        trial = u_red + alpha * du
        trial_residual = residual(trial)
        trial_norm = jnp.linalg.norm(trial_residual)
        if bool(jnp.all(jnp.isfinite(trial_residual))) and float(trial_norm) < float(
            base_norm
        ):
            candidate = trial
            candidate_residual = trial_residual
            accepted = True
            break

    return candidate, candidate_residual, accepted


def _finite_difference_tangent(
    *,
    residual,
    u_red: Array,
    fd_eps: float,
    regularization: float,
) -> Array:
    r0 = np.asarray(residual(u_red))
    n = u_red.shape[0]
    K = np.zeros((n, n), dtype=r0.dtype)
    u_np = np.asarray(u_red)

    for i in range(n):
        step = fd_eps * max(1.0, abs(float(u_np[i])))
        u_pert = u_np.copy()
        u_pert[i] += step
        ri = np.asarray(residual(jnp.asarray(u_pert, dtype=u_red.dtype)))
        K[:, i] = (ri - r0) / step

    K += regularization * np.eye(n, dtype=K.dtype)
    return jnp.asarray(K)


def _solve_linear_system(K: Array, rhs: Array) -> Array:
    sol, *_ = np.linalg.lstsq(np.asarray(K), np.asarray(rhs), rcond=None)
    return jnp.asarray(sol, dtype=rhs.dtype)


def _broyden_update_tangent(
    K: Array,
    s: Array,
    y: Array,
    *,
    regularization: float,
) -> Array:
    s_np = np.asarray(s)
    denom = float(np.dot(s_np, s_np))
    if denom <= 1e-20:
        return K

    K_np = np.asarray(K)
    y_np = np.asarray(y)
    correction = np.outer(y_np - K_np @ s_np, s_np) / denom
    K_new = K_np + correction
    K_new += regularization * np.eye(K_new.shape[0], dtype=K_new.dtype)
    return jnp.asarray(K_new, dtype=K.dtype)


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
