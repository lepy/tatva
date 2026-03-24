from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tatva.element as element
from tatva.material import (
    DogboneGeometry,
    HSPlaneStressMaterial,
    NewtonSettings,
    build_dogbone_lifter,
    build_dogbone_mesh,
    evaluate_plane_stress_responses,
    make_initial_state_field,
    make_plane_stress_residual_flat,
    simulate_dogbone_tension,
)
from tatva.operator import Operator


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the dogbone material kernel and a small end-to-end solve on the current JAX backend."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmark_dogbone_backend.json"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nx-grip", type=int, default=1)
    parser.add_argument("--nx-transition", type=int, default=1)
    parser.add_argument("--nx-gauge", type=int, default=6)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--increments", type=int, default=1)
    parser.add_argument("--final-displacement", type=float, default=0.05)
    parser.add_argument("--newton-iters", type=int, default=4)
    parser.add_argument("--newton-tol", type=float, default=1e-5)
    return parser.parse_args()


def _time_call(fn, *, repeats: int):
    t0 = time.perf_counter()
    out = fn()
    _block_output(out)
    first = time.perf_counter() - t0

    times: list[float] = []
    last_out = out
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        _block_output(out)
        times.append(time.perf_counter() - t0)
        last_out = out
    mean = float(np.mean(times)) if times else first
    return first, mean, last_out


def _block_output(out) -> None:
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
        return
    if isinstance(out, tuple):
        for item in out:
            _block_output(item)
        return
    if hasattr(out, "history") and hasattr(out.history, "reaction_force"):
        out.history.reaction_force.block_until_ready()
        return
    if hasattr(out, "state") and hasattr(out.state, "epbar"):
        out.state.epbar.block_until_ready()
        return


def benchmark_backend(args: argparse.Namespace) -> dict[str, object]:
    geometry = DogboneGeometry(
        grip_length=20.0,
        transition_length=12.0,
        gauge_length=50.0,
        grip_width=20.0,
        gauge_width=10.0,
        imperfection_depth=0.05,
        imperfection_length=10.0,
    )
    material = HSPlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=5.0,
        sigma_inf=12.0,
        m=8.0,
        n=0.7,
        newton_maxiter=20,
    )
    specimen = build_dogbone_mesh(
        geometry,
        n_x_grip=args.nx_grip,
        n_x_transition=args.nx_transition,
        n_x_gauge=args.nx_gauge,
        n_y=args.ny,
    )
    op = Operator(specimen.mesh, element.Quad4())
    lifter = build_dogbone_lifter(specimen)
    state = make_initial_state_field(op, dtype=specimen.mesh.coords.dtype)
    u = jnp.zeros((specimen.mesh.coords.shape[0], 2), dtype=specimen.mesh.coords.dtype)
    u_full = jnp.zeros((specimen.mesh.coords.shape[0] * 2,), dtype=specimen.mesh.coords.dtype)
    residual = make_plane_stress_residual_flat(op, jit=True)
    responses = jax.jit(
        lambda u_in, state_n, mat: evaluate_plane_stress_responses(op, u_in, state_n, mat)
    )
    u_flat = u.reshape(-1)

    residual_first, residual_mean, _ = _time_call(
        lambda: residual(u_flat, state, material), repeats=args.repeats
    )
    response_first, response_mean, _ = _time_call(
        lambda: responses(u, state, material), repeats=args.repeats
    )

    newton = NewtonSettings(
        max_iters=args.newton_iters,
        tol=args.newton_tol,
        line_search_steps=4,
        use_arc_length=False,
        matrix_free=True,
        fast_prepeak=True,
        krylov_tol=1e-6,
        krylov_maxiter=12,
    )
    simulation_first, simulation_mean, result = _time_call(
        lambda: simulate_dogbone_tension(
            material,
            geometry=geometry,
            n_x_grip=args.nx_grip,
            n_x_transition=args.nx_transition,
            n_x_gauge=args.nx_gauge,
            n_y=args.ny,
            n_increments=args.increments,
            final_displacement=args.final_displacement,
            newton=newton,
        ),
        repeats=max(1, min(3, args.repeats)),
    )
    _block_output(result)

    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "mesh": {
            "elements": int(specimen.mesh.elements.shape[0]),
            "nodes": int(specimen.mesh.coords.shape[0]),
            "quadrature_points_per_element": int(op.element.quad_points.shape[0]),
            "total_dofs": int(u_full.shape[0]),
            "reduced_dofs": int(lifter.reduce(u_full).shape[0]),
            "n_x_grip": int(args.nx_grip),
            "n_x_transition": int(args.nx_transition),
            "n_x_gauge": int(args.nx_gauge),
            "n_y": int(args.ny),
        },
        "kernel": {
            "residual_first_call_s": residual_first,
            "residual_mean_warm_s": residual_mean,
            "responses_first_call_s": response_first,
            "responses_mean_warm_s": response_mean,
        },
        "simulation_prepeak": {
            "increments": int(args.increments),
            "final_displacement": float(args.final_displacement),
            "first_run_s": simulation_first,
            "mean_warm_run_s": simulation_mean,
            "converged_last": bool(np.asarray(result.history.converged)[-1]),
            "iterations_last": int(np.asarray(result.history.iterations)[-1]),
            "final_residual_last": float(np.asarray(result.history.final_residual_norm)[-1]),
            "reaction_force_last": float(np.asarray(result.history.reaction_force)[-1]),
        },
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = benchmark_backend(args)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
