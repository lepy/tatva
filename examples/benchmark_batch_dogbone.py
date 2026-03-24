"""Benchmark: batched dogbone simulations with varying Hockett-Sherby parameters.

Uses ``jax.vmap`` over material parameters for efficient parallel evaluation.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tatva.material import (
    DogboneGeometry,
    HSPlaneStressMaterial,
    NewtonSettings,
    make_jit_dogbone_solver,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch dogbone benchmark")
    p.add_argument("--n-sims", type=int, default=10_000)
    p.add_argument("--nx-grip", type=int, default=2)
    p.add_argument("--nx-transition", type=int, default=2)
    p.add_argument("--nx-gauge", type=int, default=6)
    p.add_argument("--ny", type=int, default=2)
    p.add_argument("--increments", type=int, default=10)
    p.add_argument("--final-displacement", type=float, default=1.0)
    p.add_argument("--newton-iters", type=int, default=8)
    p.add_argument("--krylov-maxiter", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=0, help="0 = all at once")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_random_materials(n: int, seed: int) -> HSPlaneStressMaterial:
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    sigma0 = jax.random.uniform(k1, (n,), minval=2.0, maxval=20.0)
    sigma_inf = sigma0 + jax.random.uniform(k2, (n,), minval=10.0, maxval=200.0)
    m_vals = jax.random.uniform(k3, (n,), minval=1.0, maxval=30.0)
    n_vals = jax.random.uniform(k4, (n,), minval=0.3, maxval=1.5)

    return HSPlaneStressMaterial(
        mu=jnp.full(n, 80.0),
        kappa=jnp.full(n, 160.0),
        sigma0=sigma0,
        sigma_inf=sigma_inf,
        m=m_vals,
        n=n_vals,
        newton_maxiter=jnp.full(n, 20, dtype=jnp.int32),
        newton_tol=jnp.full(n, 1e-10),
        plane_stress_eps=jnp.full(n, 1e-6),
    )


def main() -> None:
    args = parse_args()

    geometry = DogboneGeometry(
        grip_length=20.0,
        transition_length=12.0,
        gauge_length=50.0,
        grip_width=20.0,
        gauge_width=10.0,
        imperfection_depth=0.05,
        imperfection_length=10.0,
    )
    newton = NewtonSettings(
        max_iters=args.newton_iters,
        tol=1e-6,
        line_search_steps=4,
        matrix_free=True,
        krylov_tol=1e-4,
        krylov_maxiter=args.krylov_maxiter,
        eisenstat_walker=True,
        tangent_regularization=1e-8,
    )

    print(f"Building solver (mesh: {args.nx_gauge}x{args.ny}) ...")
    solver, specimen = make_jit_dogbone_solver(
        geometry,
        n_x_grip=args.nx_grip,
        n_x_transition=args.nx_transition,
        n_x_gauge=args.nx_gauge,
        n_y=args.ny,
        n_increments=args.increments,
        final_displacement=args.final_displacement,
        newton=newton,
    )

    n_el = int(specimen.mesh.elements.shape[0])
    n_nodes = int(specimen.mesh.coords.shape[0])
    print(f"  {n_el} elements, {n_nodes} nodes, {n_nodes * 2} DOFs")

    # --- Warm-up: compile for single simulation ---
    print("Compiling single solver ...")
    single_mat = HSPlaneStressMaterial(
        mu=80.0, kappa=160.0, sigma0=5.0, sigma_inf=50.0, m=8.0, n=0.7,
    )
    solve_one = jax.jit(solver)
    t0 = time.perf_counter()
    warmup = solve_one(single_mat)
    warmup.reaction_force.block_until_ready()
    t_compile_single = time.perf_counter() - t0
    print(f"  compile: {t_compile_single:.1f}s")

    # --- Warm-up: compile for batched ---
    print(f"Compiling batched solver (vmap) ...")
    solve_batch = jax.jit(jax.vmap(solver))
    small_batch = make_random_materials(2, args.seed)
    t0 = time.perf_counter()
    warmup_batch = solve_batch(small_batch)
    warmup_batch.reaction_force.block_until_ready()
    t_compile_batch = time.perf_counter() - t0
    print(f"  compile: {t_compile_batch:.1f}s")

    # --- Run benchmark ---
    n_sims = args.n_sims
    batch_size = args.batch_size if args.batch_size > 0 else n_sims
    materials = make_random_materials(n_sims, args.seed)

    print(f"\nRunning {n_sims} simulations (batch_size={batch_size}) ...")
    t_start = time.perf_counter()

    if batch_size >= n_sims:
        results = solve_batch(materials)
        results.reaction_force.block_until_ready()
    else:
        # Process in batches
        all_converged = []
        all_force = []
        for i in range(0, n_sims, batch_size):
            end = min(i + batch_size, n_sims)
            batch_mat = jax.tree.map(lambda x: x[i:end], materials)
            batch_res = solve_batch(batch_mat)
            batch_res.reaction_force.block_until_ready()
            all_converged.append(np.asarray(batch_res.converged))
            all_force.append(np.asarray(batch_res.reaction_force))
            print(f"  batch {i}-{end} done")
        results = None

    t_total = time.perf_counter() - t_start

    # --- Report ---
    if results is not None:
        converged = np.asarray(results.converged)
        forces = np.asarray(results.reaction_force)
        iters = np.asarray(results.iterations)
    else:
        converged = np.concatenate(all_converged, axis=0)
        forces = np.concatenate(all_force, axis=0)
        iters = None

    all_steps_converged = converged.all(axis=1).sum()
    any_step_failed = (~converged).any(axis=1).sum()
    step_convergence_rate = converged.mean() * 100

    print(f"\n{'='*50}")
    print(f"  Simulationen:        {n_sims}")
    print(f"  Inkremente/Sim:      {args.increments}")
    print(f"  Mesh:                {n_el} Elemente, {n_nodes * 2} DOFs")
    print(f"  Gesamtzeit:          {t_total:.2f} s")
    print(f"  Zeit/Simulation:     {t_total / n_sims * 1000:.2f} ms")
    print(f"  Konvergenzrate:      {step_convergence_rate:.1f}% aller Schritte")
    print(f"  Voll konvergiert:    {all_steps_converged}/{n_sims}")
    print(f"  Mind. 1 Schritt fail:{any_step_failed}/{n_sims}")
    print(f"  Max Reaktionskraft:  {forces[:, -1].max():.2f}")
    print(f"  Min Reaktionskraft:  {forces[:, -1].min():.2f}")
    if iters is not None:
        print(f"  Mittl. Newton-Iter:  {iters[converged].mean():.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
