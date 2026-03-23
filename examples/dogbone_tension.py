from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tatva.material import (
    DogboneGeometry,
    HSPlaneStressMaterial,
    NewtonSettings,
    simulate_dogbone_tension,
)

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a finite-strain plane-stress dogbone tension simulation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/dogbone"))
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--final-displacement", type=float, default=2.0)
    parser.add_argument("--nx-grip", type=int, default=4)
    parser.add_argument("--nx-transition", type=int, default=4)
    parser.add_argument("--nx-gauge", type=int, default=10)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--newton-iters", type=int, default=8)
    parser.add_argument("--newton-tol", type=float, default=1e-6)
    parser.add_argument("--line-search-steps", type=int, default=6)
    parser.add_argument("--sigma0", type=float, default=10.0)
    parser.add_argument("--sigma-inf", type=float, default=120.0)
    parser.add_argument("--hardening-m", type=float, default=14.0)
    parser.add_argument("--hardening-n", type=float, default=0.7)
    parser.add_argument("--grip-length", type=float, default=8.0)
    parser.add_argument("--transition-length", type=float, default=5.0)
    parser.add_argument("--gauge-length", type=float, default=18.0)
    parser.add_argument("--grip-width", type=float, default=10.0)
    parser.add_argument("--gauge-width", type=float, default=5.0)
    parser.add_argument("--imperfection-depth", type=float, default=0.02)
    parser.add_argument("--imperfection-length", type=float, default=3.0)
    parser.add_argument("--min-displacement-increment", type=float, default=1e-3)
    parser.add_argument("--max-cutbacks", type=int, default=12)
    parser.add_argument("--fd-eps", type=float, default=1e-6)
    parser.add_argument("--tangent-regularization", type=float, default=1e-8)
    parser.add_argument("--tangent-rebuild-interval", type=int, default=6)
    parser.add_argument(
        "--allow-inexact-steps",
        action="store_true",
        help="Allow accepting non-converged increments if the residual drops sufficiently.",
    )
    parser.add_argument("--accepted-residual-ratio", type=float, default=0.25)
    parser.add_argument("--plot-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    material = HSPlaneStressMaterial(
        mu=80.0,
        kappa=160.0,
        sigma0=args.sigma0,
        sigma_inf=args.sigma_inf,
        m=args.hardening_m,
        n=args.hardening_n,
        newton_maxiter=20,
    )
    geometry = DogboneGeometry(
        grip_length=args.grip_length,
        transition_length=args.transition_length,
        gauge_length=args.gauge_length,
        grip_width=args.grip_width,
        gauge_width=args.gauge_width,
        imperfection_depth=args.imperfection_depth,
        imperfection_length=args.imperfection_length,
    )
    newton = NewtonSettings(
        max_iters=args.newton_iters,
        tol=args.newton_tol,
        line_search_steps=args.line_search_steps,
        min_displacement_increment=args.min_displacement_increment,
        max_cutbacks=args.max_cutbacks,
        fd_eps=args.fd_eps,
        tangent_regularization=args.tangent_regularization,
        tangent_rebuild_interval=args.tangent_rebuild_interval,
        allow_inexact_steps=args.allow_inexact_steps,
        accepted_residual_ratio=args.accepted_residual_ratio,
    )

    result = simulate_dogbone_tension(
        material,
        geometry=geometry,
        n_x_grip=args.nx_grip,
        n_x_transition=args.nx_transition,
        n_x_gauge=args.nx_gauge,
        n_y=args.ny,
        n_increments=args.increments,
        final_displacement=args.final_displacement,
        newton=newton,
    )

    _save_dataframes(args.output_dir, result)
    _save_plots(args.output_dir, result, plot_scale=args.plot_scale)


def _save_dataframes(output_dir: Path, result) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This example requires pandas for tabular output. Install `pandas`."
        ) from exc

    history = result.history
    history_df = pd.DataFrame(
        {
            "displacement": np.asarray(history.displacement),
            "reaction_force": np.asarray(history.reaction_force),
            "accepted_increment": np.asarray(history.accepted_increment),
            "min_gauge_width": np.asarray(history.min_gauge_width),
            "min_gauge_width_ratio": np.asarray(history.min_gauge_width_ratio),
            "center_width": np.asarray(history.center_width),
            "center_width_ratio": np.asarray(history.center_width_ratio),
            "max_epbar": np.asarray(history.max_epbar),
            "converged": np.asarray(history.converged),
            "iterations": np.asarray(history.iterations),
        }
    )
    history_df.to_pickle(output_dir / "history.pkl")
    history_df.to_csv(output_dir / "history.csv", index=False)

    coords = np.asarray(result.specimen.mesh.coords)
    u = np.asarray(result.u)
    current = coords + u
    nodal_df = pd.DataFrame(
        {
            "node_id": np.arange(coords.shape[0], dtype=np.int32),
            "x_ref": coords[:, 0],
            "y_ref": coords[:, 1],
            "ux": u[:, 0],
            "uy": u[:, 1],
            "x_cur": current[:, 0],
            "y_cur": current[:, 1],
        }
    )
    nodal_df.to_pickle(output_dir / "nodal_results.pkl")
    nodal_df.to_csv(output_dir / "nodal_results.csv", index=False)

    snapshot_rows = []
    for step_id, (disp, force, u_step) in enumerate(
        zip(
            np.asarray(history.displacement),
            np.asarray(history.reaction_force),
            np.asarray(result.u_history),
            strict=True,
        )
    ):
        current_step = coords + u_step
        for node_id, (x_ref, y_ref), (ux, uy), (x_cur, y_cur) in zip(
            np.arange(coords.shape[0], dtype=np.int32),
            coords,
            u_step,
            current_step,
            strict=True,
        ):
            snapshot_rows.append(
                {
                    "step_id": step_id,
                    "displacement": disp,
                    "reaction_force": force,
                    "node_id": int(node_id),
                    "x_ref": x_ref,
                    "y_ref": y_ref,
                    "ux": ux,
                    "uy": uy,
                    "x_cur": x_cur,
                    "y_cur": y_cur,
                }
            )
    snapshot_df = pd.DataFrame(snapshot_rows)
    snapshot_df.to_pickle(output_dir / "nodal_snapshots.pkl")
    snapshot_df.to_csv(output_dir / "nodal_snapshots.csv", index=False)

    elements = np.asarray(result.specimen.mesh.elements)
    element_df = pd.DataFrame({"element_id": np.arange(elements.shape[0], dtype=np.int32)})
    for i in range(elements.shape[1]):
        element_df[f"n{i}"] = elements[:, i]
    element_df.to_pickle(output_dir / "elements.pkl")
    element_df.to_csv(output_dir / "elements.csv", index=False)


def _save_plots(output_dir: Path, result, *, plot_scale: float) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This example requires matplotlib. Install `tatva[plotting]` or `matplotlib`."
        ) from exc

    history = result.history

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        np.asarray(history.displacement),
        np.asarray(history.reaction_force),
        lw=2.0,
        marker="o",
    )
    ax.set_xlabel("Prescribed displacement")
    ax.set_ylabel("Reaction force")
    ax.set_title("Dogbone force-displacement")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "force_displacement.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        np.asarray(history.displacement),
        np.asarray(history.min_gauge_width_ratio),
        label="Gauge width ratio",
        lw=2.0,
        marker="o",
    )
    ax.plot(
        np.asarray(history.displacement),
        np.asarray(history.center_width_ratio),
        label="Center width ratio",
        lw=2.0,
        marker="o",
    )
    ax.set_xlabel("Prescribed displacement")
    ax.set_ylabel("Current width / initial width")
    ax.set_title("Necking evolution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "necking_history.png", dpi=200)
    plt.close(fig)

    coords = np.asarray(result.specimen.mesh.coords)
    u = np.asarray(result.u)
    deformed = coords + plot_scale * u
    segments = _unique_edges(coords, np.asarray(result.specimen.mesh.elements))
    deformed_segments = deformed[segments]
    cmap_values = np.linalg.norm(u[segments].mean(axis=1), axis=1)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    lc = LineCollection(deformed_segments, cmap="viridis", linewidths=1.0)
    lc.set_array(cmap_values)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("Deformed dogbone mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(lc, ax=ax, label="Displacement magnitude")
    cbar.solids.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "deformed_mesh.png", dpi=200)
    plt.close(fig)

    steps_dir = output_dir / "steps"
    steps_dir.mkdir(exist_ok=True)
    for step_id, (disp, force, u_step) in enumerate(
        zip(
            np.asarray(history.displacement),
            np.asarray(history.reaction_force),
            np.asarray(result.u_history),
            strict=True,
        )
    ):
        step_deformed = coords + plot_scale * u_step
        step_segments = step_deformed[segments]
        step_cmap = np.linalg.norm(u_step[segments].mean(axis=1), axis=1)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        lc = LineCollection(step_segments, cmap="viridis", linewidths=1.5)
        lc.set_array(step_cmap)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(
            f"Step {step_id:02d}: u={disp:.4f}, R={force:.4f}",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(lc, ax=ax, label="Displacement magnitude")
        fig.tight_layout()
        fig.savefig(steps_dir / f"step_{step_id:03d}.png", dpi=200)
        plt.close(fig)


def _unique_edges(coords: np.ndarray, elements: np.ndarray) -> np.ndarray:
    del coords
    edges: set[tuple[int, int]] = set()
    for elem in elements:
        if len(elem) == 3:
            local_edges = ((0, 1), (1, 2), (2, 0))
        elif len(elem) == 4:
            local_edges = ((0, 1), (1, 2), (2, 3), (3, 0))
        else:
            raise ValueError("Only triangular and quadrilateral elements are supported.")
        for i, j in local_edges:
            a = int(elem[i])
            b = int(elem[j])
            edges.add((a, b) if a < b else (b, a))
    return np.asarray(sorted(edges), dtype=np.int32)


if __name__ == "__main__":
    main()
