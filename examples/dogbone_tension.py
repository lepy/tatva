from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tatva.material import (
    DogboneGeometry,
    Hill48PlaneStressMaterial,
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
    parser.add_argument("--increments", type=int, default=16)
    parser.add_argument("--final-displacement", type=float, default=15.0)
    parser.add_argument("--nx-grip", type=int, default=2)
    parser.add_argument("--nx-transition", type=int, default=2)
    parser.add_argument("--nx-gauge", type=int, default=10)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--newton-iters", type=int, default=10)
    parser.add_argument("--newton-tol", type=float, default=1e-6)
    parser.add_argument("--line-search-steps", type=int, default=6)
    parser.add_argument(
        "--yield-model",
        choices=("isotropic", "hill48"),
        default="isotropic",
    )
    parser.add_argument(
        "--arc-length",
        action="store_true",
        help="Use arc-length continuation instead of pure displacement stepping.",
    )
    parser.add_argument(
        "--arc-radius",
        type=float,
        default=0.0,
        help="Initial arc-length radius. Default derives it from final displacement / increments.",
    )
    parser.add_argument(
        "--arc-scale",
        type=float,
        default=1.0,
        help="Weight for the load-parameter part of the arc-length constraint.",
    )
    parser.add_argument(
        "--arc-growth-factor",
        type=float,
        default=1.25,
        help="Radius growth factor after easy converged arc-length steps.",
    )
    parser.add_argument(
        "--arc-shrink-factor",
        type=float,
        default=0.5,
        help="Radius shrink factor after rejected or difficult arc-length steps.",
    )
    parser.add_argument(
        "--arc-max-steps-factor",
        type=int,
        default=8,
        help="Maximum accepted arc-length steps as a multiple of the nominal increment count.",
    )
    parser.add_argument("--sigma0", type=float, default=5.0)
    parser.add_argument("--sigma-inf", type=float, default=12.0)
    parser.add_argument("--hardening-m", type=float, default=8.0)
    parser.add_argument("--hardening-n", type=float, default=0.7)
    parser.add_argument("--hill48-f", type=float, default=0.5)
    parser.add_argument("--hill48-g", type=float, default=0.5)
    parser.add_argument("--hill48-h", type=float, default=0.5)
    parser.add_argument("--hill48-n", type=float, default=1.5)
    parser.add_argument("--grip-length", type=float, default=20.0)
    parser.add_argument("--transition-length", type=float, default=12.0)
    parser.add_argument("--gauge-length", type=float, default=50.0)
    parser.add_argument("--grip-width", type=float, default=20.0)
    parser.add_argument("--gauge-width", type=float, default=10.0)
    parser.add_argument("--imperfection-depth", type=float, default=0.05)
    parser.add_argument("--imperfection-length", type=float, default=10.0)
    parser.add_argument("--min-displacement-increment", type=float, default=3e-3)
    parser.add_argument("--max-cutbacks", type=int, default=12)
    parser.add_argument("--fd-eps", type=float, default=1e-6)
    parser.add_argument("--tangent-regularization", type=float, default=1e-8)
    parser.add_argument("--tangent-rebuild-interval", type=int, default=6)
    parser.add_argument(
        "--dense-fd-tangent",
        action="store_true",
        help="Use the dense finite-difference tangent instead of the default matrix-free Krylov solve.",
    )
    parser.add_argument(
        "--full-newton",
        action="store_true",
        help="Disable the cheaper modified-Newton pre-peak path and always use the full displacement-control Newton solve.",
    )
    parser.add_argument("--krylov-tol", type=float, default=1e-6)
    parser.add_argument("--krylov-maxiter", type=int, default=40)
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

    if args.yield_model == "hill48":
        material = Hill48PlaneStressMaterial(
            mu=80.0,
            kappa=160.0,
            sigma0=args.sigma0,
            sigma_inf=args.sigma_inf,
            m=args.hardening_m,
            n=args.hardening_n,
            hill_f=args.hill48_f,
            hill_g=args.hill48_g,
            hill_h=args.hill48_h,
            hill_n=args.hill48_n,
            newton_maxiter=20,
        )
    else:
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
        use_arc_length=args.arc_length,
        arc_length_radius=args.arc_radius,
        arc_length_scale=args.arc_scale,
        arc_length_growth_factor=args.arc_growth_factor,
        arc_length_shrink_factor=args.arc_shrink_factor,
        arc_length_max_steps_factor=args.arc_max_steps_factor,
        min_displacement_increment=args.min_displacement_increment,
        max_cutbacks=args.max_cutbacks,
        fd_eps=args.fd_eps,
        tangent_regularization=args.tangent_regularization,
        tangent_rebuild_interval=args.tangent_rebuild_interval,
        matrix_free=not args.dense_fd_tangent,
        krylov_tol=args.krylov_tol,
        krylov_maxiter=args.krylov_maxiter,
        fast_prepeak=not args.full_newton,
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
            "arc_radius": np.asarray(history.arc_radius),
            "cutbacks": np.asarray(history.cutbacks),
            "initial_residual_norm": np.asarray(history.initial_residual_norm),
            "final_residual_norm": np.asarray(history.final_residual_norm),
            "linear_converged": np.asarray(history.linear_converged),
            "linear_iterations": np.asarray(history.linear_iterations),
            "linear_relative_residual": np.asarray(history.linear_relative_residual),
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
