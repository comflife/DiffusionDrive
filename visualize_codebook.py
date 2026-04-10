"""
Regenerate codebook visualisation from an existing .pkl file.

Usage:
    python visualize_codebook.py
    python visualize_codebook.py --pkl codebook_cache/agent_vocab_navsim_kdisk.pkl
"""

import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def visualize_codebook(ret_traj, output_dir, n_highlight=300):
    """
    AutoVLA Fig S1(a) style — heading quiver at (Δx, Δy).

    Each codebook token is a single-step displacement (Δx, Δy, Δθ).
    The visualisation places a short arrow at (Δx, Δy) oriented along Δθ.
      - grey arrows  = all K tokens
      - blue arrows  = randomly highlighted subset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = ret_traj.shape[0]

    if hasattr(ret_traj, 'numpy'):
        ret_traj = ret_traj.numpy()
    final_x = ret_traj[:, 0]
    final_y = ret_traj[:, 1]
    final_h = ret_traj[:, 2]

    # Heading direction vectors, scaled for visual clarity
    seg_len = 0.5
    dx = np.cos(final_h) * seg_len
    dy = np.sin(final_h) * seg_len

    highlight_idx = np.random.choice(
        n_clusters, size=min(n_highlight, n_clusters), replace=False
    )
    highlight_mask = np.zeros(n_clusters, dtype=bool)
    highlight_mask[highlight_idx] = True

    # ---- quiver-style plot (paper Fig S1(a)) ----
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.patch.set_facecolor("white")

    bg = ~highlight_mask
    ax.quiver(
        final_x[bg], final_y[bg], dx[bg], dy[bg],
        angles="xy", scale_units="xy", scale=1,
        color="grey", alpha=0.35, width=0.0015, headwidth=2.5, headlength=3,
    )
    ax.quiver(
        final_x[highlight_mask], final_y[highlight_mask],
        dx[highlight_mask], dy[highlight_mask],
        angles="xy", scale_units="xy", scale=1,
        color="#2563EB", alpha=0.85, width=0.002, headwidth=2.5, headlength=3,
    )

    ax.set_xlabel("Δx (m)", fontsize=12)
    ax.set_ylabel("Δy (m)", fontsize=12)
    ax.set_title(
        f"NavSim Action Codebook (K={n_clusters}, 1 step = 0.5 s)",
        fontsize=14, fontweight="bold",
    )
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.grid(True, alpha=0.15)

    legend_elements = [
        Line2D([0], [0], color="grey",    lw=1.5, alpha=0.5,
               label=f"All {n_clusters} tokens"),
        Line2D([0], [0], color="#2563EB", lw=2,   alpha=0.85,
               label=f"{min(n_highlight, n_clusters)} highlighted"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

    fig_path = output_dir / "navsim_codebook_trajectories.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # ---- scatter ----
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(final_x, final_y, s=5, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"Action Codebook Final Positions ({n_clusters} tokens)")
    ax.set_xlabel("Δx (m)")
    ax.set_ylabel("Δy (m)")
    ax.grid(True, alpha=0.3)
    scatter_path = output_dir / "navsim_codebook_final_positions.jpg"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {scatter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize codebook from pkl")
    parser.add_argument(
        "--pkl", type=str,
        default="codebook_cache/agent_vocab_navsim_kdisk.pkl",
        help="Path to codebook pkl file",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load pkl
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    contour = data["token_all"]["veh"]  # [K, 1, 4, 2]
    print(f"Loaded contour shape: {contour.shape}")

    # Reconstruct (Δx, Δy, Δθ) from 4-corner bounding box
    centers = contour.mean(axis=-2)  # [K, 1, 2]  center of bbox
    front = (contour[:, :, 0, :] + contour[:, :, 1, :]) / 2  # front center
    back  = (contour[:, :, 2, :] + contour[:, :, 3, :]) / 2  # back center
    diff = front - back
    heading = np.arctan2(diff[..., 1], diff[..., 0])  # [K, 1]

    # [K, 3]  (squeeze the step dim since it's 1)
    ret_traj = np.concatenate([centers[:, 0, :], heading[:, 0:1]], axis=-1)
    print(f"Tokens: {ret_traj.shape[0]}")
    print(f"  Δx range: [{ret_traj[:, 0].min():.3f}, {ret_traj[:, 0].max():.3f}]")
    print(f"  Δy range: [{ret_traj[:, 1].min():.3f}, {ret_traj[:, 1].max():.3f}]")

    out_dir = Path(args.pkl).parent
    visualize_codebook(ret_traj, out_dir)
    print("Done!")