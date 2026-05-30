"""
Compare two DiffusionDrive step-corner codebooks in one figure.

Usage:
    python visualize/compare_diffusiondrive_codebooks.py
"""

import argparse
import os
from pathlib import Path

import numpy as np

Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def load_step_corner_codebook(path):
    corners = np.load(path).astype(np.float32)
    if corners.ndim != 3 or corners.shape[1:] != (4, 2):
        raise ValueError(f"Expected [V, 4, 2] corners, got {corners.shape}: {path}")

    center = corners.mean(axis=1)
    front = (corners[:, 0] + corners[:, 1]) * 0.5
    back = (corners[:, 2] + corners[:, 3]) * 0.5
    heading = np.arctan2((front - back)[:, 1], (front - back)[:, 0])
    return {
        "corners": corners,
        "center": center,
        "heading": heading,
    }


def add_codebook_overlay(ax, data, color, label, heading_len=0.45):
    center = data["center"]
    heading = data["heading"]

    fan_segments = np.stack(
        [np.zeros_like(center), center],
        axis=1,
    )
    ax.add_collection(
        LineCollection(fan_segments, colors=color, linewidths=0.7, alpha=0.18)
    )

    heading_vec = np.stack([np.cos(heading), np.sin(heading)], axis=1) * heading_len
    heading_segments = np.stack([center, center + heading_vec], axis=1)
    ax.add_collection(
        LineCollection(heading_segments, colors=color, linewidths=0.8, alpha=0.55)
    )
    ax.scatter(
        center[:, 0],
        center[:, 1],
        s=9,
        c=color,
        alpha=0.42,
        edgecolors="none",
        label=label,
    )


def common_bins(values_a, values_b, bins=55, pad_ratio=0.04):
    lo = min(float(values_a.min()), float(values_b.min()))
    hi = max(float(values_a.max()), float(values_b.max()))
    pad = max((hi - lo) * pad_ratio, 1e-3)
    return np.linspace(lo - pad, hi + pad, bins)


def plot_hist(ax, navsim_values, waymo_values, bins, xlabel, colors):
    ax.hist(
        waymo_values,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=colors["waymo"],
        alpha=0.24,
        linewidth=0,
    )
    ax.hist(
        navsim_values,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=colors["navsim"],
        alpha=0.24,
        linewidth=0,
    )
    ax.hist(
        waymo_values,
        bins=bins,
        density=True,
        histtype="step",
        color=colors["waymo"],
        linewidth=1.8,
    )
    ax.hist(
        navsim_values,
        bins=bins,
        density=True,
        histtype="step",
        color=colors["navsim"],
        linewidth=1.8,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.2, linewidth=0.6)


def summarize(name, data):
    center = data["center"]
    heading_deg = np.rad2deg(data["heading"])
    return (
        f"{name}: V={len(center)}, "
        f"x=[{center[:, 0].min():.2f}, {center[:, 0].max():.2f}], "
        f"y=[{center[:, 1].min():.2f}, {center[:, 1].max():.2f}], "
        f"heading std={heading_deg.std():.1f} deg"
    )


def make_plot(navsim_path, waymo_path, output_path):
    navsim = load_step_corner_codebook(navsim_path)
    waymo = load_step_corner_codebook(waymo_path)

    colors = {
        "navsim": "#2563eb",
        "waymo": "#f97316",
    }

    fig = plt.figure(figsize=(18, 7.8), facecolor="white")
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.7, 1.0],
        wspace=0.25,
        hspace=0.46,
    )

    ax_main = fig.add_subplot(gs[0, :])
    add_codebook_overlay(ax_main, waymo, colors["waymo"], "Waymo")
    add_codebook_overlay(ax_main, navsim, colors["navsim"], "NAVSIM")

    centers = np.concatenate([navsim["center"], waymo["center"]], axis=0)
    x_pad = max((centers[:, 0].max() - centers[:, 0].min()) * 0.05, 0.5)
    y_pad = max((centers[:, 1].max() - centers[:, 1].min()) * 0.35, 0.65)
    ax_main.set_xlim(centers[:, 0].min() - x_pad, centers[:, 0].max() + x_pad)
    ax_main.set_ylim(centers[:, 1].min() - y_pad, centers[:, 1].max() + y_pad)
    ax_main.axhline(0, color="#4b5563", linestyle="--", linewidth=0.8, alpha=0.55)
    ax_main.axvline(0, color="#4b5563", linestyle="--", linewidth=0.8, alpha=0.55)
    ax_main.set_aspect("auto")
    ax_main.set_xlabel("forward dx (m)")
    ax_main.set_ylabel("lateral dy (m)")
    ax_main.grid(True, alpha=0.22, linewidth=0.6)

    legend_handles = [
        Line2D([0], [0], color=colors["navsim"], lw=2.5, marker="o", markersize=5, label="NAVSIM"),
        Line2D([0], [0], color=colors["waymo"], lw=2.5, marker="o", markersize=5, label="Waymo"),
    ]
    ax_main.legend(handles=legend_handles, loc="upper left", frameon=True)

    ax_x = fig.add_subplot(gs[1, 0])
    plot_hist(
        ax_x,
        navsim["center"][:, 0],
        waymo["center"][:, 0],
        common_bins(navsim["center"][:, 0], waymo["center"][:, 0]),
        "forward dx (m)",
        colors,
    )
    ax_x.set_title("Endpoint x")

    ax_y = fig.add_subplot(gs[1, 1])
    plot_hist(
        ax_y,
        navsim["center"][:, 1],
        waymo["center"][:, 1],
        common_bins(navsim["center"][:, 1], waymo["center"][:, 1]),
        "lateral dy (m)",
        colors,
    )
    ax_y.set_title("Endpoint y")

    fig.suptitle(
        "DiffusionDrive codebook comparison: NAVSIM vs Waymo",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.012,
        0.018,
        summarize("NAVSIM", navsim) + "\n" + summarize("Waymo", waymo),
        fontsize=9,
        color="#374151",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compare NAVSIM and Waymo step-corner codebooks")
    parser.add_argument(
        "--navsim",
        type=Path,
        default=Path("codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy"),
    )
    parser.add_argument(
        "--waymo",
        type=Path,
        default=Path("codebook_cache/waymo_kdisk_v2048_diffusiondrive/ego.npy"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("codebook_cache/navsim_waymo_codebook_comparison.png"),
    )
    args = parser.parse_args()

    output_path = make_plot(args.navsim, args.waymo, args.output)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
