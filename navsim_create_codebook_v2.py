"""
Create action codebook from NavSim data.

NavSim log format
-----------------
Each .pkl file under  ``navsim_logs/trainval/``  is a list of frame dicts
recorded at **2 Hz** (0.5 s per frame).  Each dict contains:

    ego2global_translation : [x, y, z]           (float64, metres)
    ego2global_rotation    : [w, x, y, z]         (quaternion)

Codebook construction
---------------------
Each codebook token is a **single-step displacement** (Δx, Δy, Δθ) between
two consecutive frames (0.5 s at 2 Hz).

Clustering uses **greedy set-cover K-Disk** (SMART-style):
  - Build KDTree on candidate pool
  - Each iteration picks the point covering the most uncovered neighbours
  - Covered points are marked and excluded from future counting
  - Achieves (1 - 1/e) approximation of optimal coverage

This replaces the AutoVLA-style random-selection K-Disk with a
deterministic, higher-quality greedy algorithm.
"""

import sys
import os
import math
import random
import pickle
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def wrap_angle(angle):
    """Wrap angle (scalar or array) to [-π, π]."""
    if isinstance(angle, torch.Tensor):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quaternion_to_yaw(quat):
    """Convert quaternion [w, x, y, z] → yaw (radians), ZYX convention."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_to_local(pos_global, head_global, pos_now, head_now):
    """
    Transform global position / heading into the local frame defined by
    (pos_now, head_now).

    Args:
        pos_global:  [B, N, 2]  global positions
        head_global: [B, N]     global headings
        pos_now:     [B, 2]     current position (frame origin)
        head_now:    [B]        current heading  (frame orientation)

    Returns:
        local_pos:   [B, N, 2]
        local_head:  [B, N]
    """
    cos_h = head_now.cos()
    sin_h = head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), dtype=head_now.dtype)
    rot_mat[:, 0, 0] = cos_h
    rot_mat[:, 0, 1] = -sin_h
    rot_mat[:, 1, 0] = sin_h
    rot_mat[:, 1, 1] = cos_h

    diff = pos_global - pos_now.unsqueeze(1)
    local_pos = torch.bmm(diff, rot_mat)

    local_head = head_global - head_now.unsqueeze(-1)
    return local_pos, local_head


def seed_everything(seed):
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Greedy set-cover K-Disk clustering (SMART-style)
# ---------------------------------------------------------------------------

def fit_kdisk(
    vectors: np.ndarray,
    n_clusters: int,
    radius: Union[float, str] = 0.03,
    seed: int = 0,
    max_candidates: int = 10000,
) -> np.ndarray:
    """Greedy set-cover K-Disk clustering.

    At each step selects the point that covers the most currently-uncovered
    points within ``radius``, then marks those points as covered.  This is
    the canonical greedy approximation to the minimum disk cover problem
    (achieves (1 - 1/e) of optimal coverage).

    Parameters
    ----------
    vectors       : float32 [N, D]
    n_clusters    : int
    radius        : float or None.  Auto-set to mean nearest-neighbour
                    distance on a 2000-point probe when None.
    seed          : int   RNG seed for subsampling
    max_candidates: int   Candidate pool size (for scalability)

    Returns
    -------
    centers : float32 [K, D]  (K ≤ n_clusters)
    """
    from scipy.spatial import KDTree

    rng = np.random.default_rng(seed)
    n, d = vectors.shape
    n_clusters = min(n_clusters, n)

    # Subsample candidate pool for scalability
    if n > max_candidates:
        cand_idx = rng.choice(n, max_candidates, replace=False)
        pool = vectors[cand_idx].astype(np.float32)
    else:
        pool = vectors.astype(np.float32)
    M = pool.shape[0]

    # Auto-estimate radius from mean nearest-neighbour distance
    if radius == "auto":
        probe_n = min(2000, M)
        probe = pool[:probe_n]
        sq = ((probe[:, None, :] - probe[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(sq, np.inf)
        radius = float(np.sqrt(sq.min(axis=1).mean()))
        log.info("k-disk auto radius=%.4f", radius)
    else:
        radius = float(radius)

    # Precompute neighbourhoods with KDTree — O(M log M)
    tree = KDTree(pool)
    neighbors = tree.query_ball_tree(tree, r=float(radius))

    covered = np.zeros(M, dtype=bool)
    counts = np.array([len(nb) for nb in neighbors], dtype=np.int32)
    centers: List[np.ndarray] = []

    for i in range(n_clusters):
        if covered.all():
            break
        # Pick the point covering the most uncovered neighbours
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        centers.append(pool[best].copy())

        # Mark neighbours as covered and update counts
        for j in neighbors[best]:
            if not covered[j]:
                covered[j] = True
                for k in neighbors[j]:
                    counts[k] -= 1

        if (i + 1) % 100 == 0 or i == 0:
            remain = (~covered).sum() * 100.0 / M
            log.info("  cluster %d/%d, uncovered=%.1f%%", i + 1, n_clusters, remain)

    if not centers:
        return np.zeros((0, d), dtype=np.float32)
    result = np.stack(centers, axis=0).astype(np.float32)
    log.info("k-disk done  centers=%s  radius=%.4f", result.shape, radius)
    return result


# ---------------------------------------------------------------------------
# NavSim data loading (single-step displacements)
# ---------------------------------------------------------------------------

def load_navsim_single_step_displacements(
    data_path: str,
    n_trajs: int,
    min_displacement: float = 0.0,
) -> np.ndarray:
    """
    Load single-step displacements (Δx, Δy, Δθ) from NavSim pkl log files.

    For each consecutive pair of frames (t → t+1), compute the displacement
    in the local ego frame of frame t.

    Returns
    -------
    displacements : np.ndarray [N, 3]  (Δx, Δy, Δθ)
    """
    pkl_files = sorted(glob.glob(os.path.join(data_path, "*.pkl")))
    assert len(pkl_files) > 0, f"No pkl files found in {data_path}"
    log.info("Found %d log files in %s", len(pkl_files), data_path)

    disp_list = []
    count = 0

    target_str = n_trajs if n_trajs is not None else "all"
    with tqdm(total=len(pkl_files),
              desc=f"Loading displacements (target={target_str})") as pbar:
        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                frames = pickle.load(f)

            if not isinstance(frames, list) or len(frames) < 2:
                pbar.update(1)
                continue

            # Extract global ego poses: (x, y, yaw)
            ego_poses = []
            for frame in frames:
                trans = frame["ego2global_translation"]
                rot = frame["ego2global_rotation"]
                yaw = quaternion_to_yaw(rot)
                ego_poses.append((trans[0], trans[1], yaw))

            for t in range(len(ego_poses) - 1):
                if n_trajs is not None and count >= n_trajs:
                    break

                pos = torch.tensor([ego_poses[t][:2]], dtype=torch.float32)
                head = torch.tensor([ego_poses[t][2]], dtype=torch.float32)
                next_pos = torch.tensor([ego_poses[t + 1][:2]], dtype=torch.float32)
                next_head = torch.tensor([ego_poses[t + 1][2]], dtype=torch.float32)

                l_pos, l_head = transform_to_local(
                    pos_global=next_pos.unsqueeze(0),
                    head_global=next_head.unsqueeze(0),
                    pos_now=pos,
                    head_now=head,
                )
                l_head = wrap_angle(l_head)

                dx = l_pos[0, 0, 0].item()
                dy = l_pos[0, 0, 1].item()

                if min_displacement > 0.0:
                    if math.sqrt(dx * dx + dy * dy) < min_displacement:
                        continue

                disp_list.append([dx, dy])
                count += 1

            pbar.update(1)
            if n_trajs is not None and count >= n_trajs:
                break

    log.info("Loaded %d single-step displacements", count)
    return np.array(disp_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Quantization error
# ---------------------------------------------------------------------------

def compute_quantization_error(
    vectors: np.ndarray,
    centers: np.ndarray,
    chunk: int = 8192,
) -> dict:
    """Compute mean L2 reconstruction error."""
    n = vectors.shape[0]
    indices = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = vectors[start:end, None, :] - centers[None, :, :]
        indices[start:end] = (diff ** 2).sum(axis=2).argmin(axis=1).astype(np.int32)

    recon = centers[indices]
    errors = np.linalg.norm(vectors - recon, axis=1)
    originals = np.linalg.norm(vectors, axis=1)
    return {
        "n": int(n),
        "mean_l2_error": float(errors.mean()),
        "max_l2_error": float(errors.max()),
        "mean_l2_original": float(originals.mean()),
        "error_ratio_pct": float(100.0 * errors.mean() / (originals.mean() + 1e-9)),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_codebook(centers: np.ndarray, output_dir: Path):
    """
    2-D fan plot (origin → each center endpoint) matching the nuScenes
    build_motion_codebook.ipynb style.

    For NavSim 3-D tokens (Δx, Δy, Δθ):
      - Left subplot  : position fan plot (Δx, Δy) with steelblue lines + tomato dots
      - Right subplot : heading distribution (Δθ histogram)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = centers.shape[0]
    pos = centers  # [K, 2]  (Δx, Δy)

    # Fan plot: line from origin (0,0) to (Δx, Δy), tomato dot at endpoint
    # col0 = Δx (forward), col1 = Δy (lateral)
    pts = np.concatenate([
        np.zeros((n_clusters, 1, 2), dtype=np.float32),
        pos.reshape(-1, 1, 2),
    ], axis=1)  # [K, 2, 2]  (origin + endpoint)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle(
        f"NavSim Action Codebook — Greedy K-Disk (V={n_clusters}, 1 step = 0.5 s)",
        fontsize=13,
    )

    for traj in pts:
        ax.plot(traj[:, 0], traj[:, 1], color="steelblue", alpha=0.5, linewidth=0.8)
        ax.scatter(traj[-1, 0], traj[-1, 1], s=6, color="tomato", zorder=3)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"ego  (V={n_clusters})")
    ax.set_xlabel("forward / Δx (m)")
    ax.set_ylabel("lateral / Δy (m)")
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3)

    ax.grid(True, linewidth=0.3)

    fig_path = output_dir / "navsim_codebook_fan.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", fig_path)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_codebook(centers: np.ndarray, meta: dict, output_dir: Path):
    """Save codebook as .npy + meta.pkl."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "ego.npy", centers)

    with (output_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f, protocol=4)

    log.info("Saved codebook to %s", output_dir)
    log.info("  ego.npy  shape=%s", centers.shape)


def load_codebook(directory: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    """Load codebook from directory."""
    d = Path(directory)
    with (d / "meta.pkl").open("rb") as f:
        meta = pickle.load(f)
    centers = np.load(d / "ego.npy").astype(np.float32)
    return centers, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create action codebook from NavSim data (greedy K-Disk)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_path", type=str,
        default="/data/navsim/dataset/navsim_logs/trainval",
        help="Root directory of NavSim log pkl files",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=None,
        help="Output directory (default: codebook_cache/ next to this script)",
    )
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument(
        "--n_trajs", type=int, default=None,
        help="Max number of single-step displacements to sample (default: use all)",
    )
    parser.add_argument(
        "--min_displacement", type=float, default=0.0,
        help="Minimum displacement (m) to filter nearly-stationary steps",
    )
    parser.add_argument(
        "--radius", type=str, default="0.03",
        help="K-Disk radius. Use 'auto' for automatic calculation from mean NN distance (default: 0.03)",
    )
    parser.add_argument(
        "--max_candidates", type=int, default=10000,
        help="K-Disk candidate pool size for scalability",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    seed_everything(args.seed)

    # Output directory
    if args.output_dir is None:
        out_dir = Path(__file__).resolve().parent / "codebook_cache"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook_dir = out_dir / f"navsim_kdisk_v{args.vocab_size}"

    log.info("Output: %s", codebook_dir)
    log.info("NavSim frame interval: 0.5 s (2 Hz)")
    log.info("Token type: single-step displacement (Δx, Δy)")
    log.info("Clustering: greedy set-cover K-Disk")

    # ---- Load data ----
    displacements = load_navsim_single_step_displacements(
        args.data_path, args.n_trajs,
        min_displacement=args.min_displacement,
    )

    # Prepend zero-displacement (stop token)
    zero_step = np.zeros((1, 2), dtype=np.float32)
    displacements = np.concatenate([zero_step, displacements], axis=0)
    log.info("Total displacement vectors: %s (including stop token)", displacements.shape)

    # ---- Greedy K-Disk clustering ----
    radius_val = "auto" if args.radius.lower() == "auto" else float(args.radius)
    log.info("Fitting greedy K-Disk (vocab=%d, radius=%s) ...", args.vocab_size, args.radius)
    centers = fit_kdisk(
        displacements,
        n_clusters=args.vocab_size,
        radius=radius_val,
        seed=args.seed,
        max_candidates=args.max_candidates,
    )

    log.info("Final codebook: %d tokens, dim=%d", centers.shape[0], centers.shape[1])

    # ---- Quantization error ----
    log.info("Evaluating quantization error ...")
    stats = compute_quantization_error(displacements, centers)
    log.info("  n=%d  mean_l2=%.4f  max_l2=%.4f  original=%.4f  ratio=%.2f%%",
             stats["n"], stats["mean_l2_error"], stats["max_l2_error"],
             stats["mean_l2_original"], stats["error_ratio_pct"])

    # ---- Visualisation ----
    visualize_codebook(centers, codebook_dir)

    # ---- Save ----
    meta = {
        "vocab_size": args.vocab_size,
        "actual_vocab_size": centers.shape[0],
        "token_dim": 2,
        "token_format": "(dx, dy)",
        "frame_interval_s": 0.5,
        "clustering": "greedy_kdisk",
        "radius": radius_val,
        "max_candidates": args.max_candidates,
        "seed": args.seed,
        "n_displacements": len(displacements),
        "quantization_error": stats,
    }
    save_codebook(centers, meta, codebook_dir)

    log.info("Done.")
    log.info("  vocab size : %d", centers.shape[0])
    log.info("  token dim  : %d  (Δx, Δy)", centers.shape[1])
    log.info("  saved to   : %s", codebook_dir)
