"""
Create action codebook for DiffusionDrive AR model — AutoVLA methodology.

Key difference from navsim_create_codebook_diffusiondrive.py:
  - Clusters on corner contours (distance metric), same as before
  - Averages *displacement vectors* (dx, dy, dtheta) per cluster, NOT corners
  - Recomputes corner contours from the mean displacement after clustering
  - This preserves valid vehicle bounding-box semantics in every codebook entry

The rotation convention matches the DiffusionDrive model
(match_to_step_corner_codebook, _decode_step_corner_tokens):
    local_forward  = dx·cos(θ) + dy·sin(θ)
    local_lateral   = -dx·sin(θ) + dy·cos(θ)

Output: ego.npy with shape [V, 4, 2] (corner contours), compatible with the
        model's _init_codebook step_corners handler.

Usage:
    python navsim_create_codebook_diffusiondrive_autovla.py \
        --data_path /path/to/navsim_logs \
        --output codebook_cache/navsim_kdisk_v2048_diffusiondrive_autovla \
        --vocab_size 2048 \
        --method greedy
"""

import sys
import os
import math
import random
import pickle
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    if isinstance(angle, torch.Tensor):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quaternion_to_yaw(quat):
    """Convert quaternion [w, x, y, z] -> yaw (radians)."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_to_local(pos_global, head_global, pos_now, head_now):
    """Transform global position/heading to local ego frame.

    Uses the standard rotation (matches the DiffusionDrive model):
        local_forward  = dx·cos(h) + dy·sin(h)
        local_lateral  = -dx·sin(h) + dy·cos(h)
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


def cal_polygon_contour(pos, head, width_length):
    """Compute 4-corner bounding box polygons.

    Corner order: left_front, right_front, right_back, left_back
    (matches DiffusionDrive model's _init_codebook extraction).
    """
    w = width_length[..., 0] / 2.0
    l = width_length[..., 1] / 2.0

    cos_h = torch.cos(head).unsqueeze(-1)
    sin_h = torch.sin(head).unsqueeze(-1)

    dx = torch.stack([l, l, -l, -l], dim=-1)
    dy = torch.stack([w, -w, -w, w], dim=-1)

    cx = dx * cos_h - dy * sin_h
    cy = dx * sin_h + dy * cos_h

    corners = torch.stack([cx, cy], dim=-1)
    corners = corners + pos.unsqueeze(-2)
    return corners


# ---------------------------------------------------------------------------
# Random K-disk clustering (AutoVLA original)
# ---------------------------------------------------------------------------

def Kdisk_cluster_random(X, N, tol, a_pos):
    """Random-selection K-disk clustering (AutoVLA methodology).

    Clusters on X (corner contours) for distance, but averages a_pos
    (displacement vectors) for the cluster representative.

    Args:
        X:     [n_trajs, 4, 2] corner contours (distance metric)
        N:     number of clusters
        tol:   tolerance distance
        a_pos: [n_trajs, 3] displacement vectors (dx, dy, dtheta) for averaging

    Returns:
        [N, 3] representative displacement vectors
    """
    n_total = X.shape[0]
    ret_list = []

    for i in range(N):
        if X.shape[0] == 0:
            print(f"Warning: ran out of data at cluster {i}/{N}")
            break

        if i == 0:
            # First cluster: pick the entry closest to zero displacement
            choice_index = int(torch.argmin(a_pos[:, :2].norm(dim=-1)))
        else:
            choice_index = torch.randint(0, X.shape[0], (1,)).item()

        x0 = X[choice_index]
        res_mask = torch.norm(X - x0, dim=-1).mean(-1) > tol

        # Average displacements (not corners) — core AutoVLA methodology
        ret_traj = a_pos[~res_mask].mean(0, keepdim=True)

        X = X[res_mask]
        a_pos = a_pos[res_mask]
        ret_list.append(ret_traj)

        remain = X.shape[0] * 100.0 / n_total
        n_inside = (~res_mask).sum().item()
        print(f"Cluster {i}: {remain:.2f}% remaining, {n_inside} samples in cluster")

    return torch.cat(ret_list, dim=0)


# ---------------------------------------------------------------------------
# Greedy set-cover K-disk clustering
# ---------------------------------------------------------------------------

def Kdisk_cluster_greedy(X, N, tol, a_pos, max_candidates=10000, seed=0):
    """Greedy set-cover K-disk clustering (deterministic, better coverage).

    Same distance metric and averaging strategy as random K-disk,
    but greedily selects the point covering the most uncovered neighbours.

    Args:
        X:            [n_trajs, 4, 2] corner contours (distance metric)
        N:            number of clusters
        tol:          tolerance distance
        a_pos:        [n_trajs, 3] displacement vectors for averaging
        max_candidates: max candidate pool for KDTree
        seed:         random seed for candidate subsampling

    Returns:
        [K, 3] representative displacement vectors (K <= N)
    """
    from scipy.spatial import KDTree

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    N = min(N, n)

    # Flatten corners for KDTree
    X_flat = X.reshape(n, -1).numpy().astype(np.float32)

    # Subsample candidate pool for scalability
    if n > max_candidates:
        cand_idx = rng.choice(n, max_candidates, replace=False)
        pool = X_flat[cand_idx]
        pool_a_idx = cand_idx
    else:
        pool = X_flat
        pool_a_idx = np.arange(n)

    M = pool.shape[0]

    # Build KDTree and precompute neighborhoods
    tree = KDTree(pool)
    neighbors = tree.query_ball_tree(tree, r=float(tol))

    covered = np.zeros(M, dtype=bool)
    counts = np.array([len(nb) for nb in neighbors], dtype=np.int32)
    selected_indices = []

    # First cluster: closest to zero displacement
    if M > 0:
        zero_dist = np.linalg.norm(pool, axis=1)
        best = int(np.argmin(zero_dist))
    else:
        return torch.zeros((0, 3), dtype=torch.float32)

    for i in range(N):
        if covered.all():
            print(f"All points covered at cluster {i}")
            break

        if i == 0:
            best = int(np.argmin(np.linalg.norm(pool, axis=1)))
        else:
            best = int(np.argmax(counts))
            if counts[best] == 0:
                break

        selected_indices.append(pool_a_idx[best])

        for j in neighbors[best]:
            if not covered[j]:
                covered[j] = True
                for k in neighbors[j]:
                    counts[k] -= 1

        if (i + 1) % 100 == 0:
            remain = (~covered).sum() * 100.0 / M
            print(f"  greedy cluster {i+1}/{N}, uncovered={remain:.1f}%")

    # Gather displacement vectors for selected indices
    # Re-cluster: for each selected center, find all original points within
    # tol distance and average their displacements
    X_selected_flat = X_flat[selected_indices]
    tree_selected = KDTree(X_selected_flat)
    assignments = tree_selected.query(X_flat, k=1)[1].flatten()

    result = []
    for i in range(len(selected_indices)):
        mask = assignments == i
        if mask.sum() == 0:
            result.append(a_pos[selected_indices[i]].unsqueeze(0))
        else:
            result.append(a_pos[mask].mean(0, keepdim=True))

    ret = torch.cat(result, dim=0)
    print(f"Greedy K-disk: selected {len(selected_indices)} clusters from {n} points")
    return ret


def load_navsim_trajectories(data_path, n_trajs):
    """Load single-step displacements (dx, dy, dtheta) from NavSim pkl files."""
    pkl_files = sorted(glob.glob(os.path.join(data_path, "*.pkl")))
    assert len(pkl_files) > 0, f"No pkl files found in {data_path}"
    print(f"Found {len(pkl_files)} log files in {data_path}")

    disp_list = []
    count = 0

    with tqdm(total=len(pkl_files), desc="Loading displacements") as pbar:
        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                frames = pickle.load(f)

            if not isinstance(frames, list) or len(frames) < 2:
                pbar.update(1)
                continue

            # Extract global ego poses
            ego_poses = []
            for frame in frames:
                trans = frame["ego2global_translation"]
                rot = frame["ego2global_rotation"]
                yaw = quaternion_to_yaw(rot)
                ego_poses.append((trans[0], trans[1], yaw))

            for t in range(len(ego_poses) - 1):
                if count >= n_trajs:
                    break

                pos_now = torch.tensor([ego_poses[t][:2]], dtype=torch.float32)
                head_now = torch.tensor([ego_poses[t][2]], dtype=torch.float32)
                next_pos = torch.tensor([ego_poses[t + 1][:2]], dtype=torch.float32)
                next_head = torch.tensor([ego_poses[t + 1][2]], dtype=torch.float32)

                l_pos, l_head = transform_to_local(
                    pos_global=next_pos.unsqueeze(0),
                    head_global=next_head.unsqueeze(0),
                    pos_now=pos_now,
                    head_now=head_now,
                )
                l_head = wrap_angle(l_head)

                # Store (dx, dy, dtheta) as [3] displacement vector
                dx_dy = l_pos.squeeze()    # [2]
                dtheta = l_head.squeeze()   # scalar
                step = torch.cat([dx_dy, dtheta.unsqueeze(0)], dim=0)  # [3]
                disp_list.append(step)
                count += 1

            pbar.update(1)
            if count >= n_trajs:
                break

    return disp_list, count


def visualize_codebook(disps, corners, output_dir, n_highlight=300):
    """Visualize codebook tokens — displacement fan plot + heading histogram."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = disps.shape[0]
    dx = disps[:, 0].numpy()
    dy = disps[:, 1].numpy()
    dtheta = disps[:, 2].numpy()

    highlight_idx = np.random.choice(n_clusters, min(n_highlight, n_clusters), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: fan plot (origin -> endpoint)
    ax = axes[0]
    for i in range(n_clusters):
        ax.plot([0, dx[i]], [0, dy[i]], color='grey', alpha=0.3, linewidth=0.5)
    for idx in highlight_idx:
        ax.plot([0, dx[idx]], [0, dy[idx]], 'b', alpha=0.8, linewidth=1)
        ax.scatter([dx[idx]], [dy[idx]], s=6, c='tomato', zorder=3)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_aspect('equal')
    ax.set_xlabel('dx (m)')
    ax.set_ylabel('dy (m)')
    ax.set_title(f'Displacement Fan (V={n_clusters})')
    ax.grid(True, alpha=0.3)

    # Right: heading histogram
    ax = axes[1]
    ax.hist(dtheta, bins=60, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('dtheta (rad)')
    ax.set_ylabel('Count')
    ax.set_title('Heading Change Distribution')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'AutoVLA-Methodology Codebook (V={n_clusters}, step_corners)', fontsize=13)
    fig.tight_layout()

    vis_path = output_dir / 'navsim_codebook_fan_autovla.png'
    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {vis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create action codebook using AutoVLA methodology "
                    "(cluster on corners, average displacements, recompute corners)"
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to NavSim log pkl files")
    parser.add_argument("--output", type=str,
                        default="codebook_cache/navsim_kdisk_v2048_diffusiondrive_autovla",
                        help="Output directory for codebook files")
    parser.add_argument("--vocab_size", type=int, default=2048,
                        help="Vocabulary size / number of clusters")
    parser.add_argument("--n_trajs", type=int, default=100000,
                        help="Number of trajectory segments to sample")
    parser.add_argument("--tol_dist", type=float, default=0.05,
                        help="Tolerance distance for K-disk clustering")
    parser.add_argument("--method", type=str, choices=["random", "greedy"], default="greedy",
                        help="K-disk clustering method: 'random' (AutoVLA original) "
                             "or 'greedy' (deterministic, better coverage)")
    parser.add_argument("--max_candidates", type=int, default=10000,
                        help="Max candidate pool for greedy K-disk")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Creating codebook with vocab_size={args.vocab_size}")
    print(f"Method: {args.method} K-disk")
    print(f"Methodology: cluster on corners, average displacements, recompute corners")
    print(f"Loading data from: {args.data_path}")

    # Load displacements
    disp_list, count = load_navsim_trajectories(args.data_path, args.n_trajs)
    print(f"Loaded {count} trajectory segments")

    if count == 0:
        print("No trajectories loaded!")
        return

    # Stack displacements: [N, 3] (dx, dy, dtheta)
    disps = torch.stack(disp_list, dim=0)
    print(f"Displacement tensor shape: {disps.shape}")

    # Prepend zero-displacement token (AutoVLA convention)
    zero_step = torch.zeros(1, 3, dtype=torch.float32)
    disps = torch.cat([zero_step, disps], dim=0)
    print(f"After prepending zero token: {disps.shape}")

    # Compute corner contours for clustering
    width_length = torch.tensor([2.0, 4.8])
    width_length_expanded = width_length.unsqueeze(0).expand(disps.shape[0], -1)

    contour = cal_polygon_contour(
        pos=disps[:, :2],       # [N, 2]
        head=disps[:, 2],        # [N]
        width_length=width_length_expanded,  # [N, 2]
    )  # [N, 4, 2]

    # K-disk clustering
    if args.method == "random":
        print(f"Running random K-disk clustering (tol={args.tol_dist})...")
        ret_disps = Kdisk_cluster_random(
            X=contour,
            N=args.vocab_size,
            tol=args.tol_dist,
            a_pos=disps,
        )
    else:
        print(f"Running greedy K-disk clustering (tol={args.tol_dist})...")
        ret_disps = Kdisk_cluster_greedy(
            X=contour,
            N=args.vocab_size,
            tol=args.tol_dist,
            a_pos=disps,
            max_candidates=args.max_candidates,
            seed=args.seed,
        )

    # Wrap headings to [-pi, pi]
    ret_disps[:, 2] = wrap_angle(ret_disps[:, 2])

    print(f"Clustered displacements shape: {ret_disps.shape}")

    # Recompute corner contours from mean displacements (AutoVLA step)
    width_length_ret = width_length.unsqueeze(0).expand(ret_disps.shape[0], -1)
    ret_corners = cal_polygon_contour(
        pos=ret_disps[:, :2],    # [V, 2]
        head=ret_disps[:, 2],     # [V]
        width_length=width_length_ret,  # [V, 2]
    )  # [V, 4, 2]

    print(f"Recomputed corners shape: {ret_corners.shape}")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize
    visualize_codebook(ret_disps, ret_corners, output_dir)

    # Save codebook as ego.npy — [V, 4, 2] corner format
    # Compatible with model's _init_codebook step_corners handler
    ego_npy_path = output_dir / 'ego.npy'
    np.save(ego_npy_path, ret_corners.numpy())
    print(f"Saved codebook to {ego_npy_path}")
    print(f"  Shape: {ret_corners.shape}")
    print(f"  Format: [V, 4, 2] corner contours (recomputed from mean displacements)")

    # Also save displacement vectors for reference
    disp_npy_path = output_dir / 'ego_displacements.npy'
    np.save(disp_npy_path, ret_disps.numpy())
    print(f"Saved displacements to {disp_npy_path}")
    print(f"  Shape: {ret_disps.shape}")
    print(f"  Format: [V, 3] (dx, dy, dtheta)")

    # Save metadata
    meta = {
        'vocab_size': args.vocab_size,
        'actual_vocab_size': ret_disps.shape[0],
        'token_dim': 3,
        'token_format': '(dx, dy, dtheta) single-step displacement',
        'corner_format': '(4, 2) bounding box corners, recomputed from mean displacement',
        'frame_interval_s': 0.5,
        'clustering': f'{args.method}_kdisk',
        'methodology': 'cluster_on_corners_average_displacements_recompute_corners',
        'radius': args.tol_dist,
        'n_steps': 1,
        'seed': args.seed,
        'n_trajectories': count,
        'rotation_convention': 'standard_global_to_local',
        'vehicle_dims': [2.0, 4.8],
    }
    meta_path = output_dir / 'meta.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()