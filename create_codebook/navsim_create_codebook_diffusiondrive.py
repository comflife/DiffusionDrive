"""
Create action codebook for DiffusionDrive Discrete AR model.

Based on NavSim log format and AutoVLA-style tokenization.
Each token represents a single-step displacement with heading information encoded as 4-corner box contour.
"""

import sys
import os
import math
import random
import pickle
import glob
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def wrap_angle(angle):
    """Wrap angle to [-π, π]."""
    if isinstance(angle, torch.Tensor):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quaternion_to_yaw(quat):
    """Convert quaternion [w, x, y, z] → yaw (radians)."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_to_local(pos_global, head_global, pos_now, head_now):
    """Transform global position/heading to local frame."""
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


def Kdisk_cluster(X, N, tol, a_pos, cal_mean_heading=True):
    """
    K-disk clustering for trajectory quantization.
    
    Args:
        X: [n_trajs, 4, 2] bbox contour of trajectory endpoint
        N: number of clusters (vocab size)
        tol: tolerance distance
        a_pos: [n_trajs, T, 4, 2] full trajectory segments
    
    Returns:
        [N, T, 4, 2] representative trajectories
    """
    n_total = X.shape[0]
    ret_traj_list = []

    for i in range(N):
        if X.shape[0] == 0:
            print(f"Warning: ran out of data at cluster {i}/{N}")
            break

        if i == 0:
            choice_index = 0  # Always include zero displacement
        else:
            choice_index = torch.randint(0, X.shape[0], (1,)).item()

        x0 = X[choice_index]
        res_mask = torch.norm(X - x0, dim=-1).mean(-1) > tol
        
        if cal_mean_heading:
            ret_traj = a_pos[~res_mask].mean(0, keepdim=True)
        else:
            ret_traj = a_pos[[choice_index]]

        X = X[res_mask]
        a_pos = a_pos[res_mask]
        ret_traj_list.append(ret_traj)

        remain = X.shape[0] * 100.0 / n_total
        n_inside = (~res_mask).sum().item()
        print(f"Cluster {i}: {remain:.2f}% remaining, {n_inside} samples in cluster")

    return torch.cat(ret_traj_list, dim=0)


def cal_polygon_contour(pos, head, width_length):
    """Compute 4-corner bounding box polygons."""
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


def load_navsim_trajectories(data_path, n_trajs):
    """Load single-step displacements from NavSim pkl files."""
    pkl_files = sorted(glob.glob(os.path.join(data_path, "*.pkl")))
    assert len(pkl_files) > 0, f"No pkl files found in {data_path}"
    print(f"Found {len(pkl_files)} log files in {data_path}")

    traj_list = []
    count = 0
    width_length = torch.tensor([2.0, 4.8])

    with tqdm(total=len(pkl_files), desc=f"Loading displacements") as pbar:
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

            # Compute single-step displacements
            for t in range(len(ego_poses) - 1):
                if count >= n_trajs:
                    break

                pos_now = torch.tensor([ego_poses[t][:2]], dtype=torch.float32)
                head_now = torch.tensor([ego_poses[t][2]], dtype=torch.float32)
                next_pos = torch.tensor([ego_poses[t+1][:2]], dtype=torch.float32)
                next_head = torch.tensor([ego_poses[t+1][2]], dtype=torch.float32)

                l_pos, l_head = transform_to_local(
                    pos_global=next_pos.unsqueeze(0),
                    head_global=next_head.unsqueeze(0),
                    pos_now=pos_now,
                    head_now=head_now,
                )
                l_head = wrap_angle(l_head)

                # Compute corner contour in local frame for single step
                corners = cal_polygon_contour(
                    pos=l_pos.squeeze(0),      # [1, 2]
                    head=l_head.squeeze(0),    # [1]
                    width_length=width_length.unsqueeze(0)  # [1, 2]
                )  # [1, 4, 2]

                traj_list.append(corners)  # [1, 4, 2]
                count += 1

            pbar.update(1)
            if count >= n_trajs:
                break

    return traj_list, count


def visualize_codebook(ret_traj, output_dir, n_highlight=300):
    """Visualize codebook tokens."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = ret_traj.shape[0]
    # Compute center and heading from single-step corners
    final_pos = ret_traj.mean(dim=1).numpy()  # [n_clusters, 2]
    final_x = final_pos[:, 0]
    final_y = final_pos[:, 1]

    diff_xy = ret_traj[:, 0] - ret_traj[:, 3]  # [n_clusters, 2]
    final_h = np.arctan2(diff_xy[:, 1].numpy(), diff_xy[:, 0].numpy())

    seg_len = 0.3
    dx = np.cos(final_h) * seg_len
    dy = np.sin(final_h) * seg_len

    highlight_idx = np.random.choice(n_clusters, min(n_highlight, n_clusters), replace=False)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all tokens in grey
    for i in range(n_clusters):
        ax.plot([final_x[i], final_x[i] + dx[i]],
                [final_y[i], final_y[i] + dy[i]],
                'grey', alpha=0.3, linewidth=0.5)

    # Highlight subset in blue
    for idx in highlight_idx:
        ax.plot([final_x[idx], final_x[idx] + dx[idx]],
                [final_y[idx], final_y[idx] + dy[idx]],
                'b', alpha=0.8, linewidth=1)

    ax.set_aspect('equal')
    ax.set_xlabel('Δx (m)')
    ax.set_ylabel('Δy (m)')
    ax.set_title(f'Action Codebook ({n_clusters} tokens)')
    ax.grid(True, alpha=0.3)

    vis_path = output_dir / 'navsim_codebook_fan.png'
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {vis_path}")


def main():
    parser = argparse.ArgumentParser(description="Create action codebook for DiffusionDrive AR")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to NavSim log pkl files")
    parser.add_argument("--output", type=str, default="codebook_cache/navsim_kdisk_v512_diffusiondrive",
                        help="Output directory for codebook files")
    parser.add_argument("--vocab_size", type=int, default=256,
                        help="Vocabulary size / number of clusters")
    parser.add_argument("--n_trajs", type=int, default=100000,
                        help="Number of trajectory segments to sample")
    parser.add_argument("--tol_dist", type=float, default=0.05,
                        help="Tolerance distance for K-disk clustering")
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
    print(f"Loading data from: {args.data_path}")

    # Load single-step displacements
    traj_list, count = load_navsim_trajectories(args.data_path, args.n_trajs)
    print(f"Loaded {count} trajectory segments")

    if count == 0:
        print("No trajectories loaded!")
        return

    # Stack trajectories
    trajs = torch.cat(traj_list, dim=0)  # [N, 4, 2]
    print(f"Trajectory tensor shape: {trajs.shape}")

    # K-disk clustering
    contour = trajs  # [N, 4, 2]

    print("Running K-disk clustering...")
    ret_traj = Kdisk_cluster(
        X=contour,
        N=args.vocab_size,
        tol=args.tol_dist,
        a_pos=trajs,
        cal_mean_heading=True
    )

    print(f"Created codebook with shape: {ret_traj.shape}")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize
    visualize_codebook(ret_traj, output_dir)

    # Save codebook as ego.npy
    ego_npy_path = output_dir / 'ego.npy'
    np.save(ego_npy_path, ret_traj.numpy())
    print(f"Saved codebook to {ego_npy_path}")

    # Save metadata as meta.pkl
    n_steps = 1
    meta = {
        'vocab_size': args.vocab_size,
        'actual_vocab_size': ret_traj.shape[0],
        'token_dim': ret_traj.shape[-1],
        'token_format': '(4, 2) - (corners, xy) single-step with heading',
        'frame_interval_s': 0.5,
        'clustering': 'greedy_kdisk',
        'radius': args.tol_dist,
        'n_steps': n_steps,
        'seed': args.seed,
        'n_trajectories': count,
    }
    meta_path = output_dir / 'meta.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
