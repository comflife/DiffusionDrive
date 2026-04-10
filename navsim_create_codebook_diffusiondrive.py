"""
Create action codebook for DiffusionDrive Discrete AR model.

Based on NavSim log format and AutoVLA-style tokenization.
Each token represents a single-step displacement (Δx, Δy) between frames.
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
        a_pos: [n_trajs, T, 2] full trajectory segments
    
    Returns:
        [N, T, 2] representative trajectories
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


def load_navsim_displacements(data_path, n_trajs):
    """Load single-step displacements from NavSim pkl files."""
    pkl_files = sorted(glob.glob(os.path.join(data_path, "*.pkl")))
    assert len(pkl_files) > 0, f"No pkl files found in {data_path}"
    print(f"Found {len(pkl_files)} log files in {data_path}")

    traj_list = []
    count = 0

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

            # Compute displacements
            for t in range(len(ego_poses) - 1):
                if count >= n_trajs:
                    break

                pos = torch.tensor([ego_poses[t][:2]], dtype=torch.float32)
                head = torch.tensor([ego_poses[t][2]], dtype=torch.float32)
                next_pos = torch.tensor([ego_poses[t+1][:2]], dtype=torch.float32)
                next_head = torch.tensor([ego_poses[t+1][2]], dtype=torch.float32)

                l_pos, l_head = transform_to_local(
                    pos_global=next_pos.unsqueeze(0),
                    head_global=next_head.unsqueeze(0),
                    pos_now=pos,
                    head_now=head,
                )
                l_head = wrap_angle(l_head)
                
                step = torch.cat([l_pos.squeeze(0), l_head.unsqueeze(-1).squeeze(0)], dim=-1)
                traj_list.append(step.unsqueeze(0))
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
    final_x = ret_traj[:, -1, 0].numpy()
    final_y = ret_traj[:, -1, 1].numpy()
    final_h = ret_traj[:, -1, 2].numpy()

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

    vis_path = output_dir / 'codebook_visualization.png'
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {vis_path}")


def main():
    parser = argparse.ArgumentParser(description="Create action codebook for DiffusionDrive AR")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to NavSim log pkl files")
    parser.add_argument("--output", type=str, default="codebook_cache/diffusiondrive_ego_vocab",
                        help="Output path prefix for codebook files")
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

    # Load displacements
    traj_list, count = load_navsim_displacements(args.data_path, args.n_trajs)
    print(f"Loaded {count} trajectory segments")

    if count == 0:
        print("No trajectories loaded!")
        return

    # Stack trajectories
    trajs = torch.cat(traj_list, dim=0).unsqueeze(1)  # [N, 1, 3]
    print(f"Trajectory tensor shape: {trajs.shape}")

    # K-disk clustering
    width_length = torch.tensor([2.0, 4.8]).unsqueeze(0)  # Vehicle dimensions
    
    contour = cal_polygon_contour(
        pos=trajs[:, -1, :2],
        head=trajs[:, -1, 2],
        width_length=width_length
    )  # [N, 4, 2]

    print("Running K-disk clustering...")
    ret_traj = Kdisk_cluster(
        X=contour,
        N=args.vocab_size,
        tol=args.tol_dist,
        a_pos=trajs
    )
    ret_traj[:, :, -1] = wrap_angle(ret_traj[:, :, -1])

    print(f"Created codebook with shape: {ret_traj.shape}")

    # Visualize
    output_dir = Path(args.output).parent
    visualize_codebook(ret_traj, output_dir)

    # Save codebook
    output_path = Path(args.output).with_suffix('.npy')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, ret_traj.numpy())
    print(f"Saved codebook to {output_path}")

    # Also save as .pkl for compatibility
    import pickle
    pkl_path = output_path.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({"token_all": {"veh": ret_traj.numpy()}}, f)
    print(f"Saved codebook (pkl) to {pkl_path}")


if __name__ == "__main__":
    main()
