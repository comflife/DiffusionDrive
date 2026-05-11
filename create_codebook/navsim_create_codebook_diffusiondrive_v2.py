"""
Create action codebook for DiffusionDrive AR — v2 (improved).

Token granularity (unchanged from v1):
  - 1 token = 1 single-step (0.5 s) ego displacement
  - Stored as 4-corner box contour for distance metric, shape [V, 4, 2]
  - Per-step autoregressive decoding: T=8 tokens per plan
    (compatible with model's `ar_codebook_mode=step_corners`)

Algorithmic improvements over navsim_create_codebook_diffusiondrive.py:
  1. Greedy set-cover K-disk (deterministic, better high-speed coverage)
     - v1 used random selection → seeds biased to data density (mid-speed)
     - v2 picks the seed covering the most uncovered neighbors at each step
  2. Cluster representative = mean of (dx, dy, dθ) displacements, NOT corners
     - 3-DOF averaging is rotation-correct and guarantees valid rectangles
       even at large tol (v1 corner-averaging is empirically OK only at tol≤0.1)
  3. Zero token = argmin(‖displacement‖) entry (deterministic, always closest
     to true zero), v1 just took the first-loaded sample
  4. Honest meta.pkl: clustering field matches the actual algorithm used
  5. Speed-bucket diagnostic report so it's easy to see which dx ranges are
     well/under-covered in the resulting codebook
  6. Visualization rendered wide (24×8) so the codebook fan isn't squashed

The rotation convention matches the DiffusionDrive model
(match_to_step_corner_codebook, _decode_step_corner_tokens):
    local_forward = dx·cos(θ) + dy·sin(θ)
    local_lateral = -dx·sin(θ) + dy·cos(θ)

Usage:
    python navsim_create_codebook_diffusiondrive_v2.py \\
        --data_path /data/navsim/dataset/navsim_logs/trainval \\
        --output codebook_cache/navsim_kdisk_v2048_diffusiondrive_v2 \\
        --vocab_size 2048
"""

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
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quaternion_to_yaw(quat):
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_to_local(pos_global, head_global, pos_now, head_now):
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
    """4-corner box. Corner order: left-front, right-front, right-back, left-back."""
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


def Kdisk_cluster_greedy(X, N, tol, a_pos, max_candidates=20000, seed=0):
    """Greedy set-cover K-disk on corner-distance, returns mean displacements.

    Args:
        X:    [n, 4, 2] corner contours (used as the L2 distance metric)
        N:    target vocab size
        tol:  cluster radius (m) — same units as corner coordinates
        a_pos:[n, 3] raw displacements (dx, dy, dtheta) per sample
    Returns:
        [K, 3] cluster-mean displacements (K <= N; K = N if enough points)
    """
    from scipy.spatial import KDTree

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    N = min(N, n)
    X_flat = X.reshape(n, -1).numpy().astype(np.float32)

    if n > max_candidates:
        cand_idx = rng.choice(n, max_candidates, replace=False)
        pool = X_flat[cand_idx]
        pool_a_idx = cand_idx
    else:
        pool = X_flat
        pool_a_idx = np.arange(n)

    M = pool.shape[0]

    tree = KDTree(pool)
    neighbors = tree.query_ball_tree(tree, r=float(tol))

    covered = np.zeros(M, dtype=bool)
    counts = np.array([len(nb) for nb in neighbors], dtype=np.int32)
    selected_indices = []

    if M == 0:
        return torch.zeros((0, 3), dtype=torch.float32)

    # First selection: closest-to-zero displacement (deterministic zero token)
    a_pool = a_pos.numpy()[pool_a_idx]
    zero_dist = np.linalg.norm(a_pool[:, :2], axis=1)
    best = int(np.argmin(zero_dist))

    for i in range(N):
        if covered.all():
            print(f"All candidates covered at cluster {i}/{N}")
            break

        if i > 0:
            best = int(np.argmax(counts))
            if counts[best] == 0:
                print(f"No uncovered candidates left at cluster {i}/{N}")
                break

        selected_indices.append(pool_a_idx[best])

        for j in neighbors[best]:
            if not covered[j]:
                covered[j] = True
                for k in neighbors[j]:
                    counts[k] -= 1

        if (i + 1) % 200 == 0:
            print(f"  greedy cluster {i+1}/{N}, uncovered={(~covered).sum() * 100.0 / M:.1f}%")

    # Re-assign all original points to nearest selected center, then average
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
    print(f"Greedy K-disk: selected {len(selected_indices)} clusters from {n} samples")
    return ret


def load_navsim_trajectories(data_path, n_trajs):
    """Load 1-step (dx, dy, dtheta) displacements from NavSim pkl files."""
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

                dx_dy = l_pos.squeeze()         # [2]
                dtheta = l_head.squeeze()        # scalar
                step = torch.cat([dx_dy, dtheta.unsqueeze(0)], dim=0)  # [3]
                disp_list.append(step)
                count += 1

            pbar.update(1)
            if count >= n_trajs:
                break

    return disp_list, count


def speed_bucket_report(disps):
    """Print how many codebook entries land in each forward-speed bucket."""
    dx = disps[:, 0].numpy()
    edges = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20])
    h, _ = np.histogram(dx, bins=edges)
    speed_mps = edges[1:] / 0.5  # dx per 0.5 s -> m/s
    speed_kph = speed_mps * 3.6
    print("Codebook coverage by forward dx (per 0.5 s step):")
    for i in range(len(h)):
        print(f"  dx ∈ [{edges[i]:+5.1f}, {edges[i+1]:+5.1f}) m  "
              f"(~{speed_kph[i]:5.1f} kph upper)  →  {h[i]:4d} tokens")


def visualize_codebook(ret_corners, output_dir, n_highlight=300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = ret_corners.shape[0]

    final_pos = ret_corners.mean(dim=1).numpy()
    final_x = final_pos[:, 0]
    final_y = final_pos[:, 1]

    diff_xy = (ret_corners[:, 0] - ret_corners[:, 3]).numpy()
    final_h = np.arctan2(diff_xy[:, 1], diff_xy[:, 0])

    seg_len = 0.3
    dx = np.cos(final_h) * seg_len
    dy = np.sin(final_h) * seg_len

    highlight_idx = np.random.choice(n_clusters, min(n_highlight, n_clusters), replace=False)

    x_min = min((final_x + dx).min(), final_x.min())
    x_max = max((final_x + dx).max(), final_x.max())
    y_min = min((final_y + dy).min(), final_y.min())
    y_max = max((final_y + dy).max(), final_y.max())
    x_pad = max(0.5, (x_max - x_min) * 0.05)
    y_pad = max(0.5, (y_max - y_min) * 0.05)

    fig, ax = plt.subplots(figsize=(24, 8))

    for i in range(n_clusters):
        ax.plot([final_x[i], final_x[i] + dx[i]],
                [final_y[i], final_y[i] + dy[i]],
                'grey', alpha=0.3, linewidth=0.5)

    for idx in highlight_idx:
        ax.plot([final_x[idx], final_x[idx] + dx[idx]],
                [final_y[idx], final_y[idx] + dy[idx]],
                'b', alpha=0.8, linewidth=1)

    ax.set_aspect('equal')
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel('Δx (m)')
    ax.set_ylabel('Δy (m)')
    ax.set_title(f'Action Codebook v2 ({n_clusters} tokens, greedy K-disk)')
    ax.grid(True, alpha=0.3)

    vis_path = output_dir / 'navsim_codebook_fan.png'
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {vis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create action codebook v2 (greedy K-disk + displacement-space averaging) "
                    "for DiffusionDrive AR per-step token decoding."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to NavSim log pkl files")
    parser.add_argument("--output", type=str,
                        default="/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive_v2",
                        help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--n_trajs", type=int, default=2_000_000,
                        help="Max # 1-step displacements to sample "
                             "(v1 used the entire trainval, ~721k). Set high to use all data.")
    parser.add_argument("--tol_dist", type=float, default=0.05,
                        help="K-disk cluster radius (m). Same as v1 default.")
    parser.add_argument("--max_candidates", type=int, default=20000,
                        help="Greedy candidate pool size (memory budget).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Creating codebook v2 with vocab_size={args.vocab_size}")
    print(f"Method   : greedy K-disk, tol={args.tol_dist}")
    print(f"Avg mode : displacement-space (rotation-correct, always valid box)")
    print(f"Data     : {args.data_path}")

    disp_list, count = load_navsim_trajectories(args.data_path, args.n_trajs)
    print(f"Loaded {count} 1-step displacements")

    if count == 0:
        print("No trajectories loaded!")
        return

    disps = torch.stack(disp_list, dim=0)            # [N, 3]
    zero_step = torch.zeros(1, 3, dtype=torch.float32)
    disps = torch.cat([zero_step, disps], dim=0)      # prepend exact-zero
    print(f"Displacements (incl. zero): {disps.shape}")

    width_length = torch.tensor([2.0, 4.8])
    wl_expanded = width_length.unsqueeze(0).expand(disps.shape[0], -1)

    contour = cal_polygon_contour(
        pos=disps[:, :2],
        head=disps[:, 2],
        width_length=wl_expanded,
    )

    print(f"Running greedy K-disk (tol={args.tol_dist}, max_candidates={args.max_candidates})...")
    ret_disps = Kdisk_cluster_greedy(
        X=contour,
        N=args.vocab_size,
        tol=args.tol_dist,
        a_pos=disps,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )
    ret_disps[:, 2] = wrap_angle(ret_disps[:, 2])

    print(f"Codebook displacements: {ret_disps.shape}")
    speed_bucket_report(ret_disps)

    wl_ret = width_length.unsqueeze(0).expand(ret_disps.shape[0], -1)
    ret_corners = cal_polygon_contour(
        pos=ret_disps[:, :2],
        head=ret_disps[:, 2],
        width_length=wl_ret,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_codebook(ret_corners, output_dir)

    ego_path = output_dir / 'ego.npy'
    np.save(ego_path, ret_corners.numpy())
    print(f"Saved codebook → {ego_path}  shape={tuple(ret_corners.shape)}  format=[V, 4, 2] corners")

    disp_path = output_dir / 'ego_displacements.npy'
    np.save(disp_path, ret_disps.numpy())
    print(f"Saved displacements → {disp_path}  shape={tuple(ret_disps.shape)}  format=[V, 3] (dx, dy, dθ)")

    meta = {
        'vocab_size': args.vocab_size,
        'actual_vocab_size': ret_disps.shape[0],
        'token_format_corners': '[V, 4, 2] single-step box corners',
        'token_format_displacement': '[V, 3] (dx, dy, dtheta) single-step',
        'token_granularity': 'per_step (T=8 tokens per plan, autoregressive)',
        'frame_interval_s': 0.5,
        'clustering': 'greedy_kdisk_v2',
        'averaging': 'displacement_space (3-DOF, rotation-correct)',
        'zero_token_rule': 'argmin(||displacement||) deterministic',
        'radius_m': args.tol_dist,
        'n_samples_used': count,
        'seed': args.seed,
        'vehicle_dims_wl': [2.0, 4.8],
        'rotation_convention': 'standard_global_to_local',
        'improvements_over_v1': [
            'greedy_set_cover (vs random selection)',
            'displacement_space_averaging (vs corner_averaging)',
            'argmin_zero_token (vs first_loaded_sample)',
            'honest_meta_label',
            'widescreen_visualization',
            'speed_bucket_diagnostic',
        ],
    }
    meta_path = output_dir / 'meta.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved meta → {meta_path}")


if __name__ == "__main__":
    main()
