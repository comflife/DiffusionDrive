"""
Create action codebook from NavSim data.

NavSim log format
-----------------
Each .pkl file under  ``navsim_logs/trainval/``  is a list of frame dicts
recorded at **2 Hz** (0.5 s per frame).  Each dict contains:

    ego2global_translation : [x, y, z]           (float64, metres)
    ego2global_rotation    : [w, x, y, z]         (quaternion)

AutoVLA-style codebook
----------------------
Following AutoVLA (action_token_cluster.py), each codebook token is a
**single-step displacement** (Δx, Δy, Δθ) between two consecutive frames.

    trajs shape : [N, 1, 3]

This matches the original AutoVLA paper where each token represents
0.5 seconds of vehicle motion (one frame interval at 2 Hz), characterised
by its final-frame bounding-box contour displacement.
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
from matplotlib.lines import Line2D
from tqdm import tqdm
import argparse


# ---------------------------------------------------------------------------
# Utility functions (self-contained, no external dependency)
# ---------------------------------------------------------------------------

def wrap_angle(angle):
    """Wrap angle (scalar or tensor) to [-π, π]."""
    if isinstance(angle, torch.Tensor):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quaternion_to_yaw(quat):
    """
    Convert quaternion [w, x, y, z] → yaw (radians).

    Uses the ZYX (intrinsic) convention which matches the pyquaternion
    ``yaw_pitch_roll`` used inside the NavSim codebase.
    """
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
    cos_h = head_now.cos()  # [B]
    sin_h = head_now.sin()  # [B]
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), dtype=head_now.dtype)
    rot_mat[:, 0, 0] = cos_h
    rot_mat[:, 0, 1] = -sin_h
    rot_mat[:, 1, 0] = sin_h
    rot_mat[:, 1, 1] = cos_h

    diff = pos_global - pos_now.unsqueeze(1)  # [B, N, 2]
    local_pos = torch.bmm(diff, rot_mat)       # [B, N, 2]

    local_head = head_global - head_now.unsqueeze(-1)  # [B, N]
    return local_pos, local_head


def cal_polygon_contour(pos, head, width_length):
    """
    Compute 4-corner bounding-box polygons.

    Args:
        pos:          [..., 2]  centre positions
        head:         [...]     headings
        width_length: broadcastable [2] or [..., 2]  (width, length)

    Returns:
        contour: [..., 4, 2]
    """
    w = width_length[..., 0] / 2.0   # half width
    l = width_length[..., 1] / 2.0   # half length

    cos_h = torch.cos(head).unsqueeze(-1)  # [..., 1]
    sin_h = torch.sin(head).unsqueeze(-1)

    # four corners in local frame: FL, FR, RR, RL
    dx = torch.stack([l, l, -l, -l], dim=-1)  # [..., 4]
    dy = torch.stack([w, -w, -w, w], dim=-1)

    # rotate
    cx = dx * cos_h - dy * sin_h  # [..., 4]
    cy = dx * sin_h + dy * cos_h

    corners = torch.stack([cx, cy], dim=-1)  # [..., 4, 2]
    corners = corners + pos.unsqueeze(-2)
    return corners


def seed_everything(seed):
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# K-disk clustering  (identical to AutoVLA action_token_cluster.py)
# ---------------------------------------------------------------------------

def Kdisk_cluster(
    X,       # [n_trajs, 4, 2], bbox of the last point of the segment
    N,       # int
    tol,     # float
    a_pos,   # [n_trajs, 1, 3], single-step displacement
    cal_mean_heading=True,
):
    n_total = X.shape[0]
    ret_traj_list = []

    for i in range(N):
        if X.shape[0] == 0:
            print(f"Warning: ran out of data at cluster {i}/{N}")
            break

        if i == 0:
            choice_index = 0  # always include the zero displacement
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
        print(f"{i=}, {remain=:.2f}%, {n_inside=}")

    return torch.cat(ret_traj_list, dim=0)


# ---------------------------------------------------------------------------
# NavSim data loading  (AutoVLA style: single-step displacements)
# ---------------------------------------------------------------------------

MIN_DISPLACEMENT = 0.0   # include all steps (including near-zero for stop token)


def load_navsim_single_step_displacements(data_path, n_trajs,
                                          min_displacement=MIN_DISPLACEMENT):
    """
    Load single-step displacements from NavSim pkl log files.

    For each consecutive pair of frames (t → t+1), compute the displacement
    (Δx, Δy, Δθ) in the local ego frame of frame t.

    This exactly mirrors AutoVLA's action_token_cluster.py:
        pos  = previous frame position
        head = previous frame heading
        next_pos / next_head = current frame pose

    Args:
        data_path:        Path to NavSim log pkl files
        n_trajs:          Maximum number of single-step displacements to load
        min_displacement: Minimum displacement (m) to filter (default 0.0)

    Returns
    -------
    trajs  : Tensor  [N, 1, 3]
    count  : int
    """
    pkl_files = sorted(glob.glob(os.path.join(data_path, "*.pkl")))
    assert len(pkl_files) > 0, f"No pkl files found in {data_path}"
    print(f"Found {len(pkl_files)} log files in {data_path}")
    print(f"AutoVLA mode: single-step displacement (0.5s per token)")

    traj_list = []
    count = 0

    with tqdm(total=len(pkl_files),
              desc=f"Loading displacements (target={n_trajs})") as pbar:
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
                rot   = frame["ego2global_rotation"]
                yaw   = quaternion_to_yaw(rot)
                ego_poses.append((trans[0], trans[1], yaw))

            # For each consecutive pair, compute 1-step displacement
            for t in range(len(ego_poses) - 1):
                if count >= n_trajs:
                    break

                pos  = torch.tensor([ego_poses[t][:2]],   dtype=torch.float32)   # [1, 2]
                head = torch.tensor([ego_poses[t][2]],    dtype=torch.float32)   # [1]
                next_pos  = torch.tensor([ego_poses[t+1][:2]], dtype=torch.float32)
                next_head = torch.tensor([ego_poses[t+1][2]],  dtype=torch.float32)

                l_pos, l_head = transform_to_local(
                    pos_global=next_pos.unsqueeze(0),    # [1, 1, 2]
                    head_global=next_head.unsqueeze(0),  # [1, 1]
                    pos_now=pos,                          # [1, 2]
                    head_now=head,                        # [1]
                )
                l_head = wrap_angle(l_head)

                # shape [1, 3]
                step = torch.cat([l_pos.squeeze(0), l_head.unsqueeze(-1).squeeze(0)], dim=-1)

                # Optional: skip near-stationary
                if min_displacement > 0.0:
                    if torch.norm(step[0, :2]).item() < min_displacement:
                        continue

                traj_list.append(step.unsqueeze(0))   # [1, 1, 3]
                count += 1

            pbar.update(1)
            if count >= n_trajs:
                break

    return traj_list, count


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_codebook(ret_traj, output_dir, n_highlight=300):
    """
    AutoVLA Fig S1(a) style — heading quiver at (Δx, Δy).

    Each codebook token is a single-step displacement (Δx, Δy, Δθ).
    The visualisation places a short line segment at (Δx, Δy) oriented
    along Δθ.  This exactly reproduces AutoVLA Fig S1(a) where:
      - grey segments  = all K tokens
      - blue segments  = randomly highlighted subset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = ret_traj.shape[0]

    # Final (and only) waypoint
    final_x = ret_traj[:, -1, 0].numpy()
    final_y = ret_traj[:, -1, 1].numpy()
    final_h = ret_traj[:, -1, 2].numpy()

    # Heading direction unit vectors, scaled for visual clarity
    seg_len = 0.3  # length of the heading indicator line (metres)
    dx = np.cos(final_h) * seg_len
    dy = np.sin(final_h) * seg_len

    highlight_idx = np.random.choice(
        n_clusters, size=min(n_highlight, n_clusters), replace=False
    )
    highlight_mask = np.zeros(n_clusters, dtype=bool)
    highlight_mask[highlight_idx] = True

    # ---- quiver-style plot (paper Fig S1(a)) ----
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor("white")

    # Draw all tokens (grey, background)
    bg = ~highlight_mask
    ax.quiver(
        final_x[bg], final_y[bg], dx[bg], dy[bg],
        angles="xy", scale_units="xy", scale=1,
        color="grey", alpha=0.35, width=0.0015, headwidth=2.5, headlength=3,
    )

    # Draw highlighted tokens (blue, foreground)
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

    # ---- scatter of final positions ----
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create AutoVLA-style action codebook from NavSim data"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_path", type=str,
        default="/data/navsim/dataset/navsim_logs/trainval",
        help="Root directory of NavSim log pkl files",
    )
    parser.add_argument(
        "--out_file_name", type=str,
        default="agent_vocab_navsim_kdisk.pkl",
        help="Output filename (saved inside codebook_cache/)",
    )
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument(
        "--n_trajs", type=int, default=2048 * 200,
        help="Max number of single-step displacements to sample",
    )
    parser.add_argument(
        "--min_displacement", type=float, default=0.0,
        help="Minimum displacement (m) to filter nearly-stationary steps "
             "(0.0 keeps all, including stop token)",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    # ---- output directory ----
    out_dir = Path(__file__).resolve().parent / "codebook_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_file_name
    print(f"Output will be saved to: {out_path}")
    print(f"NavSim frame interval : 0.5 s (2 Hz)")
    print(f"Token type            : single-step displacement (AutoVLA style)")

    # ---- parameters ----
    tol_dist    = 0.05
    num_cluster = args.vocab_size

    # ---- load data ----
    # Start with the zero-displacement (stop) token, shape [1, 1, 3]
    zero_step = torch.zeros([1, 1, 3], dtype=torch.float32)
    segments_list = [zero_step]

    loaded, total_count = load_navsim_single_step_displacements(
        args.data_path, args.n_trajs,
        min_displacement=args.min_displacement,
    )
    segments_list.extend(loaded)
    print(f"Loaded {total_count} single-step displacements")

    trajs = torch.cat(segments_list, dim=0)  # [N, 1, 3]
    print(f"Total displacement vectors: {trajs.shape}")

    # ---- K-disk cluster ----
    res = {"token_all": {}}

    width_length = torch.tensor([2.0, 4.8])
    width_length = width_length.unsqueeze(0)  # [1, 2]

    # Bounding-box contour of the final (only) waypoint
    contour = cal_polygon_contour(
        pos=trajs[:, -1, :2], head=trajs[:, -1, 2], width_length=width_length
    )  # [N, 4, 2]

    ret_traj = Kdisk_cluster(
        X=contour, N=num_cluster, tol=tol_dist, a_pos=trajs
    )
    ret_traj[:, :, -1] = wrap_angle(ret_traj[:, :, -1])

    # ---- visualisation ----
    visualize_codebook(ret_traj, out_dir)

    # ---- compute final contours and save ----
    contour = cal_polygon_contour(
        pos=ret_traj[:, :, :2],   # [K, 1, 2]
        head=ret_traj[:, :, 2],   # [K, 1]
        width_length=width_length.unsqueeze(0),
    )

    res["token_all"]["veh"] = contour.numpy()

    with open(out_path, "wb") as f:
        pickle.dump(res, f)

    print(f"\nCodebook saved to: {out_path}")
    print(f"  vocab size     : {ret_traj.shape[0]}")
    print(f"  token steps    : {ret_traj.shape[1]}  (AutoVLA: 1 step = 0.5 s)")
    print(f"  contour shape  : {contour.shape}")
