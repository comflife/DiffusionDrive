"""
Unified Codebook Generation v2 for DiffusionDrive and VAD.

This script provides a unified interface for creating motion token codebooks
using K-disk clustering (greedy set-cover with KDTree) which is more robust
than the tolerance-based approach.

Supports:
- NavSim data (DiffusionDrive format)
- nuScenes/VAD data (VAD format)
- Custom trajectory data

Usage:
    # For NavSim (DiffusionDrive style)
    python create_codebook_v2.py \
        --data_path /path/to/navsim_logs \
        --data_type navsim \
        --vocab_size 256 \
        --output codebook_cache/unified_codebook.pkl

    # For nuScenes/VAD style
    python create_codebook_v2.py \
        --data_path /path/to/nuscenes_infos.pkl \
        --data_type nuscenes \
        --vocab_size 512 \
        --superclasses vehicle pedestrian cyclist ego \
        --output codebook_cache/vad_codebook.pkl
"""

import argparse
import logging
import pickle
import glob
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Sequence, Mapping

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

log = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

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
    if isinstance(pos_global, np.ndarray):
        pos_global = torch.from_numpy(pos_global)
    if isinstance(head_global, np.ndarray):
        head_global = torch.from_numpy(head_global)
    if isinstance(pos_now, np.ndarray):
        pos_now = torch.from_numpy(pos_now)
    if isinstance(head_now, np.ndarray):
        head_now = torch.from_numpy(head_now)
    
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
    """Compute 4-corner bounding box polygons."""
    if isinstance(pos, torch.Tensor):
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
    else:
        # NumPy version
        pos = np.asarray(pos)
        head = np.asarray(head)
        width_length = np.asarray(width_length)
        
        w = width_length[..., 0] / 2.0
        l = width_length[..., 1] / 2.0

        cos_h = np.cos(head)[..., np.newaxis]
        sin_h = np.sin(head)[..., np.newaxis]

        dx = np.stack([l, l, -l, -l], axis=-1)
        dy = np.stack([w, -w, -w, w], axis=-1)

        cx = dx * cos_h - dy * sin_h
        cy = dx * sin_h + dy * cos_h

        corners = np.stack([cx, cy], axis=-1)
        corners = corners + pos[..., np.newaxis, :]
        return corners


def seed_everything(seed: int):
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Unified K-disk Clustering (VAD-style with improvements)
# ============================================================================

def fit_kdisk_unified(
    vectors: np.ndarray,
    n_clusters: int,
    radius: Optional[float] = None,
    seed: int = 0,
    max_candidates: int = 10000,
    use_contour: bool = False,
    width_length: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified K-disk clustering with optional contour-based distance.
    
    This is the improved version based on VAD's _fit_kdisk with support for
    contour-based clustering (DiffusionDrive style).
    
    Args:
        vectors: [N, D] trajectory vectors (e.g., [N, 1, 3] for single-step)
        n_clusters: number of cluster centers to select
        radius: disk radius (None = auto-estimate)
        seed: random seed
        max_candidates: max candidate pool size
        use_contour: if True, use bounding box contour for distance computation
        width_length: [2] vehicle dimensions (width, length) for contour
    
    Returns:
        centers: [K, D] selected cluster centers (K <= n_clusters)
        indices: [K] indices of selected centers in original vectors
    """
    from scipy.spatial import KDTree
    
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    n_clusters = min(n_clusters, n)
    
    # Flatten vectors if needed for KDTree
    original_shape = vectors.shape
    if vectors.ndim > 2:
        vectors_flat = vectors.reshape(n, -1)
    else:
        vectors_flat = vectors
    
    # Compute contour if requested
    if use_contour and width_length is not None:
        # Extract position and heading from vectors
        # Assumes vectors are [N, T, 3] with (x, y, theta)
        if vectors.ndim == 3:
            pos = vectors[:, -1, :2]  # [N, 2]
            head = vectors[:, -1, 2]  # [N]
        else:
            pos = vectors[:, :2]
            head = vectors[:, 2] if vectors.shape[1] >= 3 else np.zeros(n)
        
        contour = cal_polygon_contour(pos, head, width_length)  # [N, 4, 2]
        contour_flat = contour.reshape(n, -1)  # [N, 8]
        
        # Use contour for distance computation
        distance_vectors = contour_flat
    else:
        distance_vectors = vectors_flat
    
    # Subsample candidate pool
    if n > max_candidates:
        cand_idx = rng.choice(n, max_candidates, replace=False)
        pool = distance_vectors[cand_idx].astype(np.float32)
        pool_original_idx = cand_idx
    else:
        cand_idx = np.arange(n)
        pool = distance_vectors.astype(np.float32)
        pool_original_idx = cand_idx
    
    M = pool.shape[0]
    
    # Auto-estimate radius from mean nearest-neighbour distance
    if radius is None:
        probe_n = min(2000, M)
        probe = pool[:probe_n]
        sq = ((probe[:, None, :] - probe[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(sq, np.inf)
        radius = float(np.sqrt(sq.min(axis=1).mean()))
        log.info(f"K-disk auto radius={radius:.4f}")
    
    # Precompute neighborhoods with KDTree
    tree = KDTree(pool)
    neighbors = tree.query_ball_tree(tree, r=float(radius))
    
    covered = np.zeros(M, dtype=bool)
    counts = np.array([len(nb) for nb in neighbors], dtype=np.int32)
    selected_indices = []
    
    for i in range(n_clusters):
        if covered.all():
            log.warning(f"All points covered at iteration {i}, stopping early")
            break
        
        # Pick the point covering the most uncovered neighbors
        # For first iteration, optionally force zero displacement
        if i == 0:
            # Find closest to zero
            best = int(np.argmin(np.linalg.norm(pool, axis=1)))
        else:
            best = int(np.argmax(counts))
        
        selected_indices.append(pool_original_idx[best])
        
        # Mark neighbors as covered
        for j in neighbors[best]:
            if not covered[j]:
                covered[j] = True
                for k in neighbors[j]:
                    counts[k] -= 1
    
    if not selected_indices:
        return np.zeros((0, vectors_flat.shape[1]), dtype=np.float32), np.array([], dtype=np.int32)
    
    result = vectors[np.array(selected_indices)]
    log.info(f"K-disk done: selected {len(selected_indices)} centers from {n} vectors")
    
    return result, np.array(selected_indices, dtype=np.int32)


# ============================================================================
# Data Loaders
# ============================================================================

class NavSimDataLoader:
    """Load single-step displacements from NavSim pkl files."""
    
    def __init__(self, data_path: str, n_trajs: int = 100000, 
                 min_displacement: float = 0.0):
        self.data_path = data_path
        self.n_trajs = n_trajs
        self.min_displacement = min_displacement
    
    def load(self) -> Dict[str, np.ndarray]:
        """Load and return token banks."""
        pkl_files = sorted(glob.glob(os.path.join(self.data_path, "*.pkl")))
        assert len(pkl_files) > 0, f"No pkl files found in {self.data_path}"
        log.info(f"Found {len(pkl_files)} NavSim log files")
        
        traj_list = []
        count = 0
        
        with tqdm(total=len(pkl_files), desc="Loading NavSim data") as pbar:
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
                    if count >= self.n_trajs:
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
                    
                    if self.min_displacement > 0.0:
                        if torch.norm(step[0, :2]).item() < self.min_displacement:
                            continue
                    
                    traj_list.append(step.unsqueeze(0).numpy())
                    count += 1
                
                pbar.update(1)
                if count >= self.n_trajs:
                    break
        
        if not traj_list:
            return {"ego": np.zeros((0, 1, 3), dtype=np.float32)}
        
        trajs = np.concatenate(traj_list, axis=0)  # [N, 1, 3]
        log.info(f"Loaded {trajs.shape[0]} NavSim trajectories")
        return {"ego": trajs}


class NuScenesDataLoader:
    """Load tokens from nuScenes/VAD pickle info files."""
    
    SUPERCLASS_TO_NAMES = {
        "vehicle": ("car", "truck", "construction_vehicle", "bus", "trailer"),
        "pedestrian": ("pedestrian",),
        "cyclist": ("bicycle", "motorcycle"),
    }
    
    def __init__(self, info_path: str, granularity: str = "step", 
                 chunk_size: int = 1, max_samples: Optional[int] = None,
                 agent_local: bool = True):
        self.info_path = info_path
        self.granularity = granularity
        self.chunk_size = chunk_size
        self.max_samples = max_samples
        self.agent_local = agent_local
        
        self._class_to_super = {
            name: sc for sc, names in self.SUPERCLASS_TO_NAMES.items() 
            for name in names
        }
    
    def _extract_step_tokens(self, traj: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract per-step deltas for valid time steps."""
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim == 1:
            if traj.size % 2 != 0:
                raise ValueError(f"1-D traj must have even length, got {traj.size}")
            traj = traj.reshape(-1, 2)
        mask = np.asarray(mask, dtype=np.float32).reshape(-1)
        assert mask.shape[0] == traj.shape[0], "traj/mask length mismatch"
        return traj[mask > 0].astype(np.float32, copy=False)
    
    def _lidar_to_agent_local(self, deltas: np.ndarray, yaw_lidar: float) -> np.ndarray:
        """Rotate per-step deltas from LiDAR frame into agent-local frame."""
        angle = float(yaw_lidar) + np.pi
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (deltas @ R.T).astype(np.float32, copy=False)
    
    def load(self) -> Dict[str, np.ndarray]:
        """Load and return token banks."""
        with open(self.info_path, "rb") as f:
            payload = pickle.load(f)
        
        if isinstance(payload, dict) and "infos" in payload:
            infos = payload["infos"]
        elif isinstance(payload, list):
            infos = payload
        else:
            raise TypeError(f"Unexpected pickle type: {type(payload)}")
        
        if self.max_samples is not None:
            infos = infos[:self.max_samples]
        
        log.info(f"Loaded {len(infos)} samples from {self.info_path}")
        
        superclasses = list(self.SUPERCLASS_TO_NAMES.keys()) + ["ego"]
        lists = {sc: [] for sc in superclasses}
        
        for info in tqdm(infos, desc="Processing nuScenes data"):
            gt_names = np.asarray(info.get("gt_names", []))
            gt_trajs = np.asarray(info.get("gt_agent_fut_trajs", []), dtype=np.float32)
            gt_masks = np.asarray(info.get("gt_agent_fut_masks", []), dtype=np.float32)
            gt_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32)
            
            if gt_trajs.size > 0 and gt_masks.size > 0:
                for i, (name, traj, mask) in enumerate(zip(gt_names, gt_trajs, gt_masks)):
                    sc = self._class_to_super.get(str(name))
                    if sc is None:
                        continue
                    
                    traj = np.asarray(traj, dtype=np.float32)
                    if traj.ndim == 1:
                        traj = traj.reshape(-1, 2)
                    
                    if self.agent_local and gt_boxes.shape[0] > i:
                        traj = self._lidar_to_agent_local(traj, gt_boxes[i, 6])
                    
                    vecs = self._extract_step_tokens(traj, mask)
                    if vecs.size:
                        lists[sc].append(vecs)
            
            # Ego trajectory
            ego_traj = np.asarray(info.get("gt_ego_fut_trajs", []), dtype=np.float32)
            ego_mask = np.asarray(info.get("gt_ego_fut_masks", []), dtype=np.float32)
            if ego_traj.size > 0 and ego_mask.size > 0:
                vecs = self._extract_step_tokens(ego_traj, ego_mask)
                if vecs.size:
                    lists["ego"].append(vecs)
        
        banks = {}
        for sc, parts in lists.items():
            if parts:
                banks[sc] = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
                log.info(f"  [{sc}] {banks[sc].shape[0]} tokens, dim={banks[sc].shape[1]}")
            else:
                banks[sc] = np.zeros((0, 2), dtype=np.float32)
        
        return banks


# ============================================================================
# Codebook Creation
# ============================================================================

class UnifiedCodebookCreator:
    """Create motion token codebooks using unified K-disk clustering."""
    
    def __init__(
        self,
        vocab_size: int = 512,
        radius: Optional[float] = None,
        seed: int = 0,
        max_candidates: int = 10000,
        use_contour: bool = False,
        vehicle_dims: Tuple[float, float] = (2.0, 4.8),
    ):
        self.vocab_size = vocab_size
        self.radius = radius
        self.seed = seed
        self.max_candidates = max_candidates
        self.use_contour = use_contour
        self.vehicle_dims = np.array(vehicle_dims)
        self.codebooks = {}
    
    def create(self, token_banks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create codebooks for each superclass."""
        for sc, vectors in token_banks.items():
            if vectors.shape[0] == 0:
                log.warning(f"No tokens for '{sc}', skipping")
                self.codebooks[sc] = np.zeros((0, vectors.shape[1] if vectors.ndim > 1 else 2), dtype=np.float32)
                continue
            
            log.info(f"Creating codebook for '{sc}' with {vectors.shape[0]} vectors")
            
            # For ego/vehicle with (x, y, theta), use contour
            use_contour = self.use_contour and vectors.shape[-1] >= 3
            
            centers, indices = fit_kdisk_unified(
                vectors=vectors,
                n_clusters=self.vocab_size,
                radius=self.radius,
                seed=self.seed,
                max_candidates=self.max_candidates,
                use_contour=use_contour,
                width_length=self.vehicle_dims if use_contour else None,
            )
            
            self.codebooks[sc] = centers
            log.info(f"  -> Created {centers.shape[0]} clusters")
        
        return self.codebooks
    
    def save(self, output_path: Union[str, Path], format: str = "pkl", 
             save_contour: bool = True):
        """Save codebooks to file.
        
        NavSim/DiffusionDrive format:
            codebook_cache/navsim_kdisk_v{vocab_size}/
                ├── ego.npy          # codebook centers [K, T, 3]
                ├── meta.pkl         # metadata
                └── *.png            # visualizations
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save codebook as ego.npy
        if "ego" in self.codebooks and self.codebooks["ego"].shape[0] > 0:
            centers = self.codebooks["ego"]
            np.save(output_path / "ego.npy", centers)
            log.info(f"Saved ego.npy with shape {centers.shape}")
        else:
            # Fallback: save first available
            for sc, centers in self.codebooks.items():
                if centers.shape[0] > 0:
                    np.save(output_path / "ego.npy", centers)
                    log.info(f"Saved ego.npy (from {sc}) with shape {centers.shape}")
                    break
        
        # Save metadata
        meta = {
            "vocab_size": self.vocab_size,
            "radius": self.radius,
            "seed": self.seed,
            "max_candidates": self.max_candidates,
            "use_contour": self.use_contour,
            "vehicle_dims": self.vehicle_dims.tolist(),
            "actual_vocab_sizes": {sc: c.shape[0] for sc, c in self.codebooks.items()},
        }
        with open(output_path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        log.info(f"Saved meta.pkl to {output_path}")
        
        # Also save contour version if requested (for compatibility)
        if save_contour:
            res = {"token_all": {}}
            for sc, centers in self.codebooks.items():
                if centers.shape[0] > 0:
                    if centers.shape[-1] >= 3:
                        pos = centers[..., :2]
                        head = centers[..., 2]
                        contour = cal_polygon_contour(pos, head, self.vehicle_dims)
                        res["token_all"][sc] = contour
                    else:
                        res["token_all"][sc] = centers
            
            with open(output_path / "token_all.pkl", "wb") as f:
                pickle.dump(res, f)
            log.info(f"Saved token_all.pkl (contour format)")
    
    def visualize(self, output_dir: Union[str, Path], n_highlight: int = 300, 
                  data_type: str = "navsim"):
        """Visualize codebooks in fan-style (matching existing format).
        
        Args:
            output_dir: Output directory
            n_highlight: Number of tokens to highlight
            data_type: 'navsim' or 'vad' - determines filename format
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for sc, centers in self.codebooks.items():
            if centers.shape[0] == 0:
                continue
            
            n_clusters = centers.shape[0]
            
            if centers.ndim == 3 and centers.shape[1] == 1:
                # Single-step [N, 1, 3]
                final_x = centers[:, -1, 0]
                final_y = centers[:, -1, 1]
            elif centers.shape[-1] >= 3:
                # Has heading
                final_x = centers[..., 0]
                final_y = centers[..., 1]
            else:
                # Just positions
                final_x = centers[..., 0]
                final_y = centers[..., 1]
            
            # Create fan-style visualization (matching existing format)
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Draw lines from origin to each point (fan style)
            for i in range(n_clusters):
                ax.plot([0, final_x[i]], [0, final_y[i]], 
                       color='steelblue', alpha=0.3, linewidth=0.8)
            
            # Scatter plot of all tokens
            ax.scatter(final_x, final_y, c='tomato', s=15, alpha=0.8, zorder=5)
            
            # Labels and title matching existing format
            ax.set_xlabel("forward / Δx (m)", fontsize=12)
            ax.set_ylabel("lateral / Δy (m)", fontsize=12)
            ax.set_title(f"{sc} (V={n_clusters})", fontsize=14)
            ax.grid(True, alpha=0.3, linestyle='-')
            ax.set_aspect('equal', adjustable='box')
            
            # Add main title for NavSim format
            if data_type == "navsim":
                fig.suptitle(f"NavSim Action Codebook — Greedy K-Disk (V={n_clusters}, 1 step = 0.5 s)", 
                            fontsize=14, y=0.98)
            
            plt.tight_layout()
            
            # Use existing naming convention
            if data_type == "navsim":
                vis_path = output_dir / "navsim_codebook_fan.png"
            else:
                vis_path = output_dir / f"codebook_{sc}.png"
            
            fig.savefig(vis_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Saved visualization to {vis_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Codebook Generation v2 for DiffusionDrive and VAD"
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data (NavSim logs dir or nuScenes info pkl)")
    parser.add_argument("--data_type", type=str, choices=["navsim", "nuscenes"], 
                        default="navsim",
                        help="Type of data to load")
    parser.add_argument("--n_trajs", type=int, default=100000,
                        help="Max trajectories to load (NavSim)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to load (nuScenes)")
    parser.add_argument("--min_displacement", type=float, default=0.0,
                        help="Min displacement filter")
    
    # Clustering arguments
    parser.add_argument("--vocab_size", type=int, default=256,
                        help="Number of clusters per superclass")
    parser.add_argument("--radius", type=float, default=None,
                        help="K-disk radius (None = auto-estimate)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_candidates", type=int, default=10000,
                        help="Max candidate pool size for K-disk")
    parser.add_argument("--use_contour", action="store_true",
                        help="Use bounding box contour for distance")
    parser.add_argument("--vehicle_width", type=float, default=2.0)
    parser.add_argument("--vehicle_length", type=float, default=4.8)
    
    # Output arguments
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path")
    parser.add_argument("--format", type=str, choices=["pkl", "npy"], 
                        default="pkl",
                        help="Output format")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Visualization output directory")
    
    # Logging
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    
    # Set seed
    seed_everything(args.seed)
    
    log.info("=" * 60)
    log.info("Unified Codebook Generation v2")
    log.info("=" * 60)
    
    # Load data
    log.info(f"Loading data from: {args.data_path}")
    if args.data_type == "navsim":
        loader = NavSimDataLoader(
            data_path=args.data_path,
            n_trajs=args.n_trajs,
            min_displacement=args.min_displacement,
        )
        token_banks = loader.load()
    else:  # nuscenes
        loader = NuScenesDataLoader(
            info_path=args.data_path,
            max_samples=args.max_samples,
        )
        token_banks = loader.load()
    
    # Create codebooks
    creator = UnifiedCodebookCreator(
        vocab_size=args.vocab_size,
        radius=args.radius,
        seed=args.seed,
        max_candidates=args.max_candidates,
        use_contour=args.use_contour,
        vehicle_dims=(args.vehicle_width, args.vehicle_length),
    )
    
    codebooks = creator.create(token_banks)
    
    # Save - create v2 subdirectory
    output_base = Path(args.output)
    if args.data_type == "navsim":
        # NavSim format: codebook_cache/v2/navsim_kdisk_v{vocab_size}/
        v2_output_dir = output_base.parent / "v2" / f"navsim_kdisk_v{args.vocab_size}"
    else:
        # VAD format: codebook_cache/v2/vad_kdisk_v{vocab_size}/
        v2_output_dir = output_base.parent / "v2" / f"vad_kdisk_v{args.vocab_size}"
    
    creator.save(v2_output_dir, format=args.format)
    
    # Visualize
    if args.visualize:
        vis_dir = args.vis_dir or v2_output_dir
        creator.visualize(vis_dir, data_type=args.data_type)
    
    log.info("Done!")


if __name__ == "__main__":
    main()
