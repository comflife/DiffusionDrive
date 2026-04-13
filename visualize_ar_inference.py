#!/usr/bin/env python3
"""
Visualization script for DiffusionDrive-AR model evaluation.

Loads the AR model, evaluates on selected scenes from NAVSIM,
and saves BEV + camera visualizations with PDM scores.

Usage:
    python visualize_ar_inference.py [--gpu 0] [--num_scenes 20] [--mode worst]
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings(action="ignore")

# ── Environment setup ────────────────────────────────────────────────
os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
os.environ["NUPLAN_MAPS_ROOT"] = "/data/navsim/dataset/maps"
os.environ["NAVSIM_EXP_ROOT"] = "/data/navsim/exp/bg"
os.environ["NAVSIM_DEVKIT_ROOT"] = "/home/byounggun/DiffusionDrive"
os.environ["OPENSCENE_DATA_ROOT"] = "/data/navsim/dataset"

# ── Imports (after env setup) ────────────────────────────────────────
from omegaconf import OmegaConf
from hydra.utils import instantiate

from navsim.agents.diffusiondrive.transfuser_agent_ar import TransfuserAgentAR
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.human_agent import HumanAgent
from navsim.evaluate.pdm_score import pdm_score
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.training.dataset import Dataset
from navsim.visualization.plots import (
    plot_bev_frame,
    configure_bev_ax,
    configure_ax,
)
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_camera_ax


# ── Default Paths ────────────────────────────────────────────────────
DEFAULT_CKPT = "/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/owb1ii1t/checkpoints/last.ckpt"
DEFAULT_CSV = "/data2/byounggun/diffusiondrive_ar_output/eval_results/2026.04.10.21.32.51.csv"
DEFAULT_METRIC_CACHE = "/data2/byounggun/metric_cache"
DEFAULT_DATA_DIR = "/data/navsim/dataset"
DEFAULT_OUTPUT_DIR = "/home/byounggun/DiffusionDrive/plots/ar_visualization"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DiffusionDrive-AR inference results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: 0)")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Checkpoint path")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Eval results CSV path")
    parser.add_argument("--metric_cache", type=str, default=DEFAULT_METRIC_CACHE, help="Metric cache path")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="NAVSIM data directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for images")
    parser.add_argument("--num_scenes", type=int, default=20, help="Number of scenes to visualize")
    parser.add_argument(
        "--mode", type=str, default="mixed",
        choices=["worst", "best", "random", "mixed", "zero_nc", "zero_dac"],
        help="Scene selection mode: worst/best/random/mixed/zero_nc/zero_dac"
    )
    parser.add_argument("--split", type=str, default="test", help="Data split: test or trainval")
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> TransfuserAgentAR:
    """Load TransfuserAgentAR model from checkpoint."""
    cfg = TransfuserConfig()
    # Set AR-specific config from the agent YAML
    cfg.ego_vocab_size = 512
    cfg.ego_vocab_path = "/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy"
    cfg.agent_topk = 8
    cfg.agent_context_dim = 256
    cfg.latent = False

    model = TransfuserAgentAR(cfg, lr=6e-4, checkpoint_path=ckpt_path)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def select_tokens(df: pd.DataFrame, mode: str, num_scenes: int) -> List[str]:
    """Select tokens based on visualization mode."""
    if mode == "worst":
        selected = df.sort_values("score", ascending=True).head(num_scenes)
    elif mode == "best":
        selected = df.sort_values("score", ascending=False).head(num_scenes)
    elif mode == "random":
        selected = df.sample(n=min(num_scenes, len(df)), random_state=42)
    elif mode == "zero_nc":
        zero_nc = df[df["no_at_fault_collisions"] == 0]
        selected = zero_nc.head(num_scenes)
        if len(selected) < num_scenes:
            print(f"Only {len(selected)} scenes with NC=0 (requested {num_scenes})")
    elif mode == "zero_dac":
        zero_dac = df[df["drivable_area_compliance"] == 0]
        selected = zero_dac.head(num_scenes)
        if len(selected) < num_scenes:
            print(f"Only {len(selected)} scenes with DAC=0 (requested {num_scenes})")
    elif mode == "mixed":
        # Mix of worst, best, and random scenes
        n_worst = num_scenes // 3
        n_best = num_scenes // 3
        n_random = num_scenes - n_worst - n_best

        worst = df.sort_values("score", ascending=True).head(n_worst)
        best = df.sort_values("score", ascending=False).head(n_best)
        # Random from middle range (exclude already selected)
        remaining = df[~df["token"].isin(worst["token"].tolist() + best["token"].tolist())]
        random_sel = remaining.sample(n=min(n_random, len(remaining)), random_state=42)

        selected = pd.concat([worst, random_sel, best])
    else:
        selected = df.head(num_scenes)

    tokens = selected["token"].tolist()
    print(f"Selected {len(tokens)} tokens (mode={mode})")
    return tokens


def plot_bev_with_trajectories(
    scene,
    ar_trajectory,
    human_trajectory,
    pdm_result,
    human_pdm_result,
    token: str,
) -> plt.Figure:
    """
    Plot BEV with AR model and human trajectories, including PDM score text.
    """
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])

    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, ar_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    # Add score text
    score_text = (
        f"Token: {token}\n"
        f"AR  PDMS: {pdm_result.score:.4f}  "
        f"(NC={pdm_result.no_at_fault_collisions:.1f}, "
        f"DAC={pdm_result.drivable_area_compliance:.1f}, "
        f"EP={pdm_result.ego_progress:.3f}, "
        f"TTC={pdm_result.time_to_collision_within_bound:.1f}, "
        f"C={pdm_result.comfort:.1f})\n"
        f"Human PDMS: {human_pdm_result.score:.4f}  "
        f"(NC={human_pdm_result.no_at_fault_collisions:.1f}, "
        f"DAC={human_pdm_result.drivable_area_compliance:.1f}, "
        f"EP={human_pdm_result.ego_progress:.3f}, "
        f"TTC={human_pdm_result.time_to_collision_within_bound:.1f}, "
        f"C={human_pdm_result.comfort:.1f})"
    )
    ax.set_title(score_text, fontsize=8, loc="left", pad=4, family="monospace")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=TRAJECTORY_CONFIG["agent"]["line_color"], lw=2, label="AR Model"),
        Line2D([0], [0], color=TRAJECTORY_CONFIG["human"]["line_color"], lw=2, label="Human"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def plot_combined_view(
    scene,
    ar_trajectory,
    human_trajectory,
    pdm_result,
    human_pdm_result,
    token: str,
) -> plt.Figure:
    """
    Plot combined view: BEV (center) + 8 cameras in 3x3 grid with PDM score.
    """
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]

    fig, ax = plt.subplots(3, 3, figsize=(24, 16))

    # Camera layout (same as NAVSIM):
    # [cam_l0] [cam_f0] [cam_r0]
    # [cam_l1] [  BEV ] [cam_r1]
    # [cam_l2] [cam_b0] [cam_r2]
    camera_positions = [
        (0, 0, frame.cameras.cam_l0, "CAM_L0"),
        (0, 1, frame.cameras.cam_f0, "CAM_F0"),
        (0, 2, frame.cameras.cam_r0, "CAM_R0"),
        (1, 0, frame.cameras.cam_l1, "CAM_L1"),
        (1, 2, frame.cameras.cam_r1, "CAM_R1"),
        (2, 0, frame.cameras.cam_l2, "CAM_L2"),
        (2, 1, frame.cameras.cam_b0, "CAM_B0"),
        (2, 2, frame.cameras.cam_r2, "CAM_R2"),
    ]

    for r, c, cam, name in camera_positions:
        if cam.image is not None:
            add_camera_ax(ax[r, c], cam)
        else:
            ax[r, c].text(0.5, 0.5, f"{name}\n(no image)", transform=ax[r, c].transAxes,
                          ha="center", va="center", fontsize=10, color="gray")
            ax[r, c].set_facecolor("black")
        ax[r, c].set_title(name, fontsize=8)
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])

    # BEV in center (1, 1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_trajectory_to_bev_ax(ax[1, 1], human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax[1, 1], ar_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax[1, 1])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title("BEV", fontsize=8)

    # Legend on BEV
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=TRAJECTORY_CONFIG["agent"]["line_color"], lw=2, label="AR Model"),
        Line2D([0], [0], color=TRAJECTORY_CONFIG["human"]["line_color"], lw=2, label="Human"),
    ]
    ax[1, 1].legend(handles=legend_elements, loc="upper right", fontsize=7)

    # Overall title with score
    title_text = (
        f"Token: {token}  |  "
        f"AR PDMS: {pdm_result.score:.4f} "
        f"(NC={pdm_result.no_at_fault_collisions:.1f}, DAC={pdm_result.drivable_area_compliance:.1f}, "
        f"EP={pdm_result.ego_progress:.3f}, TTC={pdm_result.time_to_collision_within_bound:.1f}, "
        f"C={pdm_result.comfort:.1f})  |  "
        f"Human PDMS: {human_pdm_result.score:.4f}"
    )
    fig.suptitle(title_text, fontsize=10, family="monospace", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.02, hspace=0.05)

    return fig


def main():
    args = parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    bev_dir = os.path.join(args.output_dir, "bev")
    combined_dir = os.path.join(args.output_dir, "combined")
    os.makedirs(bev_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # ── Load model ───────────────────────────────────────────────────
    print("\n[1/5] Loading AR model...")
    model = load_model(args.ckpt, device)
    human_agent = HumanAgent()

    # ── Load eval CSV ────────────────────────────────────────────────
    print("\n[2/5] Loading eval results CSV...")
    df = pd.read_csv(args.csv)
    print(f"  Total scenes in CSV: {len(df)}")
    print(f"  Mean PDMS: {df['score'].mean():.4f}")
    print(f"  Scenes with score=0: {len(df[df['score'] == 0])}")

    # Print summary stats
    print(f"\n  Score distribution:")
    print(f"    Min:  {df['score'].min():.4f}")
    print(f"    25%:  {df['score'].quantile(0.25):.4f}")
    print(f"    50%:  {df['score'].quantile(0.50):.4f}")
    print(f"    75%:  {df['score'].quantile(0.75):.4f}")
    print(f"    Max:  {df['score'].max():.4f}")

    # ── Select tokens ────────────────────────────────────────────────
    print(f"\n[3/5] Selecting scenes (mode={args.mode})...")
    tokens_to_evaluate = select_tokens(df, args.mode, args.num_scenes)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[4/5] Loading scene data and metric cache...")
    split = args.split
    scene_filter_name = "navtrain" if split == "trainval" else "navtest"

    scene_filter_config = OmegaConf.load(
        f"navsim/planning/script/config/common/train_test_split/scene_filter/{scene_filter_name}.yaml"
    )
    val_scene_filter: SceneFilter = instantiate(scene_filter_config)

    if split == "trainval":
        trainval_logs = OmegaConf.load(
            "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
        )
        val_scene_filter.log_names = trainval_logs["val_logs"]

    navsim_log_path = os.path.join(args.data_dir, "navsim_logs", split)
    sensor_data_path = os.path.join(args.data_dir, "sensor_blobs", split)

    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache))

    val_scene_loader = SceneLoader(
        sensor_blobs_path=Path(sensor_data_path),
        data_path=Path(navsim_log_path),
        scene_filter=val_scene_filter,
        sensor_config=model.get_sensor_config(),
    )

    # Filter to tokens that exist in both scene_loader and metric_cache
    available_tokens = set(val_scene_loader.tokens) & set(metric_cache_loader.tokens)
    tokens_to_evaluate = [t for t in tokens_to_evaluate if t in available_tokens]
    print(f"  Tokens available after filtering: {len(tokens_to_evaluate)}")

    if len(tokens_to_evaluate) == 0:
        print("ERROR: No valid tokens found. Check paths and data.")
        return

    # ── Load scoring config ──────────────────────────────────────────
    scoring_cfg = OmegaConf.load(
        "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
    )
    simulator: PDMSimulator = instantiate(scoring_cfg.simulator)
    scorer: PDMScorer = instantiate(scoring_cfg.scorer)

    # ── Evaluate and visualize ───────────────────────────────────────
    print(f"\n[5/5] Evaluating and visualizing {len(tokens_to_evaluate)} scenes...")
    results_summary = []

    for idx, token in enumerate(tokens_to_evaluate):
        print(f"\n  [{idx+1}/{len(tokens_to_evaluate)}] Token: {token}")

        try:
            # Load scene and metric cache
            scene = val_scene_loader.get_scene_from_token(token)
            metric_cache = metric_cache_loader.get_from_token(token)
            agent_input = val_scene_loader.get_agent_input_from_token(token)

            # Run AR model inference (on GPU)
            with torch.no_grad():
                ar_trajectory = model.compute_trajectory(agent_input)

            # Get human trajectory
            human_trajectory = human_agent.compute_trajectory(agent_input, scene)

            # Compute PDM scores
            ar_pdm = pdm_score(metric_cache, ar_trajectory, simulator.proposal_sampling, simulator, scorer)
            human_pdm = pdm_score(metric_cache, human_trajectory, simulator.proposal_sampling, simulator, scorer)

            print(f"    AR   PDMS: {ar_pdm.score:.4f} (NC={ar_pdm.no_at_fault_collisions:.1f}, "
                  f"DAC={ar_pdm.drivable_area_compliance:.1f}, EP={ar_pdm.ego_progress:.3f})")
            print(f"    Human PDMS: {human_pdm.score:.4f} (NC={human_pdm.no_at_fault_collisions:.1f}, "
                  f"DAC={human_pdm.drivable_area_compliance:.1f}, EP={human_pdm.ego_progress:.3f})")

            # ── Save BEV plot ────────────────────────────────────────
            fig_bev = plot_bev_with_trajectories(
                scene, ar_trajectory, human_trajectory, ar_pdm, human_pdm, token
            )
            bev_path = os.path.join(bev_dir, f"{idx:03d}_{token}_bev.png")
            fig_bev.savefig(bev_path, dpi=150, bbox_inches="tight")
            plt.close(fig_bev)

            # ── Save combined view (BEV + cameras) ───────────────────
            fig_combined = plot_combined_view(
                scene, ar_trajectory, human_trajectory, ar_pdm, human_pdm, token
            )
            combined_path = os.path.join(combined_dir, f"{idx:03d}_{token}_combined.png")
            fig_combined.savefig(combined_path, dpi=120, bbox_inches="tight")
            plt.close(fig_combined)

            results_summary.append({
                "idx": idx,
                "token": token,
                "ar_score": ar_pdm.score,
                "ar_nc": ar_pdm.no_at_fault_collisions,
                "ar_dac": ar_pdm.drivable_area_compliance,
                "ar_ep": ar_pdm.ego_progress,
                "ar_ttc": ar_pdm.time_to_collision_within_bound,
                "ar_comfort": ar_pdm.comfort,
                "human_score": human_pdm.score,
            })

            print(f"    Saved: {bev_path}")

        except Exception as e:
            print(f"    ERROR processing token {token}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Save summary CSV ─────────────────────────────────────────────
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_path = os.path.join(args.output_dir, "visualization_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"Visualization complete!")
        print(f"  BEV images:      {bev_dir}")
        print(f"  Combined images:  {combined_dir}")
        print(f"  Summary CSV:     {summary_path}")
        print(f"  Total scenes:    {len(results_summary)}")
        print(f"  Mean AR PDMS:    {summary_df['ar_score'].mean():.4f}")
        print(f"  Mean Human PDMS: {summary_df['human_score'].mean():.4f}")
        print(f"{'='*60}")
    else:
        print("\nNo results were generated. Check for errors above.")


if __name__ == "__main__":
    main()
