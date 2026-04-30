#!/usr/bin/env python3
"""
Visualize specific DiffusionDrive-AR inference scenes given a token list.

This script is designed to rerun inference on the exact token lists evaluated
in DiffusionDrive_inference.ipynb (e.g. zero-score NC/DAC failure scenes).
Instead of filtering a CSV by conditions, you provide the token IDs directly
via --token_file or --tokens. The CSV is used only to look up baseline scores
for visualization.

Examples:
    python3 visualize_ar_bad_scenes.py --token_file tokens_nc.txt
    python3 visualize_ar_bad_scenes.py --tokens token1,token2,token3
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

warnings.filterwarnings(action="ignore")

# Environment setup
os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
os.environ["NUPLAN_MAPS_ROOT"] = "/data/navsim/dataset/maps"
os.environ["NAVSIM_EXP_ROOT"] = "/data/navsim/exp/bg"
os.environ["NAVSIM_DEVKIT_ROOT"] = "/home/byounggun/DiffusionDrive"
os.environ["OPENSCENE_DATA_ROOT"] = "/data/navsim/dataset"

from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.diffusiondrive.transfuser_agent_ar import TransfuserAgentAR
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.human_agent import HumanAgent
from navsim.common.dataclasses import SceneFilter, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_camera_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_ax, configure_bev_ax


DEFAULT_CKPT = "/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/24l0pgz4/checkpoints/last.ckpt"
DEFAULT_CSV = "/data2/byounggun/diffusiondrive_ar_output/eval_base_latest/2026.04.14.14.16.42.csv"
DEFAULT_METRIC_CACHE = "/data2/byounggun/metric_cache"
DEFAULT_DATA_DIR = "/data/navsim/dataset"
DEFAULT_OUTPUT_DIR = "/home/byounggun/DiffusionDrive/plots/ar_bad_scene_visualization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize specific AR inference scenes from a token list"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Checkpoint path")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Evaluation CSV path (for baseline score lookup)")
    parser.add_argument("--metric_cache", type=str, default=DEFAULT_METRIC_CACHE, help="Metric cache path")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="NAVSIM dataset root")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--split", type=str, default="test", help="Data split: test or trainval")
    parser.add_argument(
        "--token_file",
        type=str,
        default=None,
        help="Path to text file with one token per line",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Comma-separated list of tokens to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy, >0 = stochastic)",
    )
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> TransfuserAgentAR:
    cfg = TransfuserConfig()
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


def compute_ar_trajectory(model: TransfuserAgentAR, agent_input, temperature: float) -> Trajectory:
    device = next(model.parameters()).device
    features = {}
    for builder in model.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}

    with torch.no_grad():
        predictions = model._transfuser_model(features, temperature=temperature)

    poses = predictions["trajectory"].squeeze(0).cpu().numpy()
    return Trajectory(poses)


def plot_bev_with_trajectories(
    scene,
    ar_trajectory,
    human_trajectory,
    pdm_result,
    human_pdm_result,
    token: str,
    csv_row: pd.Series,
) -> plt.Figure:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])

    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, ar_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    title_text = (
        f"Token: {token}\n"
        f"CSV  score={csv_row.get('score', float('nan')):.4f}, NC={csv_row.get('no_at_fault_collisions', float('nan')):.1f}, "
        f"DAC={csv_row.get('drivable_area_compliance', float('nan')):.1f}, EP={csv_row.get('ego_progress', float('nan')):.3f}\n"
        f"AR   score={pdm_result.score:.4f}, NC={pdm_result.no_at_fault_collisions:.1f}, "
        f"DAC={pdm_result.drivable_area_compliance:.1f}, EP={pdm_result.ego_progress:.3f}, "
        f"TTC={pdm_result.time_to_collision_within_bound:.1f}, C={pdm_result.comfort:.1f}\n"
        f"Human score={human_pdm_result.score:.4f}, NC={human_pdm_result.no_at_fault_collisions:.1f}, "
        f"DAC={human_pdm_result.drivable_area_compliance:.1f}, EP={human_pdm_result.ego_progress:.3f}"
    )
    ax.set_title(title_text, fontsize=8, loc="left", pad=4, family="monospace")

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
    csv_row: pd.Series,
) -> plt.Figure:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]

    fig, ax = plt.subplots(3, 3, figsize=(24, 16))
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

    for row_idx, col_idx, cam, name in camera_positions:
        if cam.image is not None:
            add_camera_ax(ax[row_idx, col_idx], cam)
        else:
            ax[row_idx, col_idx].text(
                0.5,
                0.5,
                f"{name}\n(no image)",
                transform=ax[row_idx, col_idx].transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
            ax[row_idx, col_idx].set_facecolor("black")
        ax[row_idx, col_idx].set_title(name, fontsize=8)
        ax[row_idx, col_idx].set_xticks([])
        ax[row_idx, col_idx].set_yticks([])

    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_trajectory_to_bev_ax(ax[1, 1], human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax[1, 1], ar_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax[1, 1])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title("BEV", fontsize=8)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=TRAJECTORY_CONFIG["agent"]["line_color"], lw=2, label="AR Model"),
        Line2D([0], [0], color=TRAJECTORY_CONFIG["human"]["line_color"], lw=2, label="Human"),
    ]
    ax[1, 1].legend(handles=legend_elements, loc="upper right", fontsize=7)

    title_text = (
        f"Token: {token} | "
        f"CSV score={csv_row.get('score', float('nan')):.4f} | "
        f"AR score={pdm_result.score:.4f} "
        f"(NC={pdm_result.no_at_fault_collisions:.1f}, DAC={pdm_result.drivable_area_compliance:.1f}, "
        f"EP={pdm_result.ego_progress:.3f}, TTC={pdm_result.time_to_collision_within_bound:.1f}, "
        f"C={pdm_result.comfort:.1f}) | "
        f"Human score={human_pdm_result.score:.4f}"
    )
    fig.suptitle(title_text, fontsize=10, family="monospace", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.02, hspace=0.05)

    return fig


def load_scene_loader(model: TransfuserAgentAR, split: str, data_dir: str) -> SceneLoader:
    scene_filter_name = "navtrain" if split == "trainval" else "navtest"
    scene_filter_config = OmegaConf.load(
        f"navsim/planning/script/config/common/train_test_split/scene_filter/{scene_filter_name}.yaml"
    )
    scene_filter: SceneFilter = instantiate(scene_filter_config)

    if split == "trainval":
        trainval_logs = OmegaConf.load(
            "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
        )
        scene_filter.log_names = trainval_logs["val_logs"]

    navsim_log_path = os.path.join(data_dir, "navsim_logs", split)
    sensor_data_path = os.path.join(data_dir, "sensor_blobs", split)

    return SceneLoader(
        sensor_blobs_path=Path(sensor_data_path),
        data_path=Path(navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=model.get_sensor_config(),
    )


def main() -> None:
    args = parse_args()

    if args.token_file is None and args.tokens is None:
        print("Error: You must provide either --token_file or --tokens.")
        return

    # Load token list
    if args.token_file is not None:
        with open(args.token_file, "r") as f:
            tokens = [line.strip() for line in f if line.strip()]
    else:
        tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]

    if not tokens:
        print("Error: No tokens provided.")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Eval CSV (lookup): {args.csv}")
    print(f"Sampling temperature: {args.temperature}")
    print(f"Number of tokens to evaluate: {len(tokens)}")

    os.makedirs(args.output_dir, exist_ok=True)
    bev_dir = os.path.join(args.output_dir, "bev")
    combined_dir = os.path.join(args.output_dir, "combined")
    os.makedirs(bev_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    print("\n[1/4] Loading model...")
    model = load_model(args.ckpt, device)
    human_agent = HumanAgent()

    print("\n[2/4] Loading evaluation CSV for lookup...")
    if os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        print(f"CSV loaded: {len(df)} rows")
    else:
        print(f"Warning: CSV not found at {args.csv}. Baseline scores will be NaN.")
        df = pd.DataFrame()

    # Build a DataFrame that preserves the input token order
    token_order = {t: i for i, t in enumerate(tokens)}
    if not df.empty and "token" in df.columns:
        selected_df = df[df["token"].isin(tokens)].copy()
        missing_tokens = [t for t in tokens if t not in set(selected_df["token"])]
        if missing_tokens:
            print(f"Warning: {len(missing_tokens)} tokens not found in CSV: {missing_tokens}")
            missing_df = pd.DataFrame({"token": missing_tokens})
            selected_df = pd.concat([selected_df, missing_df], ignore_index=True)
    else:
        selected_df = pd.DataFrame({"token": tokens})

    selected_df = selected_df.sort_values(
        by="token", key=lambda col: col.map(token_order)
    ).reset_index(drop=True)

    print("\n[3/4] Loading scenes and metric cache...")
    scene_loader = load_scene_loader(model, args.split, args.data_dir)
    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache))

    available_tokens = set(scene_loader.tokens) & set(metric_cache_loader.tokens)
    selected_df = selected_df[selected_df["token"].isin(available_tokens)].reset_index(drop=True)
    skipped_tokens = [t for t in tokens if t not in available_tokens]
    if skipped_tokens:
        print(f"Skipped {len(skipped_tokens)} tokens missing from scene/metric cache.")
    print(f"Tokens available after filtering: {len(selected_df)}")
    if selected_df.empty:
        print("No selected tokens exist in both scene loader and metric cache.")
        return

    print("\n[4/4] Loading PDM scoring config...")
    scoring_cfg = OmegaConf.load(
        "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
    )
    simulator: PDMSimulator = instantiate(scoring_cfg.simulator)
    scorer: PDMScorer = instantiate(scoring_cfg.scorer)

    print(f"\nVisualizing {len(selected_df)} scenes...")
    results_summary = []

    for idx, row in selected_df.iterrows():
        token = row["token"]
        csv_score = row.get("score", float("nan"))
        csv_nc = row.get("no_at_fault_collisions", float("nan"))
        csv_dac = row.get("drivable_area_compliance", float("nan"))
        print(
            f"[{idx + 1}/{len(selected_df)}] {token} "
            f"(csv_score={csv_score:.4f}, NC={csv_nc:.1f}, DAC={csv_dac:.1f})"
        )

        try:
            scene = scene_loader.get_scene_from_token(token)
            metric_cache = metric_cache_loader.get_from_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)

            ar_trajectory = compute_ar_trajectory(model, agent_input, args.temperature)
            human_trajectory = human_agent.compute_trajectory(agent_input, scene)

            ar_pdm = pdm_score(
                metric_cache, ar_trajectory, simulator.proposal_sampling, simulator, scorer
            )
            human_pdm = pdm_score(
                metric_cache, human_trajectory, simulator.proposal_sampling, simulator, scorer
            )

            fig_bev = plot_bev_with_trajectories(
                scene, ar_trajectory, human_trajectory, ar_pdm, human_pdm, token, row
            )
            bev_path = os.path.join(
                bev_dir, f"{idx:03d}_{csv_score:.4f}_{token}_bev.png"
            )
            fig_bev.savefig(bev_path, dpi=150, bbox_inches="tight")
            plt.close(fig_bev)

            fig_combined = plot_combined_view(
                scene, ar_trajectory, human_trajectory, ar_pdm, human_pdm, token, row
            )
            combined_path = os.path.join(
                combined_dir, f"{idx:03d}_{csv_score:.4f}_{token}_combined.png"
            )
            fig_combined.savefig(combined_path, dpi=120, bbox_inches="tight")
            plt.close(fig_combined)

            results_summary.append(
                {
                    "idx": idx,
                    "token": token,
                    "csv_score": csv_score,
                    "csv_nc": csv_nc,
                    "csv_dac": csv_dac,
                    "csv_ep": row.get("ego_progress", float("nan")),
                    "csv_ttc": row.get("time_to_collision_within_bound", float("nan")),
                    "csv_comfort": row.get("comfort", float("nan")),
                    "ar_score": ar_pdm.score,
                    "ar_nc": ar_pdm.no_at_fault_collisions,
                    "ar_dac": ar_pdm.drivable_area_compliance,
                    "ar_ep": ar_pdm.ego_progress,
                    "ar_ttc": ar_pdm.time_to_collision_within_bound,
                    "ar_comfort": ar_pdm.comfort,
                    "human_score": human_pdm.score,
                    "human_nc": human_pdm.no_at_fault_collisions,
                    "human_dac": human_pdm.drivable_area_compliance,
                    "human_ep": human_pdm.ego_progress,
                    "bev_path": bev_path,
                    "combined_path": combined_path,
                }
            )

        except Exception as exc:
            print(f"  Failed to process {token}: {exc}")
            import traceback
            traceback.print_exc()

    if not results_summary:
        print("No visualizations were generated.")
        return

    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(args.output_dir, "visualization_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("Visualization complete")
    print(f"BEV images:      {bev_dir}")
    print(f"Combined images: {combined_dir}")
    print(f"Summary CSV:     {summary_path}")
    print(f"Saved scenes:    {len(summary_df)}")
    print(f"Mean CSV score:  {summary_df['csv_score'].mean():.4f}")
    print(f"Mean AR score:   {summary_df['ar_score'].mean():.4f}")
    print(f"Mean Human score:{summary_df['human_score'].mean():.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
