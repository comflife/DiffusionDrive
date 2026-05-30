#!/usr/bin/env python3
"""Save inference trajectories and BEV overlays for Waymo-v6 RL improvement scenes."""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

warnings.filterwarnings(action="ignore")

os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
os.environ["NUPLAN_MAPS_ROOT"] = "/data/navsim/dataset/maps"
os.environ["NAVSIM_EXP_ROOT"] = "/data/navsim/exp/bg"
os.environ["NAVSIM_DEVKIT_ROOT"] = "/home/byounggun/DiffusionDrive"
os.environ["OPENSCENE_DATA_ROOT"] = "/data/navsim/dataset"

from navsim.agents.diffusiondrive.transfuser_agent_ar import TransfuserAgentAR
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.human_agent import HumanAgent
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.visualization.plots import configure_ax, configure_bev_ax


DEFAULT_OUTPUT_DIR = "/home/byounggun/DiffusionDrive/plots/waymo_rl_ver2_base_bad_scene_sets"
DEFAULT_TOKEN_FILE = f"{DEFAULT_OUTPUT_DIR}/selected_tokens.txt"
DEFAULT_METRICS_LONG = f"{DEFAULT_OUTPUT_DIR}/selected_5_scene_sets_long.csv"
DEFAULT_METRIC_CACHE = "/data2/byounggun/metric_cache"
DEFAULT_DATA_DIR = "/data/navsim/dataset"

MODEL_SPECS = {
    "human": {
        "color": "#2ca02c",
        "linestyle": "-",
        "ckpt": None,
    },
    "base_v6_waymo_ep120": {
        "color": "#555555",
        "linestyle": "--",
        "ckpt": "/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6_waymo/checkpoints/milestone_epoch_120.ckpt",
    },
    "drgrpo_plus_ver2_ep26": {
        "color": "#d62728",
        "linestyle": "-",
        "ckpt": "/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2/checkpoints/grpo-epoch=26.ckpt",
    },
    "drgrpo_ver2_ep26": {
        "color": "#1f77b4",
        "linestyle": "-",
        "ckpt": "/data2/byounggun/diffusiondrive_drgrpo_output_v6_waymo_epoch120_ver2/checkpoints/grpo-epoch=26.ckpt",
    },
    "drgspo_ver2_ep26": {
        "color": "#ff7f0e",
        "linestyle": "-",
        "ckpt": "/data2/byounggun/diffusiondrive_dr_gspo_output_v6_waymo_epoch120_ver2/checkpoints/grpo-epoch=26.ckpt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--token_file", type=str, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--metrics_long", type=str, default=DEFAULT_METRICS_LONG)
    parser.add_argument("--metric_cache", type=str, default=DEFAULT_METRIC_CACHE)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_waymo_v6_config() -> TransfuserConfig:
    cfg = TransfuserConfig()
    cfg.ego_vocab_size = 2048
    cfg.ego_vocab_path = "/home/byounggun/DiffusionDrive/codebook_cache/waymo_kdisk_v2048_diffusiondrive/ego.npy"
    cfg.agent_topk = 30
    cfg.temperature = 0.0
    cfg.ar_codebook_mode = "step_corners"
    cfg.ar_teacher_forcing = False
    cfg.ar_num_modes = 1
    cfg.ar_token_loss_weight = 1.0
    cfg.ar_traj_loss_weight = 8.0
    cfg.ar_heading_loss_weight = 2.0
    cfg.ar_use_residual_delta = True
    cfg.ar_use_heading_head = True
    cfg.ar_step_aware_agent = True
    cfg.ar_use_ego_cross_attn = True
    cfg.ar_use_deformable_bev = True
    cfg.ar_use_bev_pos_enc = True
    cfg.freeze_pretrained_trunk = False
    return cfg


def load_tokens(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_scene_loader(tokens: list[str], data_dir: str) -> SceneLoader:
    scene_filter_config = OmegaConf.load(
        "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    scene_filter: SceneFilter = instantiate(scene_filter_config)
    scene_filter.tokens = tokens
    return SceneLoader(
        sensor_blobs_path=Path(data_dir) / "sensor_blobs" / "test",
        data_path=Path(data_dir) / "navsim_logs" / "test",
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[3]),
    )


def trajectory_to_array(trajectory: Trajectory) -> np.ndarray:
    return np.asarray(trajectory.poses, dtype=np.float32)


def add_traj(ax: plt.Axes, trajectory: Trajectory, color: str, label: str, linestyle: str = "-") -> None:
    poses = np.concatenate([np.array([[0.0, 0.0]], dtype=np.float32), trajectory.poses[:, :2]], axis=0)
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=color,
        linewidth=2.0,
        linestyle=linestyle,
        marker="o",
        markersize=3,
        markeredgecolor="black",
        label=label,
        zorder=4,
    )


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    trajectories_dir = output_dir / "trajectories"
    bev_dir = output_dir / "bev_overlay"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    bev_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(args.token_file)
    metrics_long = pd.read_csv(args.metrics_long)
    metrics_by_token = {
        token: metrics_long[metrics_long["token"] == token].set_index("model").to_dict("index")
        for token in tokens
    }

    print(f"Using device: {device}")
    print(f"Tokens: {len(tokens)}")
    print("Loading scene loader...")
    scene_loader = load_scene_loader(tokens, args.data_dir)
    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache))

    scoring_cfg = OmegaConf.load("navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    simulator: PDMSimulator = instantiate(scoring_cfg.simulator)
    scorer: PDMScorer = instantiate(scoring_cfg.scorer)
    human_agent = HumanAgent()

    scenes = {token: scene_loader.get_scene_from_token(token) for token in tokens}
    agent_inputs = {token: scene_loader.get_agent_input_from_token(token) for token in tokens}
    metric_caches = {token: metric_cache_loader.get_from_token(token) for token in tokens}

    trajectories: dict[str, dict[str, Trajectory]] = {token: {} for token in tokens}
    recomputed_rows = []

    for token in tokens:
        human_traj = human_agent.compute_trajectory(agent_inputs[token], scenes[token])
        trajectories[token]["human"] = human_traj
        human_pdm = pdm_score(metric_caches[token], human_traj, simulator.proposal_sampling, simulator, scorer)
        recomputed_rows.append({"token": token, "model": "human", **human_pdm.__dict__})

    cfg = build_waymo_v6_config()
    for model_name, spec in MODEL_SPECS.items():
        if model_name == "human":
            continue

        ckpt = spec["ckpt"]
        print(f"Loading {model_name}: {ckpt}")
        model = TransfuserAgentAR(cfg, lr=2e-4, checkpoint_path=ckpt)
        model.initialize()
        model.eval()

        for token in tokens:
            traj = model.compute_trajectory(agent_inputs[token])
            trajectories[token][model_name] = traj
            model_pdm = pdm_score(metric_caches[token], traj, simulator.proposal_sampling, simulator, scorer)
            recomputed_rows.append({"token": token, "model": model_name, **model_pdm.__dict__})
            print(f"  {token} {model_name}: score={model_pdm.score:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for idx, token in enumerate(tokens):
        npz_path = trajectories_dir / f"{idx:02d}_{token}.npz"
        np.savez_compressed(
            npz_path,
            **{name: trajectory_to_array(traj) for name, traj in trajectories[token].items()},
        )

        scene = scenes[token]
        frame_idx = scene.scene_metadata.num_history_frames - 1
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

        for model_name, spec in MODEL_SPECS.items():
            add_traj(
                ax,
                trajectories[token][model_name],
                color=spec["color"],
                label=model_name,
                linestyle=spec["linestyle"],
            )

        configure_bev_ax(ax)
        configure_ax(ax)
        ax.legend(loc="upper right", fontsize=7)

        metric = metrics_by_token.get(token, {})
        base_score = metric.get("base_v6_waymo_ep120", {}).get("score", np.nan)
        plus_score = metric.get("drgrpo_plus_ver2_ep26", {}).get("score", np.nan)
        grpo_score = metric.get("drgrpo_ver2_ep26", {}).get("score", np.nan)
        gspo_score = metric.get("drgspo_ver2_ep26", {}).get("score", np.nan)
        ax.set_title(
            f"{token}\n"
            f"csv PDMS base={base_score:.4f}, plus={plus_score:.4f}, drgrpo={grpo_score:.4f}, drgspo={gspo_score:.4f}",
            fontsize=9,
            loc="left",
        )
        fig.tight_layout()
        fig.savefig(bev_dir / f"{idx:02d}_{token}_bev_overlay.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    recomputed_df = pd.DataFrame(recomputed_rows)
    recomputed_df.to_csv(output_dir / "recomputed_inference_metrics.csv", index=False)
    metadata = {
        "tokens": tokens,
        "models": MODEL_SPECS,
        "trajectories_dir": str(trajectories_dir),
        "bev_overlay_dir": str(bev_dir),
        "recomputed_metrics": str(output_dir / "recomputed_inference_metrics.csv"),
    }
    (output_dir / "inference_artifacts.json").write_text(json.dumps(metadata, indent=2))

    print("=" * 72)
    print(f"Saved trajectories: {trajectories_dir}")
    print(f"Saved BEV overlays: {bev_dir}")
    print(f"Saved recomputed metrics: {output_dir / 'recomputed_inference_metrics.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
