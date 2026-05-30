#!/usr/bin/env python3
"""Visualize base-vs-GRPO+ trajectory overlays for selected score-gap scenes."""

import argparse
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
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.plots import configure_ax, configure_bev_ax


DEFAULT_SELECTION = "/home/byounggun/DiffusionDrive/plots/base_vs_plus_gap_top10/selected_base_plus_gap_top10.csv"
DEFAULT_OUTPUT = "/home/byounggun/DiffusionDrive/plots/base_vs_plus_gap_top10"
DEFAULT_DATA_DIR = "/data/navsim/dataset"
DEFAULT_METRIC_CACHE = "/data2/byounggun/metric_cache"

BASE_CKPT = "/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6_waymo/checkpoints/milestone_epoch_120.ckpt"
PLUS_CKPT = "/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2/checkpoints/grpo-epoch=26.ckpt"

BASE_STYLE = {
    "line_color": "#555555",
    "line_color_alpha": 1.0,
    "line_width": 2.4,
    "line_style": "--",
    "marker": "o",
    "marker_size": 4,
    "marker_edge_color": "black",
    "zorder": 4,
}
PLUS_STYLE = {
    "line_color": "#d62728",
    "line_color_alpha": 1.0,
    "line_width": 2.4,
    "line_style": "-",
    "marker": "o",
    "marker_size": 4,
    "marker_edge_color": "black",
    "zorder": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_csv", type=str, default=DEFAULT_SELECTION)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metric_cache", type=str, default=DEFAULT_METRIC_CACHE)
    return parser.parse_args()


def build_config() -> TransfuserConfig:
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


def load_model(ckpt: str, lr: float) -> TransfuserAgentAR:
    model = TransfuserAgentAR(build_config(), lr=lr, checkpoint_path=ckpt)
    model.initialize()
    model.eval()
    return model


def plot_overlay(scene, token: str, row: pd.Series, base_traj, plus_traj, base_pdm, plus_pdm, output_path: Path) -> None:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, base_traj, BASE_STYLE)
    add_trajectory_to_bev_ax(ax, plus_traj, PLUS_STYLE)
    configure_bev_ax(ax)
    configure_ax(ax)

    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], color=BASE_STYLE["line_color"], lw=2.4, linestyle="--", label="base v6_waymo ep120"),
            Line2D([0], [0], color=PLUS_STYLE["line_color"], lw=2.4, linestyle="-", label="GRPO+ ver2 ep26"),
        ],
        loc="upper right",
        fontsize=8,
    )
    ax.set_title(
        f"{token}\n"
        f"CSV: base={row.base_score:.4f}, plus={row.plus_score:.4f}, gap={row.gap:.4f}\n"
        f"recomputed: base={base_pdm.score:.4f}, plus={plus_pdm.score:.4f}",
        fontsize=9,
        loc="left",
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "base_plus_overlay"
    trajectory_dir = output_dir / "base_plus_trajectories"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(args.selection_csv).head(10).copy()
    tokens = selection["token"].tolist()
    (output_dir / "selected_tokens.txt").write_text("\n".join(tokens) + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    print(f"Selected tokens: {len(tokens)}")

    scene_loader = load_scene_loader(tokens, args.data_dir)
    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache))
    scoring_cfg = OmegaConf.load("navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    simulator = instantiate(scoring_cfg.simulator)
    scorer = instantiate(scoring_cfg.scorer)

    print("Loading base model...")
    base_model = load_model(BASE_CKPT, lr=2e-4)
    base_trajs = {}
    base_pdms = {}
    for token in tokens:
        agent_input = scene_loader.get_agent_input_from_token(token)
        metric_cache = metric_cache_loader.get_from_token(token)
        traj = base_model.compute_trajectory(agent_input)
        base_trajs[token] = traj
        base_pdms[token] = pdm_score(metric_cache, traj, simulator.proposal_sampling, simulator, scorer)
        print(f"  base {token}: {base_pdms[token].score:.4f}")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading GRPO+ model...")
    plus_model = load_model(PLUS_CKPT, lr=1e-5)
    plus_trajs = {}
    plus_pdms = {}
    for token in tokens:
        agent_input = scene_loader.get_agent_input_from_token(token)
        metric_cache = metric_cache_loader.get_from_token(token)
        traj = plus_model.compute_trajectory(agent_input)
        plus_trajs[token] = traj
        plus_pdms[token] = pdm_score(metric_cache, traj, simulator.proposal_sampling, simulator, scorer)
        print(f"  plus {token}: {plus_pdms[token].score:.4f}")

    summary_rows = []
    for idx, row in selection.iterrows():
        token = row["token"]
        scene = scene_loader.get_scene_from_token(token)
        np.savez_compressed(
            trajectory_dir / f"{idx:02d}_{token}_base_plus.npz",
            base=base_trajs[token].poses,
            plus=plus_trajs[token].poses,
        )
        image_path = overlay_dir / f"{idx:02d}_gap_{row.gap:.4f}_{token}_base_plus.png"
        plot_overlay(scene, token, row, base_trajs[token], plus_trajs[token], base_pdms[token], plus_pdms[token], image_path)
        summary_rows.append(
            {
                "idx": idx,
                "token": token,
                "csv_base_score": row.base_score,
                "csv_plus_score": row.plus_score,
                "csv_gap": row.gap,
                "recomputed_base_score": base_pdms[token].score,
                "recomputed_plus_score": plus_pdms[token].score,
                "recomputed_gap": plus_pdms[token].score - base_pdms[token].score,
                "image_path": str(image_path),
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "base_plus_overlay_summary.csv", index=False)
    print("=" * 72)
    print(f"Saved overlays: {overlay_dir}")
    print(f"Saved trajectories: {trajectory_dir}")
    print(f"Saved summary: {output_dir / 'base_plus_overlay_summary.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
