#!/usr/bin/env python3
"""Visualize GT/base/GRPO+ trajectory gaps with front camera context."""

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
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
from navsim.common.dataclasses import Camera, Scene, SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.visualization.camera import add_camera_ax
from navsim.visualization.plots import configure_ax, configure_bev_ax


DEFAULT_CANDIDATE_CSV = (
    "/home/byounggun/DiffusionDrive/plots/waymo_rl_ver2_base_bad_scene_sets/"
    "top50_current_base_bad_rl_good_candidates.csv"
)
DEFAULT_OUTPUT_DIR = "/home/byounggun/DiffusionDrive/plots/gt_base_plus_camera_gap_top10"
DEFAULT_DATA_DIR = "/data/navsim/dataset"
DEFAULT_METRIC_CACHE = "/data2/byounggun/metric_cache"

BASE_CKPT = (
    "/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6_waymo/"
    "checkpoints/milestone_epoch_120.ckpt"
)
PLUS_CKPT = (
    "/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2/"
    "checkpoints/grpo-epoch=26.ckpt"
)

TRAJECTORY_STYLES: dict[str, dict[str, Any]] = {
    "GT": {
        "color": "#1f8f4d",
        "linestyle": "-",
        "linewidth": 3.0,
        "marker": "o",
        "markersize": 5,
        "zorder": 8,
    },
    "Base": {
        "color": "#20242a",
        "linestyle": "--",
        "linewidth": 3.3,
        "marker": "o",
        "markersize": 5,
        "zorder": 9,
    },
    "GRPO+": {
        "color": "#d62728",
        "linestyle": "-",
        "linewidth": 3.3,
        "marker": "o",
        "markersize": 5,
        "zorder": 10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_csv", type=str, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metric_cache", type=str, default=DEFAULT_METRIC_CACHE)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--candidate_limit", type=int, default=50)
    parser.add_argument("--num_scenes", type=int, default=10)
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
    model = TransfuserAgentAR(build_waymo_v6_config(), lr=lr, checkpoint_path=ckpt)
    model.initialize()
    model.eval()
    return model


def normalize_candidates(path: str, candidate_limit: int) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if {"current_base__score", "drgrpo_plus_ver2_ep26__score"}.issubset(df.columns):
        df["csv_base_score"] = df["current_base__score"]
        df["csv_plus_score"] = df["drgrpo_plus_ver2_ep26__score"]
    elif {"base_score", "plus_score"}.issubset(df.columns):
        df["csv_base_score"] = df["base_score"]
        df["csv_plus_score"] = df["plus_score"]
    else:
        raise ValueError(
            "Candidate CSV must contain either current_base__score/drgrpo_plus_ver2_ep26__score "
            "or base_score/plus_score columns."
        )

    if "gap" in df.columns:
        df["csv_gap"] = df["gap"]
    else:
        df["csv_gap"] = df["csv_plus_score"] - df["csv_base_score"]

    df = df.sort_values(
        ["csv_gap", "csv_plus_score", "csv_base_score"],
        ascending=[False, False, True],
    )
    return df.head(candidate_limit).reset_index(drop=True)


def pdm_to_prefixed_dict(prefix: str, pdm_result: Any) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in vars(pdm_result).items()
        if isinstance(value, (int, float, np.integer, np.floating, bool))
    }


def trajectory_distance(a: Trajectory, b: Trajectory) -> dict[str, float]:
    a_xy = np.asarray(a.poses[:, :2], dtype=np.float32)
    b_xy = np.asarray(b.poses[:, :2], dtype=np.float32)
    distances = np.linalg.norm(a_xy - b_xy, axis=1)
    return {
        "path_mean_l2": float(np.mean(distances)),
        "path_max_l2": float(np.max(distances)),
        "path_final_l2": float(distances[-1]),
    }


def trajectory_to_array(trajectory: Trajectory) -> np.ndarray:
    return np.asarray(trajectory.poses, dtype=np.float32)


def add_trajectory(ax: plt.Axes, trajectory: Trajectory, label: str) -> None:
    style = TRAJECTORY_STYLES[label]
    poses = np.concatenate(
        [np.array([[0.0, 0.0]], dtype=np.float32), np.asarray(trajectory.poses[:, :2], dtype=np.float32)],
        axis=0,
    )
    line = ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=style["color"],
        linewidth=style["linewidth"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=style["markersize"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=label,
        zorder=style["zorder"],
    )[0]
    line.set_path_effects(
        [
            pe.Stroke(linewidth=style["linewidth"] + 2.2, foreground="white", alpha=0.92),
            pe.Normal(),
        ]
    )
    ax.scatter(
        poses[-1, 1],
        poses[-1, 0],
        color=style["color"],
        marker="X",
        s=90,
        edgecolor="white",
        linewidth=1.0,
        zorder=style["zorder"] + 1,
    )


def add_camera_panel(ax: plt.Axes, camera: Camera, title: str) -> None:
    if camera.image is None:
        ax.set_facecolor("black")
        ax.text(0.5, 0.5, "no image", transform=ax.transAxes, ha="center", va="center", color="white")
    else:
        add_camera_ax(ax, camera)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")


def plot_scene(
    scene: Scene,
    token: str,
    index: int,
    human_traj: Trajectory,
    base_traj: Trajectory,
    plus_traj: Trajectory,
    human_pdm: Any,
    base_pdm: Any,
    plus_pdm: Any,
    path_stats: dict[str, float],
    output_path: Path,
) -> None:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]

    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.45], hspace=0.08, wspace=0.02)
    camera_axes = [fig.add_subplot(gs[0, col]) for col in range(3)]
    bev_ax = fig.add_subplot(gs[1, :])

    add_camera_panel(camera_axes[0], frame.cameras.cam_l0, "Front Left (CAM_L0)")
    add_camera_panel(camera_axes[1], frame.cameras.cam_f0, "Front (CAM_F0)")
    add_camera_panel(camera_axes[2], frame.cameras.cam_r0, "Front Right (CAM_R0)")

    add_configured_bev_on_ax(bev_ax, scene.map_api, frame)
    add_trajectory(bev_ax, human_traj, "GT")
    add_trajectory(bev_ax, base_traj, "Base")
    add_trajectory(bev_ax, plus_traj, "GRPO+")
    configure_bev_ax(bev_ax)
    configure_ax(bev_ax)
    bev_ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    bev_ax.set_title("BEV trajectory overlay", fontsize=11, loc="left", pad=6)

    fig.suptitle(
        f"{index:02d} | scene_token={token} | log={scene.scene_metadata.log_name} | map={scene.scene_metadata.map_name}\n"
        f"PDMS: GT={human_pdm.score:.4f}, Base={base_pdm.score:.4f}, GRPO+={plus_pdm.score:.4f}, "
        f"gap={plus_pdm.score - base_pdm.score:.4f} | "
        f"path L2 mean/max/final={path_stats['path_mean_l2']:.2f}/"
        f"{path_stats['path_max_l2']:.2f}/{path_stats['path_final_l2']:.2f} m",
        fontsize=10,
        family="monospace",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = Path(args.output_dir)
    composite_dir = output_dir / "composite"
    trajectory_dir = output_dir / "trajectories"
    composite_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    candidates = normalize_candidates(args.candidate_csv, args.candidate_limit)
    tokens = candidates["token"].tolist()
    print(f"Using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    print(f"Candidate scenes: {len(tokens)}")

    scene_loader = load_scene_loader(tokens, args.data_dir)
    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache))
    scoring_cfg = OmegaConf.load("navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    simulator = instantiate(scoring_cfg.simulator)
    scorer = instantiate(scoring_cfg.scorer)

    trajectories: dict[str, dict[str, Trajectory]] = {token: {} for token in tokens}
    pdm_results: dict[str, dict[str, Any]] = {token: {} for token in tokens}

    print("Loading base model...")
    base_model = load_model(BASE_CKPT, lr=2e-4)
    for token in tokens:
        agent_input = scene_loader.get_agent_input_from_token(token)
        metric_cache = metric_cache_loader.get_from_token(token)
        trajectory = base_model.compute_trajectory(agent_input)
        trajectories[token]["base"] = trajectory
        pdm_results[token]["base"] = pdm_score(
            metric_cache, trajectory, simulator.proposal_sampling, simulator, scorer
        )
        print(f"  base {token}: {pdm_results[token]['base'].score:.4f}")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading GRPO+ model...")
    plus_model = load_model(PLUS_CKPT, lr=1e-5)
    rescore_rows: list[dict[str, Any]] = []
    for row_idx, row in candidates.iterrows():
        token = row["token"]
        agent_input = scene_loader.get_agent_input_from_token(token)
        metric_cache = metric_cache_loader.get_from_token(token)
        trajectory = plus_model.compute_trajectory(agent_input)
        trajectories[token]["plus"] = trajectory
        pdm_results[token]["plus"] = pdm_score(
            metric_cache, trajectory, simulator.proposal_sampling, simulator, scorer
        )
        path_stats = trajectory_distance(trajectories[token]["base"], trajectories[token]["plus"])
        rescore_rows.append(
            {
                "candidate_rank": row_idx,
                "token": token,
                "csv_base_score": row["csv_base_score"],
                "csv_plus_score": row["csv_plus_score"],
                "csv_gap": row["csv_gap"],
                "recomputed_base_score": pdm_results[token]["base"].score,
                "recomputed_plus_score": pdm_results[token]["plus"].score,
                "recomputed_gap": pdm_results[token]["plus"].score - pdm_results[token]["base"].score,
                **path_stats,
                **pdm_to_prefixed_dict("base", pdm_results[token]["base"]),
                **pdm_to_prefixed_dict("plus", pdm_results[token]["plus"]),
            }
        )
        print(
            f"  plus {token}: {pdm_results[token]['plus'].score:.4f}, "
            f"gap={rescore_rows[-1]['recomputed_gap']:.4f}, path_max={path_stats['path_max_l2']:.2f}m"
        )
    del plus_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rescore_df = pd.DataFrame(rescore_rows).sort_values(
        ["recomputed_gap", "path_max_l2", "path_mean_l2", "path_final_l2"],
        ascending=[False, False, False, False],
    )
    rescore_df.to_csv(output_dir / "candidate_rescore_with_path.csv", index=False)
    selected = rescore_df.head(args.num_scenes).reset_index(drop=True)
    selected.to_csv(output_dir / "selected_gt_base_plus_gap_top10.csv", index=False)
    (output_dir / "selected_tokens.txt").write_text("\n".join(selected["token"].tolist()) + "\n")

    human_agent = HumanAgent()
    summary_rows: list[dict[str, Any]] = []
    print("Saving selected composites...")
    for output_idx, row in selected.iterrows():
        token = row["token"]
        scene = scene_loader.get_scene_from_token(token)
        agent_input = scene_loader.get_agent_input_from_token(token)
        metric_cache = metric_cache_loader.get_from_token(token)
        human_traj = human_agent.compute_trajectory(agent_input, scene)
        human_pdm = pdm_score(metric_cache, human_traj, simulator.proposal_sampling, simulator, scorer)
        trajectories[token]["human"] = human_traj
        pdm_results[token]["human"] = human_pdm

        np.savez_compressed(
            trajectory_dir / f"{output_idx:02d}_{token}_gt_base_plus.npz",
            gt=trajectory_to_array(human_traj),
            base=trajectory_to_array(trajectories[token]["base"]),
            grpo_plus=trajectory_to_array(trajectories[token]["plus"]),
        )

        image_path = (
            composite_dir
            / f"{output_idx:02d}_gap_{row.recomputed_gap:.4f}_path_{row.path_max_l2:.2f}_{token}_gt_base_plus_cam.png"
        )
        path_stats = {
            "path_mean_l2": row.path_mean_l2,
            "path_max_l2": row.path_max_l2,
            "path_final_l2": row.path_final_l2,
        }
        plot_scene(
            scene=scene,
            token=token,
            index=output_idx,
            human_traj=human_traj,
            base_traj=trajectories[token]["base"],
            plus_traj=trajectories[token]["plus"],
            human_pdm=human_pdm,
            base_pdm=pdm_results[token]["base"],
            plus_pdm=pdm_results[token]["plus"],
            path_stats=path_stats,
            output_path=image_path,
        )
        summary_rows.append(
            {
                "idx": output_idx,
                "token": token,
                "scene_log": scene.scene_metadata.log_name,
                "map_name": scene.scene_metadata.map_name,
                "human_score": human_pdm.score,
                "base_score": pdm_results[token]["base"].score,
                "grpo_plus_score": pdm_results[token]["plus"].score,
                "gap": pdm_results[token]["plus"].score - pdm_results[token]["base"].score,
                "path_mean_l2": row.path_mean_l2,
                "path_max_l2": row.path_max_l2,
                "path_final_l2": row.path_final_l2,
                "image_path": str(image_path),
            }
        )
        print(f"  saved {image_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "gt_base_plus_camera_overlay_summary.csv", index=False)

    print("=" * 72)
    print(f"Saved composites: {composite_dir}")
    print(f"Saved trajectories: {trajectory_dir}")
    print(f"Saved selected CSV: {output_dir / 'selected_gt_base_plus_gap_top10.csv'}")
    print(f"Saved summary: {output_dir / 'gt_base_plus_camera_overlay_summary.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
