#!/usr/bin/env python3
"""Create closed-loop BEV rollout videos from saved trajectory artifacts.

The saved trajectories are relative model plans. This script runs them through
the same PDM tracker + bicycle simulator used by PDMS, then renders the
simulated ego rollout over time.
"""

import argparse
import io
import json
import os
import warnings
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

warnings.filterwarnings(action="ignore")

os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
os.environ["NUPLAN_MAPS_ROOT"] = "/data/navsim/dataset/maps"
os.environ["NAVSIM_EXP_ROOT"] = "/data/navsim/exp/bg"
os.environ["NAVSIM_DEVKIT_ROOT"] = "/home/byounggun/DiffusionDrive"
os.environ["OPENSCENE_DATA_ROOT"] = "/data/navsim/dataset"

from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.visualization.plots import configure_ax, configure_bev_ax


DEFAULT_ARTIFACT_DIR = (
    "/home/byounggun/DiffusionDrive/plots/waymo_rl_ver2_base_bad_scene_sets/"
    "current_selection_artifacts"
)
DEFAULT_DATA_DIR = "/data/navsim/dataset"
DEFAULT_METRIC_CACHE = "/data2/byounggun/metric_cache"

MODEL_STYLE = {
    "human": {"color": "#2ca02c", "label": "human", "linestyle": "-"},
    "base_v6_waymo_ep120": {"color": "#555555", "label": "base", "linestyle": "--"},
    "drgrpo_plus_ver2_ep26": {"color": "#d62728", "label": "GRPO+", "linestyle": "-"},
    "drgrpo_ver2_ep26": {"color": "#1f77b4", "label": "Dr.GRPO", "linestyle": "-"},
    "drgspo_ver2_ep26": {"color": "#ff7f0e", "label": "Dr.GSPO", "linestyle": "-"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--token_file", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metric_cache", type=str, default=DEFAULT_METRIC_CACHE)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--save_mp4", action="store_true")
    return parser.parse_args()


def resolve_token_file(artifact_dir: Path, token_file: str | None) -> Path:
    if token_file is not None:
        return Path(token_file)
    candidates = [
        artifact_dir / "selected_tokens.txt",
        artifact_dir.parent / "selected_tokens.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No selected_tokens.txt found near {artifact_dir}")


def load_tokens(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


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
        sensor_config=SensorConfig.build_no_sensors(),
    )


def find_npz_for_token(trajectory_dir: Path, token: str) -> Path:
    matches = sorted(trajectory_dir.glob(f"*_{token}.npz"))
    if not matches:
        raise FileNotFoundError(f"No trajectory npz found for token={token} under {trajectory_dir}")
    return matches[0]


def simulated_local_rollouts(npz_path: Path, metric_cache, simulator) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    model_names = [name for name in data.files if name in MODEL_STYLE]

    initial_ego_state = metric_cache.ego_state
    proposal_states = []
    kept_names = []
    for model_name in model_names:
        trajectory = Trajectory(np.asarray(data[model_name], dtype=np.float64))
        global_trajectory = transform_trajectory(trajectory, initial_ego_state)
        state_array = get_trajectory_as_array(
            global_trajectory,
            simulator.proposal_sampling,
            initial_ego_state.time_point,
        )
        proposal_states.append(state_array)
        kept_names.append(model_name)

    simulated_global = simulator.simulate_proposals(np.stack(proposal_states, axis=0), initial_ego_state)
    origin = initial_ego_state.rear_axle

    rollouts = {}
    for idx, model_name in enumerate(kept_names):
        global_se2 = simulated_global[idx, :, StateIndex.STATE_SE2]
        rollouts[model_name] = convert_absolute_to_relative_se2_array(origin, global_se2)
    return rollouts


def figure_to_pil(fig: plt.Figure, dpi: int) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    buffer.close()
    return image


def render_rollout_gif(
    scene,
    token: str,
    rollouts: dict[str, np.ndarray],
    output_gif: Path,
    fps: float,
    dpi: int,
) -> list[Image.Image]:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]
    num_frames = max(len(path) for path in rollouts.values())
    interval_s = 4.0 / max(num_frames - 1, 1)

    images: list[Image.Image] = []
    ordered_models = [name for name in MODEL_STYLE if name in rollouts]

    for t_idx in range(num_frames):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        add_configured_bev_on_ax(ax, scene.map_api, frame)

        for model_name in ordered_models:
            path = rollouts[model_name]
            upto = min(t_idx + 1, len(path))
            style = MODEL_STYLE[model_name]

            # Draw full rollout faintly, then the elapsed rollout brightly.
            ax.plot(
                path[:, 1],
                path[:, 0],
                color=style["color"],
                linewidth=1.0,
                linestyle=style["linestyle"],
                alpha=0.20,
                zorder=4,
            )
            ax.plot(
                path[:upto, 1],
                path[:upto, 0],
                color=style["color"],
                linewidth=2.2,
                linestyle=style["linestyle"],
                marker="o",
                markersize=3.0,
                markeredgecolor="black",
                label=style["label"],
                zorder=5,
            )
            ax.scatter(
                [path[upto - 1, 1]],
                [path[upto - 1, 0]],
                color=style["color"],
                edgecolor="black",
                s=55,
                zorder=6,
            )

        configure_bev_ax(ax)
        configure_ax(ax)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(
            f"{token} | PDM-simulated closed-loop rollout | t={t_idx * interval_s:.1f}s",
            fontsize=9,
            loc="left",
        )
        fig.tight_layout()
        images.append(figure_to_pil(fig, dpi=dpi))
        plt.close(fig)

    duration_ms = int(1000.0 / fps)
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return images


def try_save_mp4(images: list[Image.Image], output_mp4: Path, fps: float) -> bool:
    try:
        frames = [np.asarray(image) for image in images]
        imageio.mimsave(output_mp4, frames, fps=fps)
        return True
    except Exception as exc:
        print(f"MP4 save skipped for {output_mp4}: {exc}")
        return False


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    trajectory_dir = artifact_dir / "trajectories"
    video_dir = artifact_dir / "closed_loop_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    token_file = resolve_token_file(artifact_dir, args.token_file)
    tokens = load_tokens(token_file)

    scene_loader = load_scene_loader(tokens, args.data_dir)
    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache))
    scoring_cfg = OmegaConf.load("navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    simulator = instantiate(scoring_cfg.simulator)

    saved = []
    for idx, token in enumerate(tokens):
        npz_path = find_npz_for_token(trajectory_dir, token)
        scene = scene_loader.get_scene_from_token(token)
        metric_cache = metric_cache_loader.get_from_token(token)
        rollouts = simulated_local_rollouts(npz_path, metric_cache, simulator)

        gif_path = video_dir / f"{idx:02d}_{token}_closed_loop.gif"
        images = render_rollout_gif(scene, token, rollouts, gif_path, args.fps, args.dpi)
        item = {"token": token, "gif": str(gif_path), "npz": str(npz_path)}

        if args.save_mp4:
            mp4_path = video_dir / f"{idx:02d}_{token}_closed_loop.mp4"
            if try_save_mp4(images, mp4_path, args.fps):
                item["mp4"] = str(mp4_path)

        saved.append(item)
        print(f"[{idx + 1}/{len(tokens)}] saved {gif_path}")

    metadata = {
        "artifact_dir": str(artifact_dir),
        "token_file": str(token_file),
        "fps": args.fps,
        "videos": saved,
        "note": "Rollouts are PDM-simulated trajectories from saved model plans, not just raw predicted paths.",
    }
    (video_dir / "video_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("=" * 72)
    print(f"Saved closed-loop videos: {video_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
