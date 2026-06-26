"""
DataModule for GRPO training with PDM metric cache.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, List, Optional
import lzma
import pickle
import json
from omegaconf import OmegaConf

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder


def _frame_has_sensor_blobs(
    scene_dict_list: List[Dict],
    sensor_blobs_path: Path,
    num_history_frames: int,
    sensor_config: SensorConfig,
) -> bool:
    """Return True if all sensor blobs required by sensor_config exist on disk."""
    for frame_idx in range(num_history_frames):
        sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
        frame = scene_dict_list[frame_idx]
        for camera_name, camera_spec in frame["cams"].items():
            if camera_name.lower() in sensor_names:
                image_path = sensor_blobs_path / camera_spec["data_path"]
                if not image_path.is_file():
                    return False
        if "lidar_pc" in sensor_names:
            lidar_path = sensor_blobs_path / frame["lidar_path"]
            if not lidar_path.is_file():
                return False
    return True


def _pick_loadable_frame_token(
    scene_loader: SceneLoader,
    candidates: List[str],
) -> Optional[str]:
    """Pick the first cache-backed frame token whose sensor blobs exist."""
    num_history = scene_loader._scene_filter.num_history_frames
    sensor_blobs_path = scene_loader._sensor_blobs_path
    sensor_config = scene_loader._sensor_config

    for frame_token in sorted(candidates):
        scene_dict_list = scene_loader.scene_frames_dicts[frame_token]
        if _frame_has_sensor_blobs(
            scene_dict_list,
            sensor_blobs_path,
            num_history,
            sensor_config,
        ):
            return frame_token
    return None


def _pick_fallback_frame_token(candidates: List[str]) -> Optional[str]:
    """Pick a canonical cache-backed frame when no sensor blobs are on disk."""
    if not candidates:
        return None
    sorted_candidates = sorted(candidates)
    return sorted_candidates[len(sorted_candidates) // 2]


def _resolve_training_tokens(
    scene_loader: SceneLoader,
    metric_cache_loader: MetricCacheLoader,
    tokens: Optional[List[str]] = None,
) -> List[str]:
    """Map requested scene/frame tokens to cache-backed frame tokens.

    Assignment exports use ``scene_token``, while SceneLoader / metric cache
    keys use the anchor frame ``token``. Prefer sliding-window frames whose
    sensor blobs exist; otherwise fall back to a cache-backed frame so RL can
    still run (samples without blobs are skipped lazily in ``__getitem__``).
    """
    scene_token_keys = set(scene_loader.tokens)
    cache_token_keys = set(metric_cache_loader.tokens)
    available = scene_token_keys & cache_token_keys
    if tokens is None:
        resolved = []
        with_sensors = 0
        for frame_token in sorted(available):
            picked = _pick_loadable_frame_token(scene_loader, [frame_token])
            if picked is None:
                picked = frame_token
            else:
                with_sensors += 1
            resolved.append(picked)
        if resolved and with_sensors < len(resolved):
            print(
                f"Token resolution: {with_sensors}/{len(resolved)} cache tokens "
                f"have verified sensor blobs under {scene_loader._sensor_blobs_path}"
            )
        return resolved

    num_history = scene_loader._scene_filter.num_history_frames
    by_scene_token: Dict[str, List[str]] = {}
    for frame_token in available:
        frame_list = scene_loader.scene_frames_dicts[frame_token]
        anchor = frame_list[num_history - 1]
        scene_token = anchor.get("scene_token")
        if scene_token:
            by_scene_token.setdefault(scene_token, []).append(frame_token)

    resolved: List[str] = []
    seen: set = set()
    with_sensors = 0
    fallback_without_sensors = 0
    skipped_unresolved = 0
    for requested in tokens:
        candidates: List[str] = []
        if requested in available:
            candidates = [requested]
        elif requested in by_scene_token:
            candidates = by_scene_token[requested]

        frame_token = _pick_loadable_frame_token(scene_loader, candidates)
        if frame_token is not None:
            with_sensors += 1
        elif candidates:
            frame_token = _pick_fallback_frame_token(candidates)
            fallback_without_sensors += 1
        else:
            skipped_unresolved += 1
            continue

        if frame_token not in seen:
            resolved.append(frame_token)
            seen.add(frame_token)

    if fallback_without_sensors or skipped_unresolved:
        print(
            f"Token resolution: {with_sensors} with verified sensors, "
            f"{fallback_without_sensors} cache-only fallback "
            f"(missing blobs under {scene_loader._sensor_blobs_path}), "
            f"{skipped_unresolved} not in scene/cache intersection"
        )

    return resolved


def _token_has_sensor_blobs(scene_loader: SceneLoader, token: str) -> bool:
    scene_dict_list = scene_loader.scene_frames_dicts[token]
    return _frame_has_sensor_blobs(
        scene_dict_list,
        scene_loader._sensor_blobs_path,
        scene_loader._scene_filter.num_history_frames,
        scene_loader._sensor_config,
    )


def _filter_loadable_tokens(scene_loader: SceneLoader, tokens: List[str]) -> List[str]:
    return [token for token in tokens if _token_has_sensor_blobs(scene_loader, token)]


class GRPOEpisodeDataset(Dataset):
    """
    Dataset for GRPO episodes.
    
    Each sample contains:
    - features: Model input features
    - targets: Ground truth targets (optional, for logging)
    - metric_cache: PDM metric cache for reward computation
    """
    
    def __init__(
        self,
        scene_loader: SceneLoader,
        metric_cache_loader: MetricCacheLoader,
        feature_builders: List,
        target_builders: List,
        tokens: Optional[List[str]] = None,
    ):
        self.scene_loader = scene_loader
        self.metric_cache_loader = metric_cache_loader
        self.feature_builders = feature_builders
        self.target_builders = target_builders
        
        # Debug info
        scene_tokens = set(scene_loader.tokens)
        cache_tokens = set(metric_cache_loader.tokens)
        print(f"SceneLoader: {len(scene_tokens)} tokens")
        print(f"MetricCacheLoader: {len(cache_tokens)} tokens")

        if tokens is None:
            resolved = _resolve_training_tokens(scene_loader, metric_cache_loader, None)
        else:
            print(f"Requested: {len(tokens)} tokens")
            resolved = _resolve_training_tokens(scene_loader, metric_cache_loader, list(tokens))

        # Keep only tokens whose sensor blobs exist on disk so __getitem__ never
        # has to skip/recurse at training time (trainval blobs are incomplete).
        self.tokens = _filter_loadable_tokens(scene_loader, resolved)
        dropped = len(resolved) - len(self.tokens)
        if dropped:
            print(
                f"Dropped {dropped} resolved tokens without sensor blobs under "
                f"{scene_loader._sensor_blobs_path}"
            )

        print(f"GRPO Dataset: {len(self.tokens)} valid tokens (loadable)")
        if len(self.tokens) == 0:
            if len(scene_tokens) > 0:
                print(f"Sample scene loader tokens: {list(scene_tokens)[:5]}")
            if len(cache_tokens) > 0:
                print(f"Sample cache tokens: {list(cache_tokens)[:5]}")
            if tokens is not None and len(tokens) > 0:
                sample_token = tokens[0]
                print(f"Sample requested token: {sample_token}")
                print(
                    f"  Direct frame-token match: "
                    f"{sample_token in scene_tokens and sample_token in cache_tokens}"
                )
            raise ValueError(
                "GRPO dataset is empty: no resolved tokens have loadable sensor blobs. "
                "Check metric cache coverage, scene_filter.tokens, and sensor/log paths."
            )
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        num_tokens = len(self.tokens)
        last_error = None

        # Bounded scan (no recursion): tokens are pre-filtered to ones with
        # sensor blobs, but stay defensive against transient read errors.
        for attempt in range(num_tokens):
            token = self.tokens[(idx + attempt) % num_tokens]
            try:
                agent_input = self.scene_loader.get_agent_input_from_token(token)
            except (FileNotFoundError, OSError) as exc:
                last_error = exc
                continue

            features = {}
            for builder in self.feature_builders:
                features.update(builder.compute_features(agent_input))

            scene = self.scene_loader.get_scene_from_token(token)
            targets = {}
            for builder in self.target_builders:
                targets.update(builder.compute_targets(scene))

            # Return token instead of metric_cache (loaded lazily in trainer);
            # this avoids pickle issues with the DataLoader workers.
            return features, targets, token

        raise RuntimeError(
            f"No loadable sample among {num_tokens} tokens (start idx={idx}). "
            f"Last error: {last_error!r}"
        )


class GRPODataModule(pl.LightningDataModule):
    """DataModule for GRPO training."""
    
    def __init__(
        self,
        config,
        train_test_split,
        navsim_log_path: str,
        sensor_blobs_path: str,
        metric_cache_path: str,
        batch_size: int = 1,
        num_workers: int = 4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.train_test_split = train_test_split
        
    def setup(self, stage: str):
        """Setup datasets."""
        from hydra.utils import instantiate
        
        # Debug: print train_test_split structure
        print(f"DEBUG: train_test_split type: {type(self.train_test_split)}")
        print(f"DEBUG: train_test_split keys: {list(self.train_test_split.keys()) if hasattr(self.train_test_split, 'keys') else 'N/A'}")
        
        # Build scene filter (similar to run_training.py). Large token lists can
        # be passed via scene_filter.tokens_file to avoid shell argument limits.
        scene_filter_cfg = self.train_test_split.get('scene_filter', None)
        print(f"DEBUG: scene_filter_cfg: {scene_filter_cfg}")

        tokens = self.train_test_split.get('scene_filter', {}).get('tokens', None)
        tokens_file = self.train_test_split.get('scene_filter', {}).get('tokens_file', None)
        if tokens is None and tokens_file:
            with open(tokens_file, "r") as f:
                payload = json.load(f)
            tokens = payload["tokens"] if isinstance(payload, dict) else payload
        if tokens is not None:
            tokens = list(tokens)

        scene_filter_cfg_for_instantiate = OmegaConf.create(
            OmegaConf.to_container(self.train_test_split.scene_filter, resolve=True)
        )
        if "tokens_file" in scene_filter_cfg_for_instantiate:
            del scene_filter_cfg_for_instantiate["tokens_file"]
        if tokens is not None:
            scene_filter_cfg_for_instantiate.tokens = tokens

        scene_filter: SceneFilter = instantiate(scene_filter_cfg_for_instantiate)

        # Restrict logs to the train/val split. When an explicit token list is
        # provided (assignment / official val scenes), keep val logs too.
        train_logs = getattr(self.train_test_split, 'train_logs', None)
        val_logs = getattr(self.train_test_split, 'val_logs', None)
        if train_logs is not None:
            allowed_logs = set(train_logs)
            if tokens is not None and val_logs is not None:
                allowed_logs |= set(val_logs)
            if scene_filter.log_names is not None:
                scene_filter.log_names = [
                    log_name for log_name in scene_filter.log_names if log_name in allowed_logs
                ]
            else:
                scene_filter.log_names = sorted(allowed_logs)
        
        print(f"Scene filter: {len(scene_filter.log_names) if scene_filter.log_names else 'all'} logs")
        
        # Feature/target builders
        feature_builders = [TransfuserFeatureBuilder(config=self.config)]
        target_builders = [TransfuserTargetBuilder(config=self.config)]
        
        # Scene loader - only load current frame sensors (include=[3])
        # to match TransfuserAgentAR.get_sensor_config() and save memory
        from navsim.common.dataclasses import SensorConfig
        scene_loader = SceneLoader(
            sensor_blobs_path=Path(self.hparams.sensor_blobs_path),
            data_path=Path(self.hparams.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_all_sensors(include=[3]),
        )
        
        # Metric cache loader
        metric_cache_loader = MetricCacheLoader(Path(self.hparams.metric_cache_path))
        
        # Create dataset
        self.dataset = GRPOEpisodeDataset(
            scene_loader=scene_loader,
            metric_cache_loader=metric_cache_loader,
            feature_builders=feature_builders,
            target_builders=target_builders,
            # scene_filter.tokens has already been applied by SceneLoader. Do
            # not pass the scene_token list again here, or each scene_token is
            # resolved to only one representative frame token. Leaving this as
            # None preserves the old behavior: train on the full
            # SceneLoader ∩ metric-cache frame-token intersection.
            tokens=None,
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch):
        """Custom collate for GRPO batch.
        
        NOTE: We assume batch_size=1 for GRPO training.
        This avoids complex collate logic for metric_cache.
        """
        # batch is a list of (features, targets, token) tuples
        features, targets, token = batch[0]
        
        # Add batch dimension only to tensor values
        features_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                         for k, v in features.items()}
        targets_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                        for k, v in targets.items()}
        
        # Return token as-is
        return features_batch, targets_batch, token
