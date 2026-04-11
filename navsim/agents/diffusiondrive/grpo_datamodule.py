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

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder


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
        
        # Filter tokens that exist in both
        if tokens is None:
            self.tokens = list(scene_tokens & cache_tokens)
        else:
            requested_tokens = set(tokens)
            self.tokens = list(requested_tokens & scene_tokens & cache_tokens)
            print(f"Requested: {len(requested_tokens)} tokens")
        
        print(f"GRPO Dataset: {len(self.tokens)} valid tokens (intersection)")
        
        # Show some examples of missing tokens
        if len(self.tokens) == 0:
            if len(scene_tokens) > 0:
                print(f"Sample scene tokens: {list(scene_tokens)[:5]}")
            if len(cache_tokens) > 0:
                print(f"Sample cache tokens: {list(cache_tokens)[:5]}")
            if tokens is not None and len(tokens) > 0:
                sample_token = tokens[0]
                print(f"Sample requested token: {sample_token}")
                print(f"  In scene_loader: {sample_token in scene_tokens}")
                print(f"  In cache_loader: {sample_token in cache_tokens}")
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        token = self.tokens[idx]
        
        # Get agent input
        try:
            agent_input = self.scene_loader.get_agent_input_from_token(token)
        except FileNotFoundError as e:
            print(f"Warning: Missing sensor data for token {token}: {e}")
            # Return a different sample
            return self.__getitem__((idx + 1) % len(self.tokens))
        
        # Build features (use compute_features, not build)
        features = {}
        for builder in self.feature_builders:
            features.update(builder.compute_features(agent_input))
        
        # Build targets (need scene for targets)
        # Load scene for target computation
        scene = self.scene_loader.get_scene_from_token(token)
        targets = {}
        for builder in self.target_builders:
            targets.update(builder.compute_targets(scene))
        
        # Return token instead of metric_cache (will be loaded lazily in trainer)
        # This avoids pickle issues with DataLoader
        return features, targets, token


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
        
        # Build scene filter (similar to run_training.py)
        scene_filter: SceneFilter = instantiate(self.train_test_split.scene_filter)
        
        # Filter log_names based on train_logs if available
        train_logs = getattr(self.train_test_split, 'train_logs', None)
        if train_logs is not None:
            if scene_filter.log_names is not None:
                scene_filter.log_names = [
                    log_name for log_name in scene_filter.log_names if log_name in train_logs
                ]
            else:
                scene_filter.log_names = train_logs
        
        print(f"Scene filter: {len(scene_filter.log_names) if scene_filter.log_names else 'all'} logs")
        
        # Feature/target builders
        feature_builders = [TransfuserFeatureBuilder(config=self.config)]
        target_builders = [TransfuserTargetBuilder(config=self.config)]
        
        # Scene loader - use all sensors (needed for feature computation)
        from navsim.common.dataclasses import SensorConfig
        scene_loader = SceneLoader(
            sensor_blobs_path=Path(self.hparams.sensor_blobs_path),
            data_path=Path(self.hparams.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_all_sensors(),
        )
        
        # Metric cache loader
        metric_cache_loader = MetricCacheLoader(Path(self.hparams.metric_cache_path))
        
        # Create dataset
        self.dataset = GRPOEpisodeDataset(
            scene_loader=scene_loader,
            metric_cache_loader=metric_cache_loader,
            feature_builders=feature_builders,
            target_builders=target_builders,
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
