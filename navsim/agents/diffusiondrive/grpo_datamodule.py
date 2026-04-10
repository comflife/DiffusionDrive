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
from navsim.common.dataloader import SceneLoader
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.metric_caching.metric_cache_loader import MetricCacheLoader
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
        
        # Filter tokens that exist in both
        if tokens is None:
            self.tokens = list(
                set(scene_loader.tokens) & set(metric_cache_loader.tokens)
            )
        else:
            self.tokens = [
                t for t in tokens 
                if t in scene_loader.tokens and t in metric_cache_loader.tokens
            ]
        
        print(f"GRPO Dataset: {len(self.tokens)} valid tokens")
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        token = self.tokens[idx]
        
        # Get agent input
        agent_input = self.scene_loader.get_agent_input_from_token(token)
        
        # Build features
        features = {}
        for builder in self.feature_builders:
            features.update(builder.build(agent_input))
        
        # Build targets
        targets = {}
        for builder in self.target_builders:
            targets.update(builder.build(agent_input))
        
        # Load metric cache
        metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)
        
        return features, targets, metric_cache


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
        
        # Build scene filter
        scene_filter: SceneFilter = instantiate(self.train_test_split.scene_filter)
        
        # Feature/target builders
        feature_builders = [TransfuserFeatureBuilder(config=self.config)]
        target_builders = [TransfuserTargetBuilder(config=self.config)]
        
        # Scene loader
        scene_loader = SceneLoader(
            sensor_blobs_path=Path(self.hparams.sensor_blobs_path),
            data_path=Path(self.hparams.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=feature_builders[0].get_sensor_config(),
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
        """Custom collate for GRPO batch."""
        # Each item is (features, targets, metric_cache)
        # metric_cache is not tensor, handle separately
        features_list, targets_list, metric_caches = zip(*batch)
        
        # Stack features
        features = {}
        for key in features_list[0].keys():
            values = [f[key] for f in features_list]
            if isinstance(values[0], torch.Tensor):
                features[key] = torch.stack(values)
            else:
                features[key] = values
        
        # Stack targets
        targets = {}
        for key in targets_list[0].keys():
            values = [t[key] for t in targets_list]
            if isinstance(values[0], torch.Tensor):
                targets[key] = torch.stack(values)
            else:
                targets[key] = values
        
        return features, targets, metric_caches
