"""
Agent interface for Discrete Autoregressive TransFuser.

Uses V2TransfuserModelAR with discrete token prediction instead of diffusion.
"""

from typing import Any, List, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.diffusiondrive.dd_abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

# Import AR model instead of diffusion model
from navsim.agents.diffusiondrive.transfuser_model_ar import V2TransfuserModelAR as TransfuserModelAR

from navsim.agents.diffusiondrive.transfuser_callback import TransfuserCallback 
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
import os
import glob


class RollingLastNCheckpoint(pl.Callback):
    """Save a checkpoint each epoch and keep only the most recent N.

    Used when val is skipped — there's no metric to monitor, so we just
    retain a sliding window of the latest train epochs.
    """

    def __init__(self, dirpath: str, n: int = 5, filename_template: str = "epoch_{epoch:02d}.ckpt"):
        super().__init__()
        self.dirpath = dirpath
        self.n = n
        self.filename_template = filename_template

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.is_global_zero:
            os.makedirs(self.dirpath, exist_ok=True)
            epoch = trainer.current_epoch
            path = os.path.join(self.dirpath, self.filename_template.format(epoch=epoch))
            trainer.save_checkpoint(path)
            existing = sorted(glob.glob(os.path.join(self.dirpath, "epoch_*.ckpt")))
            while len(existing) > self.n:
                old = existing.pop(0)
                try:
                    os.remove(old)
                except OSError:
                    pass


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)


class TransfuserAgentAR(AbstractAgent):
    """Agent interface for Discrete Autoregressive TransFuser."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        checkpoint_save_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initializes TransFuser AR agent.

        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint
        :param checkpoint_save_dir: optional explicit dir for ModelCheckpoint to write into.
            If None, falls back to Lightning's default (which routes through WandbLogger
            and produces wandb_run_id-named subfolders).
        """
        super().__init__()

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._checkpoint_save_dir = checkpoint_save_dir
        
        # Use AR model
        self._transfuser_model = TransfuserModelAR(config)
        self.init_from_pretrained()

    def init_from_pretrained(self):
        """Initialize from pretrained checkpoint."""
        import os
        
        if self._checkpoint_path and os.path.isfile(self._checkpoint_path):
            print(f"Loading pretrained checkpoint from: {self._checkpoint_path}")
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['state_dict']
            
            # Remove common prefixes from checkpoint keys
            # Handle both 'agent._transfuser_model.' and '_transfuser_model.' prefixes
            # Also handle GRPO checkpoints which use 'policy_model.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                # Skip GRPO reference model (frozen copy, not needed for inference)
                if k.startswith('reference_model.'):
                    continue
                # Convert GRPO policy_model prefix
                elif k.startswith('policy_model.'):
                    new_key = k[len('policy_model.'):]
                # Remove 'agent._transfuser_model.' or '_transfuser_model.' prefix
                elif k.startswith('agent._transfuser_model.'):
                    new_key = k[len('agent._transfuser_model.'):]
                elif k.startswith('_transfuser_model.'):
                    new_key = k[len('_transfuser_model.'):]
                elif k.startswith('agent.'):
                    new_key = k[len('agent.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            
            # Load state dict directly into the model (not the LightningModule wrapper)
            missing_keys, unexpected_keys = self._transfuser_model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {len(missing_keys)} keys")
                print(f"First 5 missing: {missing_keys[:5]}")
                print("(This is expected for AR head parameters)")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 10:
                    print(f"Unexpected: {unexpected_keys}")
        else:
            if self._checkpoint_path:
                print(f"Checkpoint not found at: {self._checkpoint_path}")
            print("Initializing from scratch (no pretrained weights).")

        if getattr(self._config, "freeze_pretrained_trunk", False):
            self._freeze_pretrained_trunk()

    def _freeze_pretrained_trunk(self):
        """Freeze the pretrained DiffusionDrive trunk and train AR trajectory head only."""
        trainable_prefixes = [
            "_trajectory_head.",
        ]

        total_params = 0
        trainable_params = 0
        for name, param in self._transfuser_model.named_parameters():
            total_params += param.numel()
            should_train = any(name.startswith(prefix) for prefix in trainable_prefixes)
            param.requires_grad = should_train
            if should_train:
                trainable_params += param.numel()

        frozen_params = total_params - trainable_params
        print(
            "Froze pretrained trunk: "
            f"trainable={trainable_params:,}, frozen={frozen_params:,}"
        )
            
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass.
        
        For evaluation, move model to GPU if available.
        Checkpoint loading is handled in init_from_pretrained() during __init__.
        This is called in each Ray worker, so we ensure GPU usage here.
        """
        if torch.cuda.is_available():
            print(f"CUDA is available in worker (device_count={torch.cuda.device_count()}). Moving model to GPU.")
            self._transfuser_model = self._transfuser_model.to("cuda")
            print(f"Model moved to GPU: cuda:{torch.cuda.current_device()}")
            # Ensure all parameters and buffers are on GPU
            self._transfuser_model = self._transfuser_model.cuda()
        else:
            print("CUDA not available in worker, using CPU for inference.")
            self._transfuser_model = self._transfuser_model.to("cpu")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(features, targets=targets)
        
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for AR model.
        
        Uses trajectory_loss computed by AR head (CrossEntropy).
        Returns dict with 'loss' key and other loss components for logging.
        """
        # AR head returns trajectory_loss directly when targets provided
        if 'trajectory_loss' in predictions:
            traj_loss = predictions['trajectory_loss']
        else:
            # Re-run forward with targets to get loss
            # This is needed because LightningModule calls forward without targets
            predictions_with_loss = self._transfuser_model(features, targets=targets)
            if 'trajectory_loss' in predictions_with_loss:
                traj_loss = predictions_with_loss['trajectory_loss']
            else:
                # Fallback: compute L1 loss on trajectory
                pred_traj = predictions['trajectory']
                gt_traj = targets['trajectory']
                traj_loss = torch.nn.functional.l1_loss(pred_traj, gt_traj)
        
        # Return dict with loss components for logging
        loss_dict = {
            'loss': traj_loss,
            'trajectory_loss': traj_loss,
        }

        for key in ['token_loss', 'traj_loss', 'heading_loss']:
            if key in predictions:
                loss_dict[key] = predictions[key]

        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(
            self._transfuser_model.parameters(),
            lr=self._lr,
            weight_decay=self._config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self._config.lr_steps,
            gamma=0.1
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
        optimizer_cfg = dict(
            type=self._config.optimizer_type, 
            lr=self._lr, 
            weight_decay=self._config.weight_decay,
            paramwise_cfg=self._config.opt_paramwise_cfg
        )
        scheduler_cfg = dict(
            type=self._config.scheduler_type,
            milestones=self._config.lr_steps,
            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._transfuser_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._transfuser_model.parameters()
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})
        
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        from pytorch_lightning.callbacks import ModelCheckpoint

        save_top_k = getattr(self._config, "ckpt_save_top_k", 3)
        monitor    = getattr(self._config, "ckpt_monitor", "val/loss")

        if monitor in (None, "", "null", "None"):
            # No metric monitoring: rolling window of latest N epoch ckpts.
            ckpt_dir = self._checkpoint_save_dir or "lightning_logs/checkpoints"
            checkpoint_callback = RollingLastNCheckpoint(dirpath=ckpt_dir, n=save_top_k)
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=self._checkpoint_save_dir,
                filename='{epoch:02d}-{val_loss:.2f}',
                save_top_k=save_top_k,
                monitor=monitor,
                mode='min',
                save_last=True,
                every_n_epochs=1,
            )

        return [TransfuserCallback(self._config), checkpoint_callback]
