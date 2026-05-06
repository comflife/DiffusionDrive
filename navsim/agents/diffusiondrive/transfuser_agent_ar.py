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
from navsim.agents.diffusiondrive.transfuser_loss import _agent_loss
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
import torch.nn.functional as F
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
import os
import re
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
        os.makedirs(self.dirpath, exist_ok=True)
        epoch = trainer.current_epoch
        path = os.path.join(self.dirpath, self.filename_template.format(epoch=epoch))

        # In DDP, all ranks must enter Lightning's checkpoint path because the
        # strategy may perform collective synchronization internally.
        trainer.save_checkpoint(path)

        if trainer.is_global_zero:
            existing = sorted(glob.glob(os.path.join(self.dirpath, "epoch_*.ckpt")))
            while len(existing) > self.n:
                old = existing.pop(0)
                try:
                    os.remove(old)
                except OSError:
                    pass


class MilestoneEpochCheckpoint(pl.Callback):
    """Save a checkpoint at epoch ∈ {start, start+every, start+2*every, ...}.

    Independent of monitor/top-k logic — runs alongside ModelCheckpoint to
    preserve fixed-cadence snapshots for late-stage analysis.
    """

    def __init__(
        self,
        dirpath: str,
        start_epoch: int,
        every_n: int = 10,
        filename_template: str = "milestone_epoch_{epoch:03d}.ckpt",
    ):
        super().__init__()
        self.dirpath = dirpath
        self.start_epoch = start_epoch
        self.every_n = max(1, every_n)
        self.filename_template = filename_template

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        if epoch < self.start_epoch:
            return
        if (epoch - self.start_epoch) % self.every_n != 0:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        path = os.path.join(self.dirpath, self.filename_template.format(epoch=epoch))

        # In DDP, all ranks must enter Lightning's checkpoint path because the
        # strategy may perform collective synchronization internally.
        trainer.save_checkpoint(path)


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
        """Initialize from pretrained checkpoint.

        On top of standard prefix stripping, this also remaps the original
        DiffusionDrive ``_trajectory_head.diff_decoder.layers.{i}.{module}.{x}``
        keys onto the AR head's analogous module names, so the pretrained
        88.1 PDMS cross-attention weights warm-start the AR cross-attentions
        (bev_deform_attn / e2a_attn / ego_attn / ffn / norms) instead of being
        discarded — they were structurally identical, only the names differed.
        """
        if self._checkpoint_path and os.path.isfile(self._checkpoint_path):
            print(f"Loading pretrained checkpoint from: {self._checkpoint_path}")
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))

            state_dict = checkpoint['state_dict']

            # Strip framework prefixes (Lightning + GRPO/SFT pipeline variants).
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('reference_model.'):
                    continue
                elif k.startswith('policy_model.'):
                    new_key = k[len('policy_model.'):]
                elif k.startswith('agent._transfuser_model.'):
                    new_key = k[len('agent._transfuser_model.'):]
                elif k.startswith('_transfuser_model.'):
                    new_key = k[len('_transfuser_model.'):]
                elif k.startswith('agent.'):
                    new_key = k[len('agent.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v

            new_state_dict = self._remap_diff_decoder_to_ar(new_state_dict)

            model_sd = self._transfuser_model.state_dict()
            missing_keys, unexpected_keys = self._transfuser_model.load_state_dict(
                new_state_dict, strict=False)

            ar_head_total = sum(1 for k in model_sd if k.startswith('_trajectory_head.'))
            ar_head_missing = [k for k in missing_keys if k.startswith('_trajectory_head.')]
            ar_head_loaded = ar_head_total - len(ar_head_missing)
            print(
                f"Pretrained load summary: "
                f"AR head warm-started {ar_head_loaded}/{ar_head_total} tensors, "
                f"{len(ar_head_missing)} random-init."
            )
            if missing_keys:
                print(f"  total missing: {len(missing_keys)} (sample: {missing_keys[:5]})")
            if unexpected_keys:
                print(f"  total unexpected: {len(unexpected_keys)} (sample: {unexpected_keys[:5]})")
        else:
            if self._checkpoint_path:
                print(f"Checkpoint not found at: {self._checkpoint_path}")
            print("Initializing from scratch (no pretrained weights).")

        if getattr(self._config, "freeze_pretrained_trunk", False):
            self._freeze_pretrained_trunk()

    @staticmethod
    def _remap_diff_decoder_to_ar(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Map original DiffusionDrive trajectory-head keys to AR-head module names.

        Original layout (per layer i ∈ {0,1}):
            _trajectory_head.diff_decoder.layers.{i}.cross_bev_attention.{x}
            _trajectory_head.diff_decoder.layers.{i}.cross_agent_attention.{x}
            _trajectory_head.diff_decoder.layers.{i}.cross_ego_attention.{x}
            _trajectory_head.diff_decoder.layers.{i}.ffn.{0|2}.{x}
            _trajectory_head.diff_decoder.layers.{i}.norm{1|2|3}.{x}
            _trajectory_head.diff_decoder.layers.{i}.{time_modulation,task_decoder}.{x}  -> dropped

        AR layout target (per layer i):
            _trajectory_head.bev_deform_attn.{i}.{x}    <- cross_bev_attention
            _trajectory_head.e2a_attn.{i}.{x}           <- cross_agent_attention
            _trajectory_head.ego_attn.{i}.{x}           <- cross_ego_attention   (only when use_ego_cross_attn)
            _trajectory_head.ffn.{i}.{0,3}.{x}          <- ffn.{0,2} (AR ffn has Dropout at idx 2)
            _trajectory_head.bev_deform_norm.{i}.{x}    <- norm1   (post-bev LN)
            _trajectory_head.e2a_norm.{i}.{x}           <- norm1   (replicated; original norm1 was post-bev+post-agent)
            _trajectory_head.ego_norm.{i}.{x}           <- norm2
            _trajectory_head.ffn_norm.{i}.{x}           <- norm3

        Targets that don't exist on the current model (e.g. bev_deform_attn when
        use_deformable_bev=False) are silently filtered by load_state_dict's
        strict=False — they show up in unexpected_keys, which is harmless.
        """
        pat = re.compile(r'^_trajectory_head\.diff_decoder\.layers\.(\d+)\.([^.]+)\.(.+)$')

        def map_ffn(rest: str):
            # Original ffn = [Linear, ReLU, Linear]; AR ffn = [Linear, GELU, Dropout, Linear, Dropout].
            if rest.startswith('0.'):
                return [('ffn', '0.' + rest[2:])]
            if rest.startswith('2.'):
                return [('ffn', '3.' + rest[2:])]
            return []

        def fixed_target(target):
            return lambda rest: [(target, rest)]

        rules = {
            'cross_bev_attention':   fixed_target('bev_deform_attn'),
            'cross_agent_attention': fixed_target('e2a_attn'),
            'cross_ego_attention':   fixed_target('ego_attn'),
            'ffn':                   map_ffn,
            'norm1':                 lambda r: [('bev_deform_norm', r), ('e2a_norm', r)],
            'norm2':                 fixed_target('ego_norm'),
            'norm3':                 fixed_target('ffn_norm'),
        }

        new_sd: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            m = pat.match(k)
            if m is None:
                new_sd[k] = v
                continue
            i, mod, rest = m.group(1), m.group(2), m.group(3)
            rule = rules.get(mod)
            if rule is None:
                continue
            for ar_mod, ar_rest in rule(rest):
                new_sd[f'_trajectory_head.{ar_mod}.{i}.{ar_rest}'] = v
        return new_sd

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

        - trajectory_loss: AR head's internal weighted CE + traj L1 + heading L1.
        - When the trunk is being trained (freeze_pretrained_trunk=False), the
          agent_head and bev_semantic_head also need direct supervision; otherwise
          they drift and the AR head's agent_kv input degrades.
        """
        if 'trajectory_loss' in predictions:
            traj_loss = predictions['trajectory_loss']
        else:
            # Lightning calls forward(features, targets) so this is a defensive fallback only.
            predictions_with_loss = self._transfuser_model(features, targets=targets)
            traj_loss = predictions_with_loss.get(
                'trajectory_loss',
                F.l1_loss(predictions['trajectory'], targets['trajectory']),
            )

        loss_dict = {'trajectory_loss': traj_loss}
        for key in ['token_loss', 'traj_loss', 'heading_loss']:
            if key in predictions:
                loss_dict[key] = predictions[key]

        # Auxiliary supervision on agent_head + bev_semantic_head. Skip when the
        # trunk is fully frozen (only the AR head trains; aux losses contribute
        # nothing because their parameters have requires_grad=False).
        cfg = self._config
        trunk_frozen = bool(getattr(cfg, "freeze_pretrained_trunk", False))
        if not trunk_frozen and 'agent_states' in predictions and 'bev_semantic_map' in predictions:
            agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, cfg)
            bev_semantic_loss = F.cross_entropy(
                predictions['bev_semantic_map'], targets['bev_semantic_map'].long()
            )
            aux = (
                cfg.agent_class_weight * agent_class_loss
                + cfg.agent_box_weight * agent_box_loss
                + cfg.bev_semantic_weight * bev_semantic_loss
            )
            loss_dict['agent_class_loss'] = agent_class_loss.detach()
            loss_dict['agent_box_loss'] = agent_box_loss.detach()
            loss_dict['bev_semantic_loss'] = bev_semantic_loss.detach()
            loss_dict['loss'] = traj_loss + aux
        else:
            loss_dict['loss'] = traj_loss

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
        # Uniform LR policy: all params share the same base LR. The legacy
        # paramwise rule (image_encoder × cfg_lr_mult) still applies for parity
        # with the original DiffusionDrive recipe. The trunk_lr_mult head/trunk
        # split has been removed — joint fine-tuning is simpler with one rate
        # and the warm-started AR cross-attentions no longer need protection.
        if float(getattr(self._config, "trunk_lr_mult", 1.0)) != 1.0:
            print(
                f"[lr] config.trunk_lr_mult={self._config.trunk_lr_mult} is ignored "
                "(uniform LR policy now in effect)."
            )

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
            epochs=int(getattr(self._config, "cos_lr_epochs", 100)),
            warmup_epochs=int(getattr(self._config, "cos_lr_warmup_epochs", 3)),
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

        callbacks: List[pl.Callback] = [TransfuserCallback(self._config), checkpoint_callback]

        # Optional: fixed-cadence milestone checkpoints (e.g. epoch 80, 90, 100, ...).
        ms_start = int(getattr(self._config, "ckpt_milestone_start", -1))
        ms_every = int(getattr(self._config, "ckpt_milestone_every", 10))
        if ms_start >= 0:
            ms_dir = self._checkpoint_save_dir or "lightning_logs/checkpoints"
            callbacks.append(
                MilestoneEpochCheckpoint(dirpath=ms_dir, start_epoch=ms_start, every_n=ms_every)
            )

        return callbacks
