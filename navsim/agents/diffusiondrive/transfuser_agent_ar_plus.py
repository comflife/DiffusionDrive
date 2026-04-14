"""
Agent interface for the enhanced discrete autoregressive Transfuser.
"""

from typing import Dict, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim

from navsim.agents.diffusiondrive.dd_abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_model_ar_plus import (
    V2TransfuserModelARPlus as TransfuserModelARPlus,
)
from navsim.agents.diffusiondrive.transfuser_callback import TransfuserCallback
from navsim.agents.diffusiondrive.transfuser_features import (
    TransfuserFeatureBuilder,
    TransfuserTargetBuilder,
)
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type_name = cfg.pop("type")
    return getattr(obj, type_name)(**cfg, **kwargs)


class TransfuserAgentARPlus(AbstractAgent):
    """Enhanced discrete autoregressive Transfuser agent."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModelARPlus(config)
        self.init_from_pretrained()

    def init_from_pretrained(self):
        import os

        if self._checkpoint_path and os.path.isfile(self._checkpoint_path):
            print(f"Loading pretrained checkpoint from: {self._checkpoint_path}")
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))

            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("reference_model."):
                    continue
                if k.startswith("policy_model."):
                    new_key = k[len("policy_model."):]
                elif k.startswith("agent._transfuser_model."):
                    new_key = k[len("agent._transfuser_model."):]
                elif k.startswith("_transfuser_model."):
                    new_key = k[len("_transfuser_model."):]
                elif k.startswith("agent."):
                    new_key = k[len("agent."):]
                else:
                    new_key = k
                new_state_dict[new_key] = v

            missing_keys, unexpected_keys = self._transfuser_model.load_state_dict(
                new_state_dict, strict=False
            )
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {len(missing_keys)} keys")
                print(f"First 5 missing: {missing_keys[:5]}")
                print("(This is expected for enhanced AR head parameters)")
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
        trainable_prefixes = ["_trajectory_head."]
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
        return self.__class__.__name__

    def initialize(self) -> None:
        if torch.cuda.is_available():
            self._transfuser_model = self._transfuser_model.to("cuda")
            self._transfuser_model = self._transfuser_model.cuda()
        else:
            self._transfuser_model = self._transfuser_model.to("cpu")

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._transfuser_model(features, targets=targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if "trajectory_loss" in predictions:
            traj_loss = predictions["trajectory_loss"]
        else:
            predictions_with_loss = self._transfuser_model(features, targets=targets)
            if "trajectory_loss" in predictions_with_loss:
                traj_loss = predictions_with_loss["trajectory_loss"]
            else:
                traj_loss = torch.nn.functional.l1_loss(
                    predictions["trajectory"], targets["trajectory"]
                )

        loss_dict = {
            "loss": traj_loss,
            "trajectory_loss": traj_loss,
        }
        for key in ["token_loss", "traj_loss", "heading_loss"]:
            if key in predictions:
                loss_dict[key] = predictions[key]
        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return self.get_coslr_optimizers()

    def get_coslr_optimizers(self):
        optimizer_cfg = dict(
            type=self._config.optimizer_type,
            lr=self._lr,
            weight_decay=self._config.weight_decay,
            paramwise_cfg=self._config.opt_paramwise_cfg,
        )
        scheduler_cfg = dict(
            type=self._config.scheduler_type,
            milestones=self._config.lr_steps,
            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)

        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop("paramwise_cfg", None)

        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg["name"]]

            for k, v in self._transfuser_model.named_parameters():
                if not v.requires_grad:
                    continue
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg["name"].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = [p for p in self._transfuser_model.parameters() if p.requires_grad]

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)

        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg["name"].items()):
                if not pg:
                    continue
                cfg = {}
                if "lr_mult" in pg_cfg:
                    cfg["lr"] = optimizer_cfg["lr"] * pg_cfg["lr_mult"]
                optimizer.add_param_group({"params": pg, **cfg})

        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_training_callbacks(self):
        from pytorch_lightning.callbacks import ModelCheckpoint

        checkpoint_callback = ModelCheckpoint(
            dirpath=None,
            filename="{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
            every_n_epochs=1,
        )
        return [TransfuserCallback(self._config), checkpoint_callback]
