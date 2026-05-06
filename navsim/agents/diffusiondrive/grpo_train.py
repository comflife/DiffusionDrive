"""
Entry point for GRPO training.
Uses Hydra for configuration.

Config-construction policy
--------------------------
The TransfuserConfig used by the policy / reference models is built from the
Hydra cfg, NOT hardcoded — so it can be matched to whichever SFT checkpoint
is being fine-tuned (V=2048 step_corners + v6 modules etc.). Pass overrides
via `++config.<field>=...` on the command line, e.g.:

    ++config.ego_vocab_size=2048
    ++config.ego_vocab_path=/.../codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy
    ++config.ar_codebook_mode=step_corners
    ++config.ar_use_residual_delta=true
    ++config.ar_use_heading_head=true
    ++config.ar_step_aware_agent=true
    ++config.ar_use_ego_cross_attn=true
    ++config.ar_use_deformable_bev=true
    ++config.ar_use_bev_pos_enc=true
    ++config.agent_topk=30

Anything not overridden falls back to the TransfuserConfig dataclass default.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pathlib import Path
from dataclasses import fields

from navsim.agents.diffusiondrive.grpo_trainer import GRPOTrainer
from navsim.agents.diffusiondrive.grpo_datamodule import GRPODataModule
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from pytorch_lightning.loggers import WandbLogger


def _build_transfuser_config(cfg: DictConfig) -> TransfuserConfig:
    """Build a TransfuserConfig from cfg.config.* (Hydra overrides).

    All TransfuserConfig dataclass fields are accepted; unknown keys are
    ignored with a warning. This lets GRPO match any SFT recipe by simply
    forwarding the same `agent.config.*` overrides as `++config.*`.
    """
    raw = OmegaConf.to_container(cfg.get('config', OmegaConf.create({})), resolve=True) or {}

    valid_keys = {f.name for f in fields(TransfuserConfig)}
    accepted, unknown = {}, []
    for k, v in raw.items():
        if k in valid_keys:
            accepted[k] = v
        else:
            unknown.append(k)
    if unknown:
        print(f"[grpo_train] Ignoring non-TransfuserConfig overrides: {unknown}")

    config = TransfuserConfig(**accepted)
    print(
        f"[grpo_train] TransfuserConfig built with overrides: "
        f"V={config.ego_vocab_size}, codebook_mode={config.ar_codebook_mode}, "
        f"agent_topk={config.agent_topk}, "
        f"residual_delta={config.ar_use_residual_delta}, "
        f"heading_head={config.ar_use_heading_head}, "
        f"step_aware={config.ar_step_aware_agent}, "
        f"ego_cross_attn={config.ar_use_ego_cross_attn}, "
        f"deformable_bev={config.ar_use_deformable_bev}, "
        f"bev_pos_enc={config.ar_use_bev_pos_enc}"
    )
    return config


@hydra.main(config_path="../../planning/script/config/training", config_name="default_training", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for GRPO training."""

    pl.seed_everything(0)

    config = _build_transfuser_config(cfg)

    # Setup datamodule
    # NOTE: Use top-level num_workers override if provided (e.g., ++num_workers=0),
    # otherwise fall back to dataloader.params.num_workers from default_training.yaml
    num_workers = cfg.get('num_workers', cfg.dataloader.params.num_workers)
    datamodule = GRPODataModule(
        config=config,
        train_test_split=cfg.train_test_split,
        navsim_log_path=cfg.get('navsim_log_path'),
        sensor_blobs_path=cfg.get('sensor_blobs_path'),
        metric_cache_path=cfg.get('metric_cache_path'),
        batch_size=cfg.get('batch_size', 1),
        num_workers=num_workers,
    )

    # Setup model
    model = GRPOTrainer(
        config=config,
        checkpoint_path=cfg.get('checkpoint_path'),
        metric_cache_path=cfg.get('metric_cache_path'),
        lr=cfg.get('lr', 1e-5),
        group_size=cfg.get('group_size', 8),
        kl_coef=cfg.get('kl_coef', 0.01),
        temperature=cfg.get('temperature', 1.0),
        clip_eps=cfg.get('clip_eps', 0.2),                       # GRPO token-level clip
        algorithm=cfg.get('algorithm', 'grpo'),                  # 'grpo' | 'gspo' | 'gspo_token' | 'grpo_plus'
        clip_eps_seq=cfg.get('clip_eps_seq', 4e-4),              # GSPO / GRPO+ sequence-level clip
        token_attention_alpha=cfg.get('token_attention_alpha', 0.5),  # GRPO+ blend
    )

    # Setup callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=Path(cfg.output_dir) / "checkpoints",
            filename='grpo-{epoch:02d}',
            save_top_k=3,
            monitor='train/mean_reward',
            mode='max',
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]

    # Setup wandb logger if enabled
    loggers = []
    if cfg.wandb.get("enabled", False):
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.get("name", None),
            entity=cfg.wandb.get("entity", None),
            tags=cfg.wandb.get("tags", []),
            notes=cfg.wandb.get("notes", None),
            save_dir=cfg.output_dir,
        )
        loggers.append(wandb_logger)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.params.max_epochs,
        devices=cfg.trainer.params.devices,
        strategy=cfg.trainer.params.strategy,
        precision=cfg.trainer.params.precision,
        callbacks=callbacks,
        logger=loggers if loggers else None,
        default_root_dir=cfg.output_dir,
        accumulate_grad_batches=cfg.trainer.params.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.params.gradient_clip_val,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    print(f"Training complete! Checkpoints saved to: {cfg.output_dir}/checkpoints")


if __name__ == "__main__":
    main()
