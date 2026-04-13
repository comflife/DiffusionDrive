"""
Entry point for GRPO training.
Uses Hydra for configuration.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pathlib import Path

from navsim.agents.diffusiondrive.grpo_trainer import GRPOTrainer
from navsim.agents.diffusiondrive.grpo_datamodule import GRPODataModule
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="../../planning/script/config/training", config_name="default_training", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for GRPO training."""
    
    pl.seed_everything(0)
    
    # Override config for GRPO
    config = TransfuserConfig(
        ego_vocab_path='/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy',
        ego_vocab_size=512,
        agent_topk=8,
    )
    
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
        clip_eps=cfg.get('clip_eps', 0.2),   # PPO clipping epsilon
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
