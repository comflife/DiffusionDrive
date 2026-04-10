"""
Entry point for Custom Multi-Scale RL training.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pathlib import Path

from navsim.agents.diffusiondrive.custom_rl_trainer import MultiScaleRLTrainer
from navsim.agents.diffusiondrive.grpo_datamodule import GRPODataModule
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


@hydra.main(config_path="../../planning/script/config/training", config_name="default_training", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for Custom RL training."""
    
    pl.seed_everything(0)
    
    print("=" * 60)
    print("Custom Multi-Scale RL Training")
    print("=" * 60)
    print(f"GSPO Aggregation: {cfg.get('gspo_aggregation', 'mean')}")
    print(f"Global Advantage Weight: {cfg.get('global_advantage_weight', 0.7)}")
    print(f"Local Advantage Weight: {cfg.get('local_advantage_weight', 0.3)}")
    print("=" * 60)
    
    # Config
    config = TransfuserConfig(
        ego_vocab_path='/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy',
        ego_vocab_size=512,
        agent_topk=8,
    )
    
    # DataModule (reusing GRPO datamodule)
    datamodule = GRPODataModule(
        config=config,
        train_test_split=cfg.train_test_split,
        navsim_log_path=cfg.navsim_log_path,
        sensor_blobs_path=cfg.sensor_blobs_path,
        metric_cache_path=cfg.metric_cache_path,
        batch_size=cfg.get('batch_size', 1),
        num_workers=cfg.dataloader.params.num_workers,
    )
    
    # Multi-Scale RL Trainer
    model = MultiScaleRLTrainer(
        config=config,
        checkpoint_path=cfg.get('checkpoint_path'),
        lr=cfg.get('lr', 5e-6),
        group_size=cfg.get('group_size', 8),
        kl_coef=cfg.get('kl_coef', 0.005),
        temperature=cfg.get('temperature', 0.8),
        global_advantage_weight=cfg.get('global_advantage_weight', 0.7),
        local_advantage_weight=cfg.get('local_advantage_weight', 0.3),
        local_window_size=cfg.get('local_window_size', 3),
        gspo_aggregation=cfg.get('gspo_aggregation', 'mean'),
        use_token_level_kl=cfg.get('use_token_level_kl', True),
    )
    
    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=Path(cfg.output_dir) / "checkpoints",
            filename='customrl-{epoch:02d}-{train/mean_reward:.3f}',
            save_top_k=3,
            monitor='train/mean_reward',
            mode='max',
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.params.precision,
        callbacks=callbacks,
        default_root_dir=cfg.output_dir,
        accumulate_grad_batches=cfg.trainer.params.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.params.gradient_clip_val,
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    print(f"Training complete! Checkpoints: {cfg.output_dir}/checkpoints")


if __name__ == "__main__":
    main()
