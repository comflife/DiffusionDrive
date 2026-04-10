"""
GRPO (Group Relative Policy Optimization) Trainer for DiffusionDrive-AR.

References:
- AutoVLA: https://arxiv.org/abs/2410.23218
- GRPO: https://arxiv.org/abs/2402.03300

Architecture:
- Policy model: Trainable AR decoder
- Reference model: Frozen pre-trained AR decoder (for KL penalty)
- Reward: PDM Score from simulation
- Rollout: Sample multiple trajectories per scene, compute advantages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import pytorch_lightning as pl
from omegaconf import DictConfig

from navsim.agents.diffusiondrive.transfuser_model_ar import V2TransfuserModelAR
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer


@dataclass
class GRPORollout:
    """Single rollout trajectory with reward."""
    features: Dict[str, torch.Tensor]
    tokens: torch.Tensor  # [T] predicted token indices
    trajectory: torch.Tensor  # [T, 2] predicted trajectory
    reward: float  # PDM score
    log_probs: torch.Tensor  # [T] log probabilities


class GRPOTrainer(pl.LightningModule):
    """
    GRPO Trainer for DiffusionDrive-AR.
    
    Key components:
    - Policy model: Finetuned AR decoder
    - Reference model: Frozen pretrained (for KL penalty)
    - Group sampling: Multiple rollouts per scene
    - Advantage: Relative performance within group
    """
    
    def __init__(
        self,
        config: TransfuserConfig,
        checkpoint_path: Optional[str] = None,
        lr: float = 1e-5,
        group_size: int = 8,  # Number of rollouts per scene
        kl_coef: float = 0.01,  # KL penalty coefficient
        temperature: float = 1.0,  # Sampling temperature
        max_grad_norm: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.temperature = temperature
        self.max_grad_norm = max_grad_norm
        
        # Policy model (trainable)
        self.policy_model = V2TransfuserModelAR(config)
        
        # Load pretrained weights if provided
        if checkpoint_path:
            self._load_pretrained(checkpoint_path)
        
        # Reference model (frozen, for KL penalty)
        self.reference_model = V2TransfuserModelAR(config)
        if checkpoint_path:
            self._load_pretrained(checkpoint_path, model=self.reference_model)
        self._freeze_model(self.reference_model)
        
        # PDM components for reward computation
        self.simulator: Optional[PDMSimulator] = None
        self.scorer: Optional[PDMScorer] = None
        
    def _load_pretrained(self, checkpoint_path: str, model=None):
        """Load pretrained weights."""
        model = model or self.policy_model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove common prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('agent._transfuser_model.'):
                new_key = k[len('agent._transfuser_model.'):]
            elif k.startswith('_transfuser_model.'):
                new_key = k[len('_transfuser_model.'):]
            elif k.startswith('agent.'):
                new_key = k[len('agent.'):]
            else:
                new_key = k
            new_state_dict[new_key] = v
            
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
        
    def _freeze_model(self, model: nn.Module):
        """Freeze model parameters."""
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
    def setup(self, stage: str):
        """Setup PDM components."""
        if self.simulator is None:
            from hydra.utils import instantiate
            # These should be provided via config
            self.simulator = instantiate(self.trainer.datamodule.hparams.simulator)
            self.scorer = instantiate(self.trainer.datamodule.hparams.scorer)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through policy model."""
        return self.policy_model(features, targets=None)
    
    @torch.no_grad()
    def sample_rollouts(
        self, 
        features: Dict[str, torch.Tensor],
        metric_cache,
        group_size: Optional[int] = None
    ) -> List[GRPORollout]:
        """
        Sample multiple rollouts for a single scene.
        
        Args:
            features: Input features
            metric_cache: PDM metric cache for reward computation
            group_size: Number of rollouts to sample
            
        Returns:
            List of rollouts with rewards
        """
        group_size = group_size or self.group_size
        rollouts = []
        
        self.policy_model.eval()
        
        for _ in range(group_size):
            # Sample trajectory from policy
            with torch.no_grad():
                output = self.policy_model(features, targets=None, temperature=self.temperature)
                
            trajectory = output['trajectory']  # [1, T, 2]
            tokens = output.get('ego_tokens')  # [1, T]
            log_probs = output.get('ego_log_probs')  # [1, T]
            
            if trajectory.dim() == 3:
                trajectory = trajectory[0]  # Remove batch dim
            if tokens is not None and tokens.dim() == 2:
                tokens = tokens[0]
            if log_probs is not None and log_probs.dim() == 2:
                log_probs = log_probs[0]
            
            # Compute PDM reward
            reward = self._compute_pdm_reward(trajectory, metric_cache)
            
            rollout = GRPORollout(
                features=features,
                tokens=tokens,
                trajectory=trajectory,
                reward=reward,
                log_probs=log_probs if log_probs is not None else torch.zeros_like(tokens, dtype=torch.float32)
            )
            rollouts.append(rollout)
            
        return rollouts
    
    def _compute_pdm_reward(self, trajectory: torch.Tensor, metric_cache) -> float:
        """Compute PDM score as reward."""
        if self.simulator is None or self.scorer is None:
            # Fallback: use dummy reward if PDM not available
            return 0.0
            
        try:
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=self.simulator.proposal_sampling,
                simulator=self.simulator,
                scorer=self.scorer,
            )
            # PDM score is typically 0-1, scale to reasonable range
            return pdm_result.score
        except Exception as e:
            print(f"PDM scoring failed: {e}")
            return 0.0
    
    def compute_grpo_loss(
        self,
        rollouts: List[GRPORollout],
        features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss from rollouts.
        
        GRPO objective:
        - Sample G rollouts per scene
        - Compute advantage: A_i = (r_i - mean(r)) / std(r)
        - Loss = -E[log π(a|s) * A] + KL_penalty
        """
        rewards = torch.tensor([r.reward for r in rollouts], device=self.device)
        
        # Compute advantages (group-relative)
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        
        # Compute policy loss
        total_loss = 0.0
        total_kl = 0.0
        valid_rollouts = 0
        
        self.policy_model.train()
        
        for rollout, advantage in zip(rollouts, advantages):
            # Forward pass through policy (trainable)
            features_batch = {k: v.unsqueeze(0) if v.dim() > 0 else v for k, v in rollout.features.items()}
            policy_output = self.policy_model(features_batch, targets=None)
            
            # Get log probs from policy
            policy_logits = policy_output.get('ego_logits')  # [1, T, V]
            if policy_logits is None or rollout.tokens is None:
                continue
                
            policy_log_probs = F.log_softmax(policy_logits / self.temperature, dim=-1)
            # Gather log probs for sampled tokens
            token_log_probs = policy_log_probs[0, torch.arange(len(rollout.tokens)), rollout.tokens]
            
            # Forward pass through reference (frozen)
            with torch.no_grad():
                ref_output = self.reference_model(features_batch, targets=None)
                ref_logits = ref_output.get('ego_logits')
                ref_log_probs = F.log_softmax(ref_logits / self.temperature, dim=-1)
                ref_token_log_probs = ref_log_probs[0, torch.arange(len(rollout.tokens)), rollout.tokens]
            
            # KL penalty per token
            kl_per_token = token_log_probs - ref_token_log_probs
            kl_loss = kl_per_token.sum()
            
            # Policy gradient loss
            policy_loss = -(token_log_probs.sum() * advantage)
            
            # Combined loss
            loss = policy_loss + self.kl_coef * kl_loss
            
            total_loss += loss
            total_kl += kl_loss.item()
            valid_rollouts += 1
        
        if valid_rollouts == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                'grpo_loss': 0.0,
                'mean_reward': mean_reward.item(),
                'std_reward': std_reward.item(),
                'kl_div': 0.0
            }
        
        avg_loss = total_loss / valid_rollouts
        avg_kl = total_kl / valid_rollouts
        
        metrics = {
            'grpo_loss': avg_loss.item(),
            'mean_reward': mean_reward.item(),
            'std_reward': std_reward.item(),
            'kl_div': avg_kl,
            'max_reward': rewards.max().item(),
            'min_reward': rewards.min().item(),
        }
        
        return avg_loss, metrics
    
    def training_step(self, batch, batch_idx):
        """Training step with GRPO."""
        features, targets, metric_cache = batch
        
        # Sample rollouts (no gradient)
        rollouts = self.sample_rollouts(features, metric_cache, self.group_size)
        
        # Compute GRPO loss
        loss, metrics = self.compute_grpo_loss(rollouts, features)
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f'train/{key}', value, on_step=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr * 0.1
        )
        
        return [optimizer], [scheduler]
