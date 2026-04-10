"""
Custom RL Trainer for DiffusionDrive-AR with Multi-Scale Advantage.

Combines:
1. GSPO-style (Qwen): Sequence-level importance ratio aggregation
   - Avoids token-level noise accumulation
2. Multi-scale advantage: Sequence-level (global) + Token-level (local)
   - Captures both overall trajectory quality and local maneuver changes

Reference:
- GSPO: https://arxiv.org/abs/2410.01203 (Qwen2.5 technical report)
- GRPO: https://arxiv.org/abs/2402.03300
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


@dataclass
class MultiScaleRollout:
    """Rollout with both sequence and token-level information."""
    features: Dict[str, torch.Tensor]
    tokens: torch.Tensor  # [T] predicted token indices
    trajectory: torch.Tensor  # [T, 2] predicted trajectory
    
    # Sequence-level reward (PDMS)
    sequence_reward: float
    
    # Token-level rewards (optional, for local advantage)
    token_rewards: Optional[torch.Tensor] = None  # [T] or None
    
    # Log probs
    log_probs: torch.Tensor  # [T] per-token log probs
    
    # Importance: sequence-level aggregated (GSPO style)
    sequence_log_prob: float  # sum or mean of token log probs


class MultiScaleRLTrainer(pl.LightningModule):
    """
    Custom RL Trainer with Multi-Scale Advantage and GSPO-style aggregation.
    
    Key innovations:
    1. GSPO Aggregation: Token-level ratios → Sequence-level (reduces variance)
    2. Multi-Scale Advantage:
       - Global (Sequence): Overall PDMS score
       - Local (Token): Detect local changes (e.g., evasive maneuvers)
    3. Combined Loss: Balances global trajectory quality + local precision
    """
    
    def __init__(
        self,
        config: TransfuserConfig,
        checkpoint_path: Optional[str] = None,
        lr: float = 1e-5,
        group_size: int = 8,
        kl_coef: float = 0.01,
        temperature: float = 1.0,
        
        # Multi-scale advantage weights
        global_advantage_weight: float = 0.7,  # Sequence-level (PDMS)
        local_advantage_weight: float = 0.3,   # Token-level (local changes)
        
        # Local advantage computation
        local_window_size: int = 3,  # Window for local advantage
        use_token_level_kl: bool = True,  # Whether to use token-level KL
        
        # GSPO aggregation method: "mean", "sum", "last"
        gspo_aggregation: str = "mean",
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.temperature = temperature
        
        self.global_weight = global_advantage_weight
        self.local_weight = local_advantage_weight
        self.local_window = local_window_size
        self.use_token_kl = use_token_level_kl
        self.gspo_agg = gspo_aggregation
        
        # Policy model (trainable)
        self.policy_model = V2TransfuserModelAR(config)
        
        # Load pretrained
        if checkpoint_path:
            self._load_pretrained(checkpoint_path)
        
        # Reference model (frozen)
        self.reference_model = V2TransfuserModelAR(config)
        if checkpoint_path:
            self._load_pretrained(checkpoint_path, model=self.reference_model)
        self._freeze_model(self.reference_model)
        
    def _load_pretrained(self, checkpoint_path: str, model=None):
        """Load pretrained weights."""
        model = model or self.policy_model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove prefixes
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
    
    @torch.no_grad()
    def sample_rollouts(
        self, 
        features: Dict[str, torch.Tensor],
        metric_cache,
        compute_local_rewards: bool = True,
    ) -> List[MultiScaleRollout]:
        """
        Sample rollouts with both sequence and token-level information.
        """
        rollouts = []
        self.policy_model.eval()
        
        for _ in range(self.group_size):
            with torch.no_grad():
                output = self.policy_model(
                    features, 
                    targets=None, 
                    temperature=self.temperature,
                    return_log_probs=True  # Need per-token log probs
                )
                
            trajectory = output['trajectory']
            tokens = output.get('ego_tokens')
            log_probs = output.get('ego_log_probs')  # [T] per-token
            
            if trajectory.dim() == 3:
                trajectory = trajectory[0]
            if tokens is not None and tokens.dim() == 2:
                tokens = tokens[0]
            if log_probs is not None and log_probs.dim() == 2:
                log_probs = log_probs[0]
            
            # Sequence-level reward (PDMS)
            sequence_reward = self._compute_sequence_reward(trajectory, metric_cache)
            
            # Token-level rewards (for local advantage)
            token_rewards = None
            if compute_local_rewards and log_probs is not None:
                token_rewards = self._compute_token_rewards(
                    trajectory, tokens, metric_cache
                )
            
            # GSPO: Aggregate token log probs to sequence level
            if log_probs is not None:
                if self.gspo_agg == "mean":
                    sequence_log_prob = log_probs.mean().item()
                elif self.gspo_agg == "sum":
                    sequence_log_prob = log_probs.sum().item()
                elif self.gspo_agg == "last":
                    sequence_log_prob = log_probs[-1].item()
                else:
                    sequence_log_prob = log_probs.mean().item()
            else:
                sequence_log_prob = 0.0
                log_probs = torch.zeros_like(tokens, dtype=torch.float32)
            
            rollout = MultiScaleRollout(
                features=features,
                tokens=tokens,
                trajectory=trajectory,
                sequence_reward=sequence_reward,
                token_rewards=token_rewards,
                log_probs=log_probs,
                sequence_log_prob=sequence_log_prob,
            )
            rollouts.append(rollout)
            
        return rollouts
    
    def _compute_sequence_reward(self, trajectory: torch.Tensor, metric_cache) -> float:
        """Compute sequence-level reward (PDMS score)."""
        # Placeholder: actual implementation needs simulator/scorer
        # For now, return dummy or cached value
        return 0.0
    
    def _compute_token_rewards(
        self, 
        trajectory: torch.Tensor, 
        tokens: torch.Tensor,
        metric_cache
    ) -> torch.Tensor:
        """
        Compute token-level rewards for local advantage.
        
        Ideas:
        1. Trajectory smoothness penalty per step
        2. Collision proximity at each step
        3. Comfort metrics (jerk, acceleration)
        """
        T = len(tokens)
        token_rewards = torch.zeros(T)
        
        # Local smoothness: penalize sudden direction changes
        if T > 1:
            velocities = trajectory[1:] - trajectory[:-1]  # [T-1, 2]
            if T > 2:
                accelerations = velocities[1:] - velocities[:-1]  # [T-2, 2]
                jerk = torch.norm(accelerations, dim=1)
                # Assign jerk penalty to middle tokens
                for i in range(len(jerk)):
                    token_rewards[i+1] -= jerk[i].item() * 0.1
        
        # Progress reward: closer to goal at later tokens
        if hasattr(metric_cache, 'goal_position'):
            goal = metric_cache.goal_position
            for t in range(T):
                dist_to_goal = torch.norm(trajectory[t] - goal)
                # Later tokens should be closer to goal
                token_rewards[t] += 1.0 / (dist_to_goal + 1.0)
        
        return token_rewards
    
    def compute_multi_scale_advantages(
        self,
        rollouts: List[MultiScaleRollout]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both sequence-level and token-level advantages.
        
        Returns:
            global_advantages: [G] sequence-level advantages
            local_advantages: [G, T] token-level advantages (or None)
        """
        G = len(rollouts)
        
        # Sequence-level advantages
        seq_rewards = torch.tensor([r.sequence_reward for r in rollouts])
        seq_mean = seq_rewards.mean()
        seq_std = seq_rewards.std() + 1e-8
        global_advantages = (seq_rewards - seq_mean) / seq_std
        
        # Token-level advantages
        local_advantages = None
        if rollouts[0].token_rewards is not None:
            T = len(rollouts[0].token_rewards)
            token_rewards_all = torch.stack([r.token_rewards for r in rollouts])  # [G, T]
            
            # Compute advantage per token position across group
            token_mean = token_rewards_all.mean(dim=0)  # [T]
            token_std = token_rewards_all.std(dim=0) + 1e-8  # [T]
            local_advantages = (token_rewards_all - token_mean.unsqueeze(0)) / token_std.unsqueeze(0)
            
            # Also add sequence advantage broadcast to all tokens for global context
            global_broadcast = global_advantages.unsqueeze(1).expand(-1, T)  # [G, T]
            
            # Combine: weighted sum of global context + local detail
            local_advantages = (
                self.global_weight * global_broadcast + 
                self.local_weight * local_advantages
            )
        
        return global_advantages, local_advantages
    
    def compute_gspo_loss(
        self,
        rollouts: List[MultiScaleRollout],
        global_advantages: torch.Tensor,
        local_advantages: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GSPO-style loss with sequence-level importance ratio.
        
        Key difference from GRPO:
        - GRPO: prod_t (π_θ/π_ref) for each token → high variance
        - GSPO: aggregate tokens first, then ratio → lower variance
        """
        total_loss = 0.0
        total_seq_kl = 0.0
        total_token_kl = 0.0
        valid_rollouts = 0
        
        self.policy_model.train()
        
        for i, (rollout, global_adv) in enumerate(zip(rollouts, global_advantages)):
            # Forward pass (trainable)
            features_batch = {
                k: v.unsqueeze(0) if v.dim() > 0 else v 
                for k, v in rollout.features.items()
            }
            policy_output = self.policy_model(features_batch, targets=None)
            
            # Get current policy log probs
            policy_logits = policy_output.get('ego_logits')  # [1, T, V]
            if policy_logits is None:
                continue
            
            policy_log_probs = F.log_softmax(policy_logits / self.temperature, dim=-1)
            token_log_probs = policy_log_probs[0, torch.arange(len(rollout.tokens)), rollout.tokens]
            
            # GSPO: Aggregate to sequence level
            if self.gspo_agg == "mean":
                current_seq_log_prob = token_log_probs.mean()
            elif self.gspo_agg == "sum":
                current_seq_log_prob = token_log_probs.sum()
            else:
                current_seq_log_prob = token_log_probs.mean()
            
            # Reference model (frozen)
            with torch.no_grad():
                ref_output = self.reference_model(features_batch, targets=None)
                ref_logits = ref_output.get('ego_logits')
                ref_log_probs = F.log_softmax(ref_logits / self.temperature, dim=-1)
                ref_token_log_probs = ref_log_probs[0, torch.arange(len(rollout.tokens)), rollout.tokens]
                
                if self.gspo_agg == "mean":
                    ref_seq_log_prob = ref_token_log_probs.mean()
                elif self.gspo_agg == "sum":
                    ref_seq_log_prob = ref_token_log_probs.sum()
                else:
                    ref_seq_log_prob = ref_token_log_probs.mean()
            
            # Sequence-level ratio and loss
            seq_ratio = torch.exp(current_seq_log_prob - ref_seq_log_prob)
            seq_ratio_clipped = torch.clamp(seq_ratio, 0.2, 5.0)  # PPO-style clipping
            
            # Surrogate loss with sequence-level advantage
            seq_loss1 = -seq_ratio * global_adv
            seq_loss2 = -seq_ratio_clipped * global_adv
            seq_loss = torch.max(seq_loss1, seq_loss2)
            
            # Sequence-level KL penalty
            seq_kl = current_seq_log_prob - ref_seq_log_prob
            
            # Token-level loss (optional, for local precision)
            token_loss = 0.0
            token_kl = 0.0
            if local_advantages is not None and self.local_weight > 0:
                local_adv = local_advantages[i]  # [T]
                
                # Per-token policy gradient with local advantage
                token_pg_loss = -(token_log_probs * local_adv).sum()
                
                # Per-token KL
                if self.use_token_kl:
                    per_token_kl = token_log_probs - ref_token_log_probs
                    token_kl = per_token_kl.sum()
                
                token_loss = token_pg_loss
            
            # Combined loss
            loss = (
                seq_loss 
                + self.kl_coef * seq_kl
                + self.local_weight * token_loss
                + (self.kl_coef * 0.1 if self.use_token_kl else 0) * token_kl
            )
            
            total_loss += loss
            total_seq_kl += seq_kl.item()
            total_token_kl += token_kl if isinstance(token_kl, float) else token_kl.item()
            valid_rollouts += 1
        
        if valid_rollouts == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                'total_loss': 0.0,
                'seq_kl': 0.0,
                'token_kl': 0.0,
            }
        
        avg_loss = total_loss / valid_rollouts
        
        metrics = {
            'total_loss': avg_loss.item(),
            'seq_kl': total_seq_kl / valid_rollouts,
            'token_kl': total_token_kl / valid_rollouts,
            'mean_global_adv': global_advantages.mean().item(),
            'std_global_adv': global_advantages.std().item(),
        }
        
        if local_advantages is not None:
            metrics['mean_local_adv'] = local_advantages.mean().item()
            metrics['std_local_adv'] = local_advantages.std().item()
        
        return avg_loss, metrics
    
    def training_step(self, batch, batch_idx):
        """Training step with multi-scale RL."""
        features, targets, metric_cache = batch
        
        # Sample rollouts
        rollouts = self.sample_rollouts(
            features, 
            metric_cache, 
            compute_local_rewards=(self.local_weight > 0)
        )
        
        # Compute multi-scale advantages
        global_advantages, local_advantages = self.compute_multi_scale_advantages(rollouts)
        
        # Compute GSPO loss
        loss, metrics = self.compute_gspo_loss(rollouts, global_advantages, local_advantages)
        
        # Log
        for key, value in metrics.items():
            self.log(f'train/{key}', value, on_step=True, prog_bar=True)
        
        # Log reward stats
        rewards = torch.tensor([r.sequence_reward for r in rollouts])
        self.log('train/mean_reward', rewards.mean(), on_step=True)
        self.log('train/max_reward', rewards.max(), on_step=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=self.hparams.lr * 0.01
        )
        
        return [optimizer], [scheduler]
