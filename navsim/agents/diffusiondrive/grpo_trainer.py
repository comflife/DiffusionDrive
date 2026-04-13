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

Fixes applied (v2):
1. [Bug1] sum→mean for log_prob: prevents T×loss explosion (was 200~-40)
2. [Bug2] Teacher-forced log prob recomputation via compute_token_log_probs()
          (AR inference mode used wrong context — own predictions vs rollout tokens)
3. [Bug3] Fixed dim handling: ego_tokens is [B,M,T] (dim3), not [M,T] (dim2)
4. [Bug4] PPO importance ratio w/ clipping using stored old log_probs
5. [Bug5] Advantage std clamped (min=1e-3) + advantage clipped to [-5, 5]
6. [Bug6] Batched forward pass: single backbone call for all G rollouts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import pytorch_lightning as pl

from navsim.agents.diffusiondrive.transfuser_model_ar import V2TransfuserModelAR
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer


@dataclass
class GRPORollout:
    """Single rollout trajectory with reward."""
    features: Dict[str, torch.Tensor]
    tokens: torch.Tensor      # [T]   predicted token indices (mode 0)
    trajectory: torch.Tensor  # [T, 3] predicted trajectory (x, y, heading)
    reward: float             # PDM score
    log_probs: torch.Tensor   # [T]   OLD log probs from sampling-time logits


class GRPOTrainer(pl.LightningModule):
    """
    GRPO Trainer for DiffusionDrive-AR.

    Key components:
    - Policy model  : Finetuned AR decoder
    - Reference model: Frozen pretrained (for KL penalty)
    - Group sampling : Multiple rollouts per scene
    - Advantage      : Normalised relative performance within group
    - PPO clipping   : Prevents excessively large updates
    """

    def __init__(
        self,
        config: TransfuserConfig,
        checkpoint_path: Optional[str] = None,
        metric_cache_path: Optional[str] = None,
        lr: float = 1e-5,
        group_size: int = 8,    # number of rollouts per scene
        kl_coef: float = 0.01,  # KL penalty coefficient
        temperature: float = 1.0,  # sampling temperature
        clip_eps: float = 0.2,  # PPO clipping epsilon  ← NEW
        max_grad_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.temperature = temperature
        self.clip_eps = clip_eps          # ← NEW
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
        self.scorer:    Optional[PDMScorer]    = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _load_pretrained(self, checkpoint_path: str, model=None):
        """Load pretrained weights."""
        model = model or self.policy_model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = checkpoint.get('state_dict', checkpoint)

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
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    # ------------------------------------------------------------------
    # Setup (PDM scorer / metric cache)
    # ------------------------------------------------------------------

    def setup(self, stage: str):
        if self.simulator is None:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from pathlib import Path

            # Bug fix: use absolute path based on __file__ so Hydra's CWD change
            # (outputs/date/time/) doesn't break the config lookup.
            scoring_cfg_path = (
                Path(__file__).resolve().parent.parent.parent
                / 'planning/script/config/pdm_scoring/default_scoring_parameters.yaml'
            )
            if not scoring_cfg_path.exists():
                raise FileNotFoundError(
                    f"PDM scoring config not found: {scoring_cfg_path}\n"
                    f"  (called from __file__={__file__})"
                )
            scoring_cfg = OmegaConf.load(scoring_cfg_path)
            self.simulator = instantiate(scoring_cfg.simulator)
            self.scorer    = instantiate(scoring_cfg.scorer)

            from navsim.common.dataloader import MetricCacheLoader
            metric_cache_path = getattr(
                self.hparams, 'metric_cache_path', '/data2/byounggun/metric_cache'
            )
            self.metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))

    def _load_metric_cache(self, token: str):
        import lzma, pickle
        path = self.metric_cache_loader.metric_cache_paths[token]
        with lzma.open(path, 'rb') as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.policy_model(features, targets=None)

    # ------------------------------------------------------------------
    # Rollout sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_rollouts(
        self,
        features: Dict[str, torch.Tensor],
        metric_cache,
        group_size: Optional[int] = None,
    ) -> List[GRPORollout]:
        """
        Sample G rollouts for a single scene.

        Stores OLD log_probs (captured from sampling-time logits) so that
        compute_grpo_loss can compute the importance ratio π_new / π_old.
        """
        group_size = group_size or self.group_size
        rollouts: List[GRPORollout] = []

        self.policy_model.eval()

        for _ in range(group_size):
            output = self.policy_model(
                features, targets=None, temperature=self.temperature
            )

            trajectory = output['trajectory']   # [B, T, 3]
            tokens     = output.get('ego_tokens')  # [B, M, T]  ← dim 3!
            ego_logits = output.get('ego_logits')  # [B, T, V]

            # --- trajectory ---
            if trajectory.dim() == 3:
                trajectory = trajectory[0]   # → [T, 3]

            # --- tokens: [B, M, T] → [T]  (Bug 3 fix: was checking dim==2) ---
            if tokens is not None:
                if tokens.dim() == 3:
                    tokens = tokens[0, 0]   # batch-0, mode-0 → [T]
                elif tokens.dim() == 2:
                    tokens = tokens[0]      # mode-0 → [T]

            # --- logits: [B, T, V] → [T, V] ---
            if ego_logits is not None and ego_logits.dim() == 3:
                ego_logits = ego_logits[0]  # → [T, V]

            # --- OLD log_probs from sampling-time logits ---
            if ego_logits is not None and tokens is not None:
                T_len = tokens.shape[0]
                log_probs_dist = F.log_softmax(ego_logits, dim=-1)  # [T, V]
                token_log_probs = log_probs_dist[
                    torch.arange(T_len, device=ego_logits.device), tokens
                ]  # [T]
            else:
                T_len = trajectory.shape[0]
                token_log_probs = torch.zeros(T_len, dtype=torch.float32,
                                              device=trajectory.device)

            reward = self._compute_pdm_reward(trajectory, metric_cache)

            rollouts.append(GRPORollout(
                features=features,
                tokens=tokens,
                trajectory=trajectory,
                reward=reward,
                log_probs=token_log_probs,
            ))

        return rollouts

    # ------------------------------------------------------------------
    # PDM reward
    # ------------------------------------------------------------------

    def _compute_pdm_reward(self, trajectory: torch.Tensor, metric_cache) -> float:
        """
        Compute PDM score for a predicted trajectory.

        trajectory : [T, 3]  (x, y, heading) in ego frame.
                     NOTE: The AR model sets heading=0 for all steps,
                     which is incorrect for curved paths.  We fix this by
                     estimating heading from consecutive (x, y) positions.

        Fixes applied
        -------------
        Fix A: Heading estimated via atan2(dy, dx) instead of always 0.
               Incorrect heading causes wrong ego-box orientation in PDM
               simulation → false collision / drivable-area scores on curves.
        Fix B: TrajectorySampling derived from simulator.proposal_sampling
               instead of hardcoded (time_horizon=4, interval_length=0.1).
        Fix C: .float() before F.interpolate to guard against fp16 inputs.
        Fix D: poses cast to np.float32 matching Trajectory dtype spec.
        """
        if self.simulator is None or self.scorer is None:
            return 0.0
        try:
            from navsim.common.dataclasses import Trajectory
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

            prop = self.simulator.proposal_sampling          # TrajectorySampling
            target_num_poses  = prop.num_poses               # 40
            target_interval   = prop.interval_length         # 0.1

            # Fix C: ensure float32 for interpolation (guards fp16 AMP)
            trajectory = trajectory.float()

            model_num_poses = trajectory.shape[0]
            if model_num_poses != target_num_poses:
                traj_perm   = trajectory.permute(1, 0).unsqueeze(0)   # [1, 3, T]
                traj_interp = F.interpolate(
                    traj_perm, size=target_num_poses,
                    mode='linear', align_corners=True,
                )
                trajectory_3d = traj_interp.squeeze(0).permute(1, 0)  # [target, 3]
            else:
                trajectory_3d = trajectory   # [target, 3]

            # Fix A: estimate heading from consecutive (x, y) positions.
            # The AR model stores heading=0 (column 2) because the codebook
            # only encodes (dx, dy) step-wise displacements.  Using heading=0
            # means relative_to_absolute_poses keeps the ego facing the INITIAL
            # direction forever — wrong for curves.  atan2(dy, dx) gives a
            # physically meaningful heading for each waypoint.
            pos_xy = trajectory_3d[:, :2]                         # [N, 2]
            if pos_xy.shape[0] > 1:
                diffs      = pos_xy[1:] - pos_xy[:-1]             # [N-1, 2]
                headings_t = torch.atan2(diffs[:, 1], diffs[:, 0])  # [N-1]
                # Repeat first heading so shape stays [N]
                headings_t = torch.cat([headings_t[:1], headings_t], dim=0)  # [N]
            else:
                headings_t = torch.zeros(pos_xy.shape[0], device=trajectory_3d.device)

            trajectory_3d = torch.stack(
                [pos_xy[:, 0], pos_xy[:, 1], headings_t], dim=1
            )  # [N, 3]

            # Fix B: build TrajectorySampling from simulator params (not hardcoded)
            traj_sampling = TrajectorySampling(
                num_poses=target_num_poses,
                interval_length=target_interval,
            )

            # Fix D: cast to float32 as required by Trajectory.__post_init__
            model_trajectory = Trajectory(
                poses=trajectory_3d.cpu().numpy().astype(np.float32),
                trajectory_sampling=traj_sampling,
            )

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=model_trajectory,
                future_sampling=prop,
                simulator=self.simulator,
                scorer=self.scorer,
            )
            return float(pdm_result.score)
        except Exception as e:
            print(f"[WARN] PDM scoring failed: {e}")
            import traceback; traceback.print_exc()
            return 0.0

    # ------------------------------------------------------------------
    # GRPO loss  (all 6 bugs fixed here)
    # ------------------------------------------------------------------

    def compute_grpo_loss(
        self,
        rollouts: List[GRPORollout],
        features: Dict[str, torch.Tensor],   # kept for API compatibility
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss with PPO-style token-level clipping.

        Bug-fix summary
        ---------------
        Bug 1 : .sum() → .mean() for T-step log probs  (prevents T×explosion)
        Bug 2 : compute_token_log_probs() for teacher-forced log probs
                (AR inference would condition on wrong—model-own—context)
        Bug 3 : dim-3 token handling fixed in sample_rollouts (see above)
        Bug 4 : PPO importance ratio using stored old_log_probs
        Bug 5 : std clamped to ≥1e-3 and advantages clipped to [-5, 5]
        Bug 6 : single batched backbone call for all G rollouts
        """
        rewards     = torch.tensor([r.reward for r in rollouts], device=self.device)
        mean_reward = rewards.mean()

        # Bug 5: clamp std and clip advantages
        std_reward  = rewards.std(unbiased=False).clamp(min=1e-3)
        advantages  = ((rewards - mean_reward) / std_reward).clamp(-5.0, 5.0)

        # Filter rollouts that have valid tokens / log_probs
        valid_idx = [
            i for i, r in enumerate(rollouts)
            if r.tokens is not None
            and r.log_probs is not None
            and r.tokens.numel() > 0
        ]

        if not valid_idx:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                'grpo_loss': 0.0, 'policy_loss': 0.0,
                'mean_reward': mean_reward.item(),
                'std_reward':  std_reward.item(),
                'kl_div': 0.0,
            }

        G = len(valid_idx)

        # Bug 6: single batched backbone call — expand 1 scene to G copies
        scene_features = rollouts[0].features
        base_features  = {
            k: v[0:1] if (isinstance(v, torch.Tensor) and v.dim() > 0) else v
            for k, v in scene_features.items()
        }
        batched_features = {
            k: v.expand(G, *v.shape[1:]).contiguous()
               if isinstance(v, torch.Tensor) else v
            for k, v in base_features.items()
        }

        # [G, T]
        all_tokens        = torch.stack([rollouts[i].tokens for i in valid_idx]).to(self.device)
        all_old_log_probs = torch.stack(
            [rollouts[i].log_probs.detach() for i in valid_idx]
        ).to(self.device)   # [G, T]
        valid_advantages  = advantages[valid_idx]  # [G]

        # Bug 2 fix: teacher-forced log probs (not AR-inference mode)
        # compute_token_log_probs() conditions each step t on the ROLLOUT tokens
        # a_0,...,a_{t-1} (via BOS-shifted teacher forcing), not on model's own
        # predictions.  This gives the correct π_θ(a_t | s, a_{<t}).
        self.policy_model.train()
        new_token_log_probs, _ = self.policy_model.compute_token_log_probs(
            batched_features, all_tokens
        )  # [G, T]

        with torch.no_grad():
            ref_token_log_probs, _ = self.reference_model.compute_token_log_probs(
                batched_features, all_tokens
            )  # [G, T]

        # Bug 4: PPO importance ratio w/ clipping
        ratio         = torch.exp(new_token_log_probs - all_old_log_probs)  # [G, T]
        ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        # [G] → [G, T] for token-level weighting
        adv_expanded = valid_advantages.unsqueeze(1).expand_as(ratio)

        # Bug 1 fix: .mean() over tokens (not .sum())
        pg_loss = -torch.min(ratio * adv_expanded, ratio_clipped * adv_expanded).mean()

        # KL penalty: per-token mean  (Bug 1 fix)
        kl_loss = (new_token_log_probs - ref_token_log_probs).mean()

        total_loss = pg_loss + self.kl_coef * kl_loss

        metrics = {
            'grpo_loss':      total_loss.item(),
            'policy_loss':    pg_loss.item(),
            'kl_div':         kl_loss.item(),
            'mean_reward':    mean_reward.item(),
            'std_reward':     std_reward.item(),
            'max_reward':     rewards.max().item(),
            'min_reward':     rewards.min().item(),
            'mean_ratio':     ratio.mean().item(),
            'mean_advantage': valid_advantages.mean().item(),
        }

        return total_loss, metrics

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        features, targets, token = batch

        metric_cache = self._load_metric_cache(token)

        # Rollout phase (no grad) — stores old log_probs
        rollouts = self.sample_rollouts(features, metric_cache, self.group_size)

        # Loss phase (with grad) — teacher-forced log probs
        loss, metrics = self.compute_grpo_loss(rollouts, features)

        for key, value in metrics.items():
            self.log(f'train/{key}', value, on_step=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr * 0.1,
        )
        return [optimizer], [scheduler]
