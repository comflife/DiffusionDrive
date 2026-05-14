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

Fixes applied (v3):
7. [Bug7] log_softmax now uses the SAMPLING temperature for old/new/ref alike,
          so the importance ratio correctly tracks π(·|T) → π'(·|T). Previously
          old/new were at unit T while samples came from π(·|T<1), biasing PPO.
8. [Bug8] PDM reward no longer overrides the model's predicted heading with
          atan2(pos_diffs). For step_corners / heading_head=true models the
          override discarded the actual heading channel; now we trust the
          model's heading. (Legacy step_delta + heading_head=false models
          already produced atan2-equivalent heading inside _build_trajectory,
          so the override was redundant in that case too.)
9. [Bug9] PDM-scoring failures are tracked instead of silently returning 0:
          `pdm_failure_streak` counts consecutive failures and raises after
          PDM_FAIL_RAISE_AFTER (default 50) — long silent streaks used to
          collapse advantages to ~0 and stall training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import pytorch_lightning as pl

from navsim.agents.diffusiondrive.transfuser_model_ar import V2TransfuserModelAR
from navsim.agents.diffusiondrive.transfuser_agent_ar import TransfuserAgentAR
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
        clip_eps: float = 0.2,  # PPO clipping epsilon (GRPO token-level)
        algorithm: str = 'grpo',     # 'grpo' | 'dr_grpo' | 'gspo' | 'gspo_token' | 'grpo_plus'
        clip_eps_seq: float = 4e-4,  # sequence-level clipping eps for GSPO / GRPO+
        token_attention_alpha: float = 0.5,  # GRPO+: blend (1-α)·GSPO + α·token-attn
        max_grad_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if algorithm not in ('grpo', 'dr_grpo', 'gspo', 'gspo_token', 'grpo_plus', 'dr_gspo', 'dr_grpo_plus'):
            raise ValueError(
                "algorithm must be one of 'grpo', 'dr_grpo', 'gspo', 'gspo_token', "
                "'grpo_plus', 'dr_gspo', 'dr_grpo_plus'; "
                f"got {algorithm!r}"
            )
        if not (0.0 <= float(token_attention_alpha) <= 1.0):
            raise ValueError(
                f"token_attention_alpha must be in [0, 1]; got {token_attention_alpha}"
            )

        self.config = config
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.temperature = temperature
        self.clip_eps = clip_eps
        self.algorithm = algorithm
        self.clip_eps_seq = clip_eps_seq
        self.token_attention_alpha = float(token_attention_alpha)
        self.max_grad_norm = max_grad_norm

        # Policy model (trainable)
        self.policy_model = V2TransfuserModelAR(config)

        # Load pretrained weights if provided
        if checkpoint_path:
            self._load_pretrained(checkpoint_path)

        # Reference model (frozen, for KL penalty).
        # We construct it then copy policy's state_dict in. Building each model
        # separately would give the AR-specific freshly-random modules
        # (ego_token_emb, ego_ctx_proj, agent_encoder, step_agent_proj, t_attn,
        # ...) DIFFERENT random draws between policy and reference — at step 0
        # the categorical KL would already be > 0, contaminating the KL signal
        # and pulling the policy back toward a different random init instead
        # of the warm-started one. Mirroring policy guarantees KL == 0 at init.
        self.reference_model = V2TransfuserModelAR(config)
        self.reference_model.load_state_dict(self.policy_model.state_dict())
        self._freeze_model(self.reference_model)

        # PDM components for reward computation
        self.simulator: Optional[PDMSimulator] = None
        self.scorer:    Optional[PDMScorer]    = None

        # PDM failure tracking (Bug9): consecutive failures cause silent
        # advantage collapse, so raise once a long streak builds up.
        self.pdm_failure_streak: int = 0
        self.pdm_failure_total:  int = 0
        self.PDM_FAIL_RAISE_AFTER: int = 50

    @staticmethod
    def _safe_temperature(t: float) -> float:
        """Clamp sampling temperature to a strictly positive value.

        T == 0 means greedy sampling — there is no well-defined importance
        ratio in that case, so we fall back to T = 1 for log-prob computations
        (the gradient is then taken w.r.t. the unit-T policy, which is still
        well-defined; the user just shouldn't expect PPO-style correction).
        """
        return max(float(t), 1e-6) if float(t) > 0 else 1.0

    @staticmethod
    def _gather_token_log_probs(
        log_probs_all: torch.Tensor,   # [..., T, V]
        tokens:        torch.Tensor,   # [..., T]
    ) -> torch.Tensor:
        """Index log_probs_all by tokens to get [..., T] log probs."""
        return torch.gather(log_probs_all, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _load_pretrained(self, checkpoint_path: str, model=None):
        """Load pretrained weights, including the diff_decoder→AR remap.

        The diff_decoder→AR remap (TransfuserAgentAR._remap_diff_decoder_to_ar)
        warm-starts the AR head's bev_deform_attn / e2a_attn / ego_attn / ffn /
        norms from the original 88.1 PDMS DiffusionDrive trajectory head. Without
        this remap, both policy and reference would each receive a fresh random
        AR head — they'd start out DIFFERENT (different RNG draws), inflating
        KL at step 0 and discarding the pretrained cross-attention knowledge.
        """
        model = model or self.policy_model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = checkpoint.get('state_dict', checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('reference_model.'):
                continue   # GRPO ckpts: reference is a frozen copy, skip
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

        # Apply diff_decoder→AR remap (no-op for ckpts that already use the
        # AR head structure, e.g. SFT checkpoints from v3..v6).
        new_state_dict = TransfuserAgentAR._remap_diff_decoder_to_ar(new_state_dict)

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        ar_head_total = sum(1 for k in model.state_dict() if k.startswith('_trajectory_head.'))
        ar_head_missing = sum(1 for k in missing if k.startswith('_trajectory_head.'))
        print(
            f"GRPO load: AR head warm-started "
            f"{ar_head_total - ar_head_missing}/{ar_head_total} tensors, "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys"
        )

    def _freeze_model(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    @contextmanager
    def _temporary_eval_mode(self, model: nn.Module):
        """Temporarily disable dropout while preserving autograd."""
        was_training = model.training
        model.eval()
        try:
            yield
        finally:
            model.train(was_training)

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
                    # GRPO currently optimizes the primary planning mode only.
                    tokens = tokens[0, 0]   # batch-0, mode-0 → [T]
                elif tokens.dim() == 2:
                    tokens = tokens[0]      # mode-0 → [T]

            # --- logits: [B, T, V] → [T, V] ---
            if ego_logits is not None and ego_logits.dim() == 3:
                ego_logits = ego_logits[0]  # → [T, V]

            # --- OLD log_probs from sampling-time logits ---
            # Bug7 fix: scale logits by the sampling temperature so the stored
            # log_prob matches the actual sampling distribution π(a|s, T).
            # Without this, importance ratio π_new(a|T)/π_old(a|T=1) is biased.
            if ego_logits is not None and tokens is not None:
                T_len = tokens.shape[0]
                T_sample = self._safe_temperature(self.temperature)
                log_probs_dist = F.log_softmax(ego_logits / T_sample, dim=-1)  # [T, V]
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

        trajectory : [T, 3]  (x, y, heading) in ego frame, as returned by
                     V2TransfuserModelAR. step_corners and trajectory_corners
                     codebooks already produce a real heading; the legacy
                     step_delta path with heading_head=False also returns
                     atan2(local_delta) heading, which is sensible.

        Fixes applied
        -------------
        Fix B: TrajectorySampling derived from simulator.proposal_sampling
               instead of hardcoded (time_horizon=4, interval_length=0.1).
        Fix C: .float() before F.interpolate to guard against fp16 inputs.
        Fix D: poses cast to np.float32 matching Trajectory dtype spec.
        Fix E (Bug8): Trust the model's predicted heading instead of
               overwriting it with atan2(pos_diffs). The previous override
               discarded the heading channel entirely — for step_corners or
               heading_head=true models that channel is meaningful, and for
               step_delta the model already stores atan2(local_delta) so the
               override was redundant. As a defensive last resort we still
               recompute heading via atan2 if the model channel is exactly
               zero everywhere (e.g. an old step_delta model with no heading
               head was loaded).
        Bug9: PDM-scoring failures are tracked. Long silent streaks used to
              collapse advantages to ~0 and stall training, so we raise after
              `PDM_FAIL_RAISE_AFTER` consecutive failures.
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

            # Fix E (Bug8): only fall back to atan2(pos_diffs) when the model
            # truly produced no heading (all-zero channel). For step_corners /
            # heading_head=true, trust the model's heading.
            heading_channel = trajectory_3d[:, 2]
            heading_is_dead = bool(heading_channel.abs().max().item() < 1e-6)
            if heading_is_dead and trajectory_3d.shape[0] > 1:
                pos_xy = trajectory_3d[:, :2]
                diffs      = pos_xy[1:] - pos_xy[:-1]
                headings_t = torch.atan2(diffs[:, 1], diffs[:, 0])
                headings_t = torch.cat([headings_t[:1], headings_t], dim=0)
                trajectory_3d = torch.stack(
                    [pos_xy[:, 0], pos_xy[:, 1], headings_t], dim=1
                )

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
            self.pdm_failure_streak = 0
            return float(pdm_result.score)
        except Exception as e:
            self.pdm_failure_streak += 1
            self.pdm_failure_total  += 1
            print(
                f"[WARN] PDM scoring failed (streak={self.pdm_failure_streak}, "
                f"total={self.pdm_failure_total}): {e}"
            )
            import traceback; traceback.print_exc()
            if self.pdm_failure_streak >= self.PDM_FAIL_RAISE_AFTER:
                raise RuntimeError(
                    f"PDM scoring failed {self.pdm_failure_streak} times in a row "
                    f"(total {self.pdm_failure_total}). Aborting GRPO training "
                    "before silent advantage collapse stalls everything."
                )
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
        with self._temporary_eval_mode(self.policy_model):
            _, new_logits = self.policy_model.compute_token_log_probs(
                batched_features, all_tokens
            )  # _, [G, T, V]

        with torch.no_grad():
            with self._temporary_eval_mode(self.reference_model):
                _, ref_logits = self.reference_model.compute_token_log_probs(
                    batched_features, all_tokens
                )  # _, [G, T, V]

        # Bug7 fix: rebuild new/ref log_probs at SAMPLING temperature so they
        # match `all_old_log_probs` (also stored at sampling T). The default
        # token_log_probs returned by compute_token_log_probs are at unit T
        # and would inject a temperature mismatch into the importance ratio.
        T_sample = self._safe_temperature(self.temperature)
        new_log_probs_all = F.log_softmax(new_logits / T_sample, dim=-1)   # [G, T, V]
        ref_log_probs_all = F.log_softmax(ref_logits / T_sample, dim=-1)   # [G, T, V]

        new_token_log_probs = self._gather_token_log_probs(new_log_probs_all, all_tokens)  # [G, T]

        # ────────────────────────────────────────────────────────────────
        # Algorithm dispatch: GRPO (token) / GSPO (sequence) / GSPO-token
        # ────────────────────────────────────────────────────────────────
        # Per-token log-ratio Δlog π_t = log π_new(a_t) − log π_old(a_t),
        # both already at SAMPLING temperature (Bug7).
        log_ratio_token = new_token_log_probs - all_old_log_probs            # [G, T]

        if self.algorithm == 'grpo':
            # Token-level PPO clipping. ratio_t can swing wildly per token; we
            # average the clipped surrogate over the T tokens.
            ratio         = torch.exp(log_ratio_token)                       # [G, T]
            ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            adv_expanded  = valid_advantages.unsqueeze(1).expand_as(ratio)
            pg_loss       = -torch.min(ratio * adv_expanded,
                                       ratio_clipped * adv_expanded).mean()
            ratio_for_log = ratio
            clip_eps_used = self.clip_eps

        elif self.algorithm == 'dr_grpo':
            # Dr. GRPO (arXiv:2503.20783) — "GRPO Done Right".
            # Removes two normalization biases from vanilla GRPO:
            #   • std-norm in advantage  ⇒  A_i = r_i − mean(r)   (no /std)
            #     Vanilla GRPO's /std penalises groups where the policy is
            #     already strong (low variance) by squashing their gradient,
            #     and amplifies noise on high-variance groups.
            #   • length-norm 1/|o_i| in loss  ⇒  Σ_t (no /T)
            #     Biases against long correct rollouts (and toward long
            #     incorrect ones). In our setting T=8 is constant per
            #     trajectory so this is just a constant 8× scale absorbed by
            #     LR — but kept for definitional correctness.
            # Token-level PPO clipping is preserved (same as vanilla GRPO).
            raw_advantages = (rewards - mean_reward).clamp(-5.0, 5.0)        # [N], no /std
            valid_raw_adv  = raw_advantages[valid_idx]                       # [G]
            ratio         = torch.exp(log_ratio_token)                       # [G, T]
            ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            adv_expanded  = valid_raw_adv.unsqueeze(1).expand_as(ratio)
            # SUM over T (no length normalization), MEAN over G.
            pg_loss       = -torch.min(ratio * adv_expanded,
                                       ratio_clipped * adv_expanded).sum(dim=1).mean()
            ratio_for_log = ratio
            clip_eps_used = self.clip_eps

        elif self.algorithm == 'gspo':
            # Qwen GSPO (arXiv:2507.18071): sequence-level importance ratio
            # with length normalization. For our AR head, |y_i| = T = const.
            #   s_i = exp((1/T) · Σ_t Δlog π_t)
            # Clipping is applied to s_i directly, then averaged over the G
            # sequences in the group. clip_eps_seq is much tighter than GRPO's
            # token clip (paper reports 3e-4 ~ 4e-4 for sequence-level).
            log_seq_ratio   = log_ratio_token.mean(dim=-1)                   # [G]
            seq_ratio       = torch.exp(log_seq_ratio)                       # [G]
            seq_ratio_clipd = torch.clamp(seq_ratio,
                                          1.0 - self.clip_eps_seq,
                                          1.0 + self.clip_eps_seq)
            pg_loss = -torch.min(seq_ratio       * valid_advantages,
                                 seq_ratio_clipd * valid_advantages).mean()
            ratio_for_log = seq_ratio
            clip_eps_used = self.clip_eps_seq

        elif self.algorithm == 'dr_gspo':
            # Dr. GSPO: NoRD recipe applied to GSPO.
            #   • No /std in advantage  (same fix as Dr. GRPO)
            #   • Token-level PPO clip epsilon (0.2) instead of the tiny
            #     sequence-level clip (4e-4) which is designed for LLM-scale T.
            #     T=8 is far too short for 4e-4; it locks the ratio to ~1.0.
            #   • Sequence-level ratio s_i = exp(mean_t Δlog π_t) kept for
            #     consistency with GSPO formulation.
            log_seq_ratio   = log_ratio_token.mean(dim=-1)                   # [G]
            seq_ratio       = torch.exp(log_seq_ratio)                       # [G]
            seq_ratio_clipd = torch.clamp(seq_ratio,
                                          1.0 - self.clip_eps,
                                          1.0 + self.clip_eps)
            raw_advantages = (rewards - mean_reward).clamp(-5.0, 5.0)        # [N], no /std
            valid_raw_adv  = raw_advantages[valid_idx]                       # [G]
            pg_loss = -torch.min(seq_ratio       * valid_raw_adv,
                                 seq_ratio_clipd * valid_raw_adv).mean()
            ratio_for_log = seq_ratio
            clip_eps_used = self.clip_eps

        elif self.algorithm == 'gspo_token':
            # Sequence-level importance ratio in VALUE, per-token gradient.
            #   s_{i,t} = sg[s_i] · π_θ(y_t)/sg[π_θ(y_t)]  ≡ s_i (in value)
            # Lets us mix GSPO's sequence-level clipping with per-token gradient
            # routing — useful if per-token rewards/values are added later.
            log_seq_ratio_sg = log_ratio_token.detach().mean(dim=-1)         # [G]   (sg[s_i] in log-space)
            s_i_sg           = torch.exp(log_seq_ratio_sg)                   # [G]
            # token-level factor that equals 1 in value but carries gradient
            token_grad_factor = torch.exp(
                new_token_log_probs - new_token_log_probs.detach()
            )                                                                # [G, T]
            s_it          = s_i_sg.unsqueeze(1) * token_grad_factor          # [G, T]
            s_it_clipd    = torch.clamp(s_i_sg, 1.0 - self.clip_eps_seq,
                                        1.0 + self.clip_eps_seq).unsqueeze(1) * token_grad_factor
            adv_expanded  = valid_advantages.unsqueeze(1).expand_as(s_it)
            pg_loss = -torch.min(s_it       * adv_expanded,
                                 s_it_clipd * adv_expanded).mean()
            ratio_for_log = s_i_sg
            clip_eps_used = self.clip_eps_seq

        elif self.algorithm == 'grpo_plus':
            # GRPO+ : GSPO sequence-level importance ratio (low variance) +
            # hybrid sequence/token advantage. Designed for the trajectory-
            # reward setting (PDMS) where:
            #   - per-token importance ratio piles noise across T tokens
            #     (handled by GSPO sequence-level ratio)
            #   - sequence-only advantage flattens decision granularity — e.g.
            #     obstacle avoidance affects only a few waypoints, but pure
            #     GSPO gives every token in the rollout the same advantage
            #     (handled by per-token attention weight derived from
            #      group-divergence of predicted trajectories)
            #
            # Per-token weight w[i, t] = ||pos_xy[i,t] − mean(pos_xy[:,t])||
            #                            normalized so mean over t equals 1.
            # Tokens where rollout i diverged from the group mean are the
            # "differentiating decisions" in this scene; combined with the
            # signed sequence advantage A_seq[i] they form A_tok[i, t] which
            # is positive on good decisions and negative on bad ones.

            # ---- (a) Pure GSPO term (sequence ratio × sequence advantage) --
            log_seq_ratio   = log_ratio_token.mean(dim=-1)                   # [G]
            seq_ratio       = torch.exp(log_seq_ratio)
            seq_ratio_clipd = torch.clamp(seq_ratio,
                                          1.0 - self.clip_eps_seq,
                                          1.0 + self.clip_eps_seq)
            pg_seq = -torch.min(seq_ratio       * valid_advantages,
                                seq_ratio_clipd * valid_advantages).mean()

            # ---- (b) Per-token attention weight from rollout divergence ----
            # Stack the predicted trajectories of the valid rollouts.
            valid_trajs = torch.stack([
                rollouts[i].trajectory.to(self.device, dtype=torch.float32)
                for i in valid_idx
            ])                                                               # [G, T, 3]
            pos_xy = valid_trajs[..., :2]                                    # [G, T, 2]

            if pos_xy.shape[0] >= 2:
                mean_xy   = pos_xy.mean(dim=0, keepdim=True)                 # [1, T, 2]
                divergence = (pos_xy - mean_xy).norm(dim=-1)                 # [G, T]
                # Per-rollout normalization → mean weight over t equals 1.
                # If a rollout matches the group mean exactly (zero divergence
                # everywhere), fall back to a uniform weight of 1.
                norm = divergence.mean(dim=-1, keepdim=True)                 # [G, 1]
                uniform = torch.ones_like(divergence)
                weight  = torch.where(
                    norm > 1e-6,
                    divergence / norm.clamp(min=1e-6),
                    uniform,
                )
                # Cap extreme weights (defensive — shouldn't fire in practice).
                weight = weight.clamp(max=5.0)
            else:
                # Group has only one valid rollout: no divergence signal.
                weight = torch.ones_like(new_token_log_probs)                # [G, T]

            A_tok = valid_advantages.unsqueeze(1) * weight                   # [G, T]

            # ---- (c) Token-attention term (GSPO-token style routing) -------
            s_i_sg = seq_ratio.detach()                                      # [G]
            token_grad_factor = torch.exp(
                new_token_log_probs - new_token_log_probs.detach()
            )                                                                # [G, T]
            s_it       = s_i_sg.unsqueeze(1) * token_grad_factor             # [G, T]
            s_it_clipd = torch.clamp(s_i_sg, 1.0 - self.clip_eps_seq,
                                     1.0 + self.clip_eps_seq).unsqueeze(1) * token_grad_factor
            pg_tok = -torch.min(s_it       * A_tok,
                                s_it_clipd * A_tok).mean()

            # ---- (d) Convex combination ------------------------------------
            alpha   = self.token_attention_alpha
            pg_loss = (1.0 - alpha) * pg_seq + alpha * pg_tok

            ratio_for_log = seq_ratio
            clip_eps_used = self.clip_eps_seq

        elif self.algorithm == 'dr_grpo_plus':
            # Dr. GRPO+: NoRD recipe applied to GRPO+.
            #   • No /std in advantage  (same fix as Dr. GRPO)
            #   • Token-level PPO clip epsilon (0.2) for both sequence and
            #     token-attention terms. 4e-4 is far too small for T=8.
            #   • pg_tok uses SUM over T (no length normalisation) matching
            #     Dr. GRPO; pg_seq stays MEAN over G as it is sequence-level.
            raw_advantages = (rewards - mean_reward).clamp(-5.0, 5.0)        # [N], no /std
            valid_raw_adv  = raw_advantages[valid_idx]                       # [G]

            # ---- (a) Sequence-level term -----------------------------------
            log_seq_ratio   = log_ratio_token.mean(dim=-1)                   # [G]
            seq_ratio       = torch.exp(log_seq_ratio)
            seq_ratio_clipd = torch.clamp(seq_ratio,
                                          1.0 - self.clip_eps,
                                          1.0 + self.clip_eps)
            pg_seq = -torch.min(seq_ratio       * valid_raw_adv,
                                seq_ratio_clipd * valid_raw_adv).mean()

            # ---- (b) Per-token attention weight from rollout divergence ----
            valid_trajs = torch.stack([
                rollouts[i].trajectory.to(self.device, dtype=torch.float32)
                for i in valid_idx
            ])                                                               # [G, T, 3]
            pos_xy = valid_trajs[..., :2]                                    # [G, T, 2]

            if pos_xy.shape[0] >= 2:
                mean_xy   = pos_xy.mean(dim=0, keepdim=True)                 # [1, T, 2]
                divergence = (pos_xy - mean_xy).norm(dim=-1)                 # [G, T]
                norm = divergence.mean(dim=-1, keepdim=True)                 # [G, 1]
                uniform = torch.ones_like(divergence)
                weight  = torch.where(
                    norm > 1e-6,
                    divergence / norm.clamp(min=1e-6),
                    uniform,
                )
                weight = weight.clamp(max=5.0)
            else:
                weight = torch.ones_like(new_token_log_probs)                # [G, T]

            A_tok = valid_raw_adv.unsqueeze(1) * weight                      # [G, T]

            # ---- (c) Token-attention term ----------------------------------
            s_i_sg = seq_ratio.detach()                                      # [G]
            token_grad_factor = torch.exp(
                new_token_log_probs - new_token_log_probs.detach()
            )                                                                # [G, T]
            s_it       = s_i_sg.unsqueeze(1) * token_grad_factor             # [G, T]
            s_it_clipd = torch.clamp(s_i_sg, 1.0 - self.clip_eps,
                                     1.0 + self.clip_eps).unsqueeze(1) * token_grad_factor
            pg_tok = -torch.min(s_it       * A_tok,
                                s_it_clipd * A_tok).sum(dim=1).mean()        # SUM over T

            # ---- (d) Blend -------------------------------------------------
            alpha   = self.token_attention_alpha
            pg_loss = (1.0 - alpha) * pg_seq + alpha * pg_tok

            ratio_for_log = seq_ratio
            clip_eps_used = self.clip_eps

        # True categorical KL on the SAMPLING-temperature distributions — more
        # stable than sampled log-prob differences and aligned with the policy
        # we are actually optimising (π(·|T)). Same KL term across algorithms.
        new_probs_all = new_log_probs_all.exp()
        kl_loss = (new_probs_all * (new_log_probs_all - ref_log_probs_all)).sum(dim=-1).mean()

        total_loss = pg_loss + self.kl_coef * kl_loss

        # Fraction of sequences (or tokens for GRPO) that hit the clip. A non-
        # trivial fraction is expected and acceptable in GSPO per the paper.
        if self.algorithm in ('grpo', 'dr_grpo', 'dr_gspo', 'dr_grpo_plus'):
            clip_frac = ((torch.exp(log_ratio_token) - 1.0).abs() > self.clip_eps).float().mean()
        else:
            clip_frac = ((torch.exp(log_ratio_token.mean(dim=-1)) - 1.0).abs() > self.clip_eps_seq).float().mean()

        metrics = {
            f'{self.algorithm}_loss': total_loss.item(),
            'policy_loss':            pg_loss.item(),
            'kl_div':                 kl_loss.item(),
            'mean_reward':            mean_reward.item(),
            'std_reward':             std_reward.item(),
            'max_reward':             rewards.max().item(),
            'min_reward':             rewards.min().item(),
            'mean_ratio':             ratio_for_log.mean().item(),
            'std_ratio':              ratio_for_log.std(unbiased=False).item() if ratio_for_log.numel() > 1 else 0.0,
            'clip_frac':              clip_frac.item(),
            'clip_eps_used':          float(clip_eps_used),
            'mean_advantage':         valid_advantages.mean().item(),
        }

        # Dr. GRPO actually optimises the un-std-normalised advantage; expose
        # it explicitly so wandb shows the magnitude that drives the gradient.
        if self.algorithm == 'dr_grpo':
            metrics['mean_raw_advantage'] = valid_raw_adv.mean().item()
            metrics['std_raw_advantage']  = (
                valid_raw_adv.std(unbiased=False).item() if valid_raw_adv.numel() > 1 else 0.0
            )

        # GRPO+ specific diagnostics: how much per-token weighting actually
        # differentiates tokens. weight_dispersion = max/mean − 1 indicates
        # how concentrated the attention is (0 = uniform, larger = focused).
        if self.algorithm == 'grpo_plus':
            metrics['token_attention_alpha'] = self.token_attention_alpha
            try:
                w_max  = weight.max(dim=-1).values.mean().item()
                w_disp = max(w_max - 1.0, 0.0)
                metrics['token_weight_max_mean']    = w_max
                metrics['token_weight_dispersion']  = w_disp
                metrics['pg_seq_term']              = pg_seq.item()
                metrics['pg_tok_term']              = pg_tok.item()
            except Exception:
                pass

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
