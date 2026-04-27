"""
Discrete Autoregressive Transfuser Model for DiffusionDrive.

Ego planning: Discrete token-based AR
Agent representation: Continuous (no tokenization)

v2 additions (for GRPO):
- DiscreteARTrajectoryHead._compute_token_log_probs_tf()
    Teacher-forced log prob computation using given token sequences.
    Required for correct GRPO policy gradient (AR inference mode
    would condition on the model's own—not the rollout—tokens).
- V2TransfuserModelAR._run_backbone()
    Shared backbone logic used by both forward() and compute_token_log_probs().
- V2TransfuserModelAR.compute_token_log_probs()
    Public API for GRPO loss computation.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln


class AgentHead(nn.Module):
    """Bounding box prediction head (unchanged from original)."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        super().__init__()
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT]   = agent_states[..., BoundingBox2DIndex.POINT].tanh()   * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class DiscreteARTrajectoryHead(nn.Module):
    """
    Discrete token-based autoregressive trajectory head for EGO only.
    Agents use continuous representation as context.
    """

    def __init__(
        self,
        num_poses: int,
        d_ffn: int,
        d_model: int,
        config: TransfuserConfig,
    ):
        super().__init__()
        self._num_poses = num_poses
        self._d_model   = d_model
        self._config    = config

        # Discrete AR parameters for EGO only
        self.ego_fut_mode  = getattr(config, 'ar_num_modes', 1)
        self.agent_topk    = getattr(config, 'agent_topk', 8)
        self.score_thresh  = 0.05
        self.num_layers    = 2
        self.num_heads     = 8
        self.dropout       = 0.1
        self.token_loss_weight = getattr(config, 'ar_token_loss_weight', 1.0)
        self.traj_loss_weight = getattr(config, 'ar_traj_loss_weight', 8.0)
        self.heading_loss_weight = getattr(config, 'ar_heading_loss_weight', 2.0)
        self.use_residual_delta = getattr(config, 'ar_use_residual_delta', True)
        self.use_heading_head = getattr(config, 'ar_use_heading_head', True)
        self.codebook_mode = getattr(config, 'ar_codebook_mode', 'step_delta')
        self.match_heading_weight = getattr(config, 'ar_match_heading_weight', 1.0)
        self.teacher_forcing = getattr(config, 'ar_teacher_forcing', True)

        # Ego vocabulary
        self.ego_vocab_size = getattr(config, 'ego_vocab_size', 512)
        self.ego_vocab_path = getattr(config, 'ego_vocab_path', None)

        self._init_codebook()

        # Ego context projector
        self.ego_ctx_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

        # BEV feature projector
        self.bev_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

        # Agent continuous feature encoder (no tokenization)
        agent_ctx_dim = getattr(config, 'agent_context_dim', 256)
        self.agent_encoder = nn.Sequential(
            nn.Linear(d_model, agent_ctx_dim),
            nn.LayerNorm(agent_ctx_dim),
            nn.ReLU(inplace=True),
        )

        # Ego token embeddings
        self.ego_token_emb = nn.Embedding(self.ego_vocab_size, d_model)

        # Positional embeddings
        self.step_emb     = nn.Embedding(num_poses, d_model)
        self.ego_mode_emb = nn.Embedding(self.ego_fut_mode, d_model)
        self.role_emb     = nn.Embedding(2, d_model)   # 0=ego, 1=agent

        # AR Attention Stack
        # 1. Temporal self-attention
        self.t_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, self.num_heads, dropout=self.dropout, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.t_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_layers)])

        # 2. Ego-agent cross-attention (continuous agent features)
        self.e2a_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, self.num_heads, dropout=self.dropout, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.e2a_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_layers)])

        # 3. BEV cross-attention
        self.bev_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, self.num_heads, dropout=self.dropout, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.bev_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_layers)])

        # FFN
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(d_ffn, d_model),
                nn.Dropout(self.dropout),
            )
            for _ in range(self.num_layers)
        ])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_layers)])

        # BOS embedding
        self.bos_emb = nn.Embedding(1, d_model)

        # Ego prediction head
        self.ego_token_head = nn.Linear(d_model, self.ego_vocab_size)
        self.ego_delta_head = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, 2),
        )
        self.ego_heading_head = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, 1),
        )

        self._init_weights()
        self._configure_optional_heads()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_codebook(self):
        """Initialize or load ego codebook only."""
        import os

        if self.ego_vocab_path and os.path.isfile(self.ego_vocab_path):
            arr = np.load(self.ego_vocab_path).astype(np.float32)
            # Supported formats:
            # - step_delta: [V, 2] step displacement tokens.
            # - step_corners: [V, 4, 2] single-step box-corner action tokens.
            # - trajectory_corners: [V, T, 4, 2] full box-corner trajectories.
            #   This is converted to [V, T, 3] = center x/y + heading.
            if arr.ndim == 4 and arr.shape[-2:] == (4, 2):
                if self.codebook_mode != 'trajectory_corners':
                    raise ValueError(
                        f"Loaded corner trajectory codebook {arr.shape}, but "
                        f"ar_codebook_mode={self.codebook_mode!r}. Use "
                        "agent.config.ar_codebook_mode=trajectory_corners."
                    )
                centers = arr.mean(axis=2)
                heading_vec = arr[:, :, 0, :] - arr[:, :, 3, :]
                headings = np.arctan2(heading_vec[..., 1], heading_vec[..., 0])[..., None]
                ego_cb = torch.from_numpy(np.concatenate([centers, headings], axis=-1).astype(np.float32))
            elif arr.ndim == 3 and arr.shape[-2:] == (4, 2):
                if self.codebook_mode != 'step_corners':
                    raise ValueError(
                        f"Loaded single-step corner codebook {arr.shape}, but "
                        f"ar_codebook_mode={self.codebook_mode!r}. Use "
                        "agent.config.ar_codebook_mode=step_corners."
                    )
                centers = arr.mean(axis=1)
                heading_vec = arr[:, 0, :] - arr[:, 3, :]
                headings = np.arctan2(heading_vec[:, 1], heading_vec[:, 0])[:, None]
                ego_cb = torch.from_numpy(np.concatenate([centers, headings], axis=-1).astype(np.float32))
            elif arr.ndim == 3:
                if self.codebook_mode == 'trajectory_corners':
                    if arr.shape[-1] < 3:
                        raise ValueError(f"Trajectory codebook must include heading, got {arr.shape}")
                    ego_cb = torch.from_numpy(arr[:, :, :3])
                else:
                    # [V, T, 3] → [V, T, 2] for legacy step-token path.
                    ego_cb = torch.from_numpy(arr[:, :, :2])
            else:
                ego_cb = torch.from_numpy(arr)
            self.ego_vocab_size = ego_cb.shape[0]
            print(f"Loaded ego codebook: {ego_cb.shape} from {self.ego_vocab_path}")
        else:
            print(f"Warning: Ego codebook not found at {self.ego_vocab_path}, using random init")
            if self.codebook_mode == 'trajectory_corners':
                ego_cb = torch.randn(self.ego_vocab_size, self._num_poses, 3) * 0.1
            elif self.codebook_mode == 'step_corners':
                ego_cb = torch.randn(self.ego_vocab_size, 3) * 0.1
            else:
                ego_cb = torch.randn(self.ego_vocab_size, 2) * 0.1

        self.register_buffer('ego_codebook', ego_cb, persistent=False)

    def _init_weights(self):
        nn.init.normal_(self.step_emb.weight,     std=0.02)
        nn.init.normal_(self.ego_mode_emb.weight, std=0.02)
        nn.init.normal_(self.role_emb.weight,     std=0.02)
        nn.init.normal_(self.bos_emb.weight,      std=0.02)
        nn.init.normal_(self.ego_token_emb.weight, mean=0.0, std=0.02)

        nn.init.xavier_uniform_(self.ego_token_head.weight)
        nn.init.zeros_(self.ego_token_head.bias)
        for head in [self.ego_delta_head, self.ego_heading_head]:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def _configure_optional_heads(self):
        if not self.use_residual_delta:
            for param in self.ego_delta_head.parameters():
                param.requires_grad = False
        if not self.use_heading_head:
            for param in self.ego_heading_head.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _build_trajectory(
        self,
        hidden: torch.Tensor,   # [B, M, T, D]
        tokens: torch.Tensor,   # [B, M, T]
    ) -> torch.Tensor:
        """Decode discrete tokens plus learned residuals into poses."""
        if self.codebook_mode == 'trajectory_corners':
            return self._decode_trajectory_tokens(tokens)
        if self.codebook_mode == 'step_corners':
            return self._decode_step_corner_tokens(hidden, tokens)

        token_deltas = self.ego_codebook[tokens]          # [B, M, T, 2]
        if self.use_residual_delta:
            residual_deltas = self.ego_delta_head(hidden)     # [B, M, T, 2]
        else:
            residual_deltas = torch.zeros_like(token_deltas)
        deltas_xy = token_deltas + residual_deltas
        pos_xy = deltas_xy.cumsum(dim=2)

        if self.use_heading_head:
            heading = self.ego_heading_head(hidden)
        else:
            heading = torch.atan2(deltas_xy[..., 1], deltas_xy[..., 0]).unsqueeze(-1)
        return torch.cat([pos_xy, heading], dim=-1)

    def _decode_step_corner_tokens(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compose local single-step [dx, dy, dtheta] tokens autoregressively."""
        token_actions = self.ego_codebook[tokens]          # [B, M, T, 3]
        local_deltas = token_actions[..., :2]

        if self.use_residual_delta:
            local_deltas = local_deltas + self.ego_delta_head(hidden)

        token_dtheta = token_actions[..., 2:3]
        if self.use_heading_head:
            step_dtheta = self.ego_heading_head(hidden)
        else:
            step_dtheta = token_dtheta

        positions = []
        headings = []
        pos = torch.zeros_like(local_deltas[..., 0, :])
        heading = torch.zeros_like(step_dtheta[..., 0, :])

        for t in range(local_deltas.shape[2]):
            cos_h = torch.cos(heading)
            sin_h = torch.sin(heading)
            dx = local_deltas[..., t, 0:1]
            dy = local_deltas[..., t, 1:2]
            global_delta = torch.cat(
                [dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h],
                dim=-1,
            )
            pos = pos + global_delta
            heading = self._wrap_angle(heading + step_dtheta[..., t, :])
            positions.append(pos)
            headings.append(heading)

        pos_xy = torch.stack(positions, dim=2)
        heading = torch.stack(headings, dim=2)
        return torch.cat([pos_xy, heading], dim=-1)

    def _decode_trajectory_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode whole-trajectory tokens to [B, M, T, 3] poses."""
        if tokens.dim() == 3:
            tokens = tokens[..., 0]
        traj = self.ego_codebook[tokens]  # [B, M, T, 3]
        if traj.shape[2] != self._num_poses:
            raise ValueError(
                f"Trajectory codebook length {traj.shape[2]} does not match "
                f"model num_poses {self._num_poses}."
            )
        return traj

    def select_topk_agents(
        self,
        agent_states: torch.Tensor,
        agent_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-K agents by distance and score."""
        B, N, _ = agent_states.shape
        K = min(self.agent_topk, N)

        scores    = agent_labels.sigmoid()
        valid     = scores > self.score_thresh
        positions = agent_states[..., :2]
        distances = torch.linalg.norm(positions, dim=-1)
        distances = distances.masked_fill(~valid, float('inf'))

        _, topk_idx  = torch.topk(distances, k=K, dim=-1, largest=False)
        topk_valid   = valid.gather(1, topk_idx)

        return topk_idx, topk_valid

    @torch.no_grad()
    def match_to_codebook(self, traj_pos: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Match trajectory positions to codebook tokens.

        Args:
            traj_pos: [B, M, T, 2] cumulative positions
            codebook: [V, 2] single-step displacements

        Returns: [B, M, T] token indices
        """
        B, M, T, _ = traj_pos.shape
        V = codebook.shape[0]

        indices     = torch.zeros(B, M, T, dtype=torch.long, device=traj_pos.device)
        accumulated = torch.zeros(B, M, 2, device=traj_pos.device, dtype=traj_pos.dtype)

        for t in range(T):
            candidates = accumulated.unsqueeze(-2) + codebook.view(1, 1, V, 2)
            dist       = (candidates - traj_pos[..., t:t+1, :]).pow(2).sum(-1)
            chosen     = dist.argmin(-1)
            indices[..., t] = chosen
            accumulated = accumulated + codebook[chosen]

        return indices

    @torch.no_grad()
    def match_to_step_corner_codebook(
        self,
        gt_traj: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        """Match GT trajectory to local autoregressive single-step action tokens."""
        B, M, T, _ = gt_traj.shape
        V = codebook.shape[0]

        indices = torch.zeros(B, M, T, dtype=torch.long, device=gt_traj.device)
        prev_pos = torch.zeros(B, M, 2, device=gt_traj.device, dtype=gt_traj.dtype)
        prev_heading = torch.zeros(B, M, 1, device=gt_traj.device, dtype=gt_traj.dtype)

        cb_delta = codebook[:, :2].view(1, 1, V, 2)
        cb_heading = codebook[:, 2].view(1, 1, V)

        for t in range(T):
            delta_global = gt_traj[..., t, :2] - prev_pos
            cos_h = torch.cos(prev_heading)
            sin_h = torch.sin(prev_heading)
            local_delta = torch.cat(
                [
                    delta_global[..., 0:1] * cos_h + delta_global[..., 1:2] * sin_h,
                    -delta_global[..., 0:1] * sin_h + delta_global[..., 1:2] * cos_h,
                ],
                dim=-1,
            )
            local_heading = self._wrap_angle(gt_traj[..., t, 2:3] - prev_heading)

            xy_dist = (cb_delta - local_delta.unsqueeze(2)).pow(2).sum(-1)
            heading_dist = self._wrap_angle(cb_heading - local_heading.squeeze(-1).unsqueeze(2)).pow(2)
            chosen = (xy_dist + self.match_heading_weight * heading_dist).argmin(-1)

            indices[..., t] = chosen
            prev_pos = gt_traj[..., t, :2]
            prev_heading = gt_traj[..., t, 2:3]

        return indices

    @torch.no_grad()
    def match_to_trajectory_codebook(
        self,
        gt_traj: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        """Match full [x, y, heading] trajectories to whole-trajectory tokens."""
        if gt_traj.dim() != 4:
            raise ValueError(f"Expected gt_traj [B, M, T, 3], got {gt_traj.shape}")
        if codebook.dim() != 3 or codebook.shape[-1] < 3:
            raise ValueError(f"Expected trajectory codebook [V, T, 3], got {codebook.shape}")

        gt_xy = gt_traj[..., :2].unsqueeze(2)       # [B, M, 1, T, 2]
        cb_xy = codebook[:, :, :2].view(1, 1, codebook.shape[0], codebook.shape[1], 2)
        xy_dist = (gt_xy - cb_xy).pow(2).sum(-1).mean(-1)

        gt_h = gt_traj[..., 2].unsqueeze(2)         # [B, M, 1, T]
        cb_h = codebook[:, :, 2].view(1, 1, codebook.shape[0], codebook.shape[1])
        heading_dist = self._wrap_angle(gt_h - cb_h).pow(2).mean(-1)

        dist = xy_dist + self.match_heading_weight * heading_dist
        return dist.argmin(dim=-1)                  # [B, M]

    def _attn_stack(
        self,
        ego_q:     torch.Tensor,   # [B, M, T, D]
        agent_kv:  torch.Tensor,   # [B, K, T, D]
        bev_feat:  torch.Tensor,   # [B, P, D]
        topk_valid: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        """AR attention stack."""
        B, M, T, D = ego_q.shape
        K   = agent_kv.shape[1]
        caus = self._causal_mask(T, ego_q.device)
        inv_ag = ~topk_valid

        inv_ag_bm  = inv_ag.unsqueeze(1).expand(-1, M, -1).reshape(B * M, K)
        all_invalid = inv_ag_bm.all(dim=-1, keepdim=True)
        inv_ag_bm   = inv_ag_bm & ~all_invalid

        P = bev_feat.shape[1]

        for i in range(self.num_layers):
            # Temporal self-attention
            eg  = ego_q.reshape(B * M, T, D)
            eg2, _ = self.t_attn[i](eg, eg, eg, attn_mask=caus)
            ego_q  = self.t_norm[i](eg + eg2).reshape(B, M, T, D)

            # Ego-agent cross-attention (continuous agent features per timestep)
            new_ego = []
            for t in range(T):
                q  = ego_q[:, :, t, :].reshape(B * M, 1, D)
                kv = (agent_kv[:, :, t, :]
                      .unsqueeze(1).expand(-1, M, -1, -1)
                      .reshape(B * M, K, D))
                out, _ = self.e2a_attn[i](q, kv, kv, key_padding_mask=inv_ag_bm)
                new_ego.append(self.e2a_norm[i](q + out).reshape(B, M, D))
            ego_q = torch.stack(new_ego, dim=2)

            # BEV cross-attention
            eg_MT = ego_q.reshape(B * M, T, D)
            bev_bm = bev_feat.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, P, D)
            eg2, _ = self.bev_attn[i](eg_MT, bev_bm, bev_bm)
            ego_q  = self.bev_norm[i](eg_MT + eg2).reshape(B, M, T, D)

            # FFN
            ego_q = self.ffn_norm[i](ego_q + self.ffn[i](ego_q))

        return ego_q

    # ------------------------------------------------------------------
    # Training / inference paths
    # ------------------------------------------------------------------

    def _forward_train(
        self, ego_base, agent_kv, bev_feat, topk_valid, targets,
        B, M, T, D, device,
    ):
        """Training with teacher forcing."""
        gt_traj = targets['trajectory']   # [B, T, 3]

        if gt_traj.dim() == 3:
            gt_traj = gt_traj.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 3]

        if self.codebook_mode == 'trajectory_corners':
            return self._forward_train_trajectory_token(
                ego_base, agent_kv, bev_feat, topk_valid, gt_traj,
                B, M, T, D, device,
            )

        if self.codebook_mode == 'step_corners':
            ego_gt_tokens = self.match_to_step_corner_codebook(gt_traj, self.ego_codebook.to(device))
        else:
            gt_pos = gt_traj[..., :2]  # [B, M, T, 2]
            ego_gt_tokens = self.match_to_codebook(gt_pos, self.ego_codebook.to(device))

        if not self.teacher_forcing:
            rollout = self._forward_test(
                ego_base, agent_kv, bev_feat, topk_valid,
                B, M, T, D, device, temperature=0.0,
            )
            logits = rollout['ego_logits'].unsqueeze(1)  # [B, 1, T, V]
            if M > 1:
                logits = logits.expand(-1, M, -1, -1)

            token_loss = F.cross_entropy(
                logits.reshape(-1, self.ego_vocab_size),
                ego_gt_tokens.reshape(-1),
                reduction='mean',
            )

            ego_pred = rollout['trajectory_modes']
            traj_loss = F.smooth_l1_loss(ego_pred[..., :2], gt_traj[..., :2], reduction='mean')
            heading_loss = F.smooth_l1_loss(ego_pred[..., 2:], gt_traj[..., 2:], reduction='mean')
            loss = (
                self.token_loss_weight * token_loss
                + self.traj_loss_weight * traj_loss
                + self.heading_loss_weight * heading_loss
            )

            return {
                'trajectory_loss': loss,
                'trajectory':      ego_pred[:, 0],
                'ego_tokens':      rollout['ego_tokens'],
                'ego_logits':      rollout['ego_logits'],
                'token_loss':      token_loss.detach(),
                'traj_loss':       traj_loss.detach(),
                'heading_loss':    heading_loss.detach(),
            }

        mode_e = self.ego_mode_emb.weight.view(1, M, 1, D)
        step_e = self.step_emb.weight.view(1, 1, T, D)
        role_e = self.role_emb.weight[0].view(1, 1, 1, D)

        bos = (self.bos_emb(torch.zeros(1, dtype=torch.long, device=device))
               .view(1, 1, D) + ego_base.unsqueeze(1))
        bos = bos.expand(B, M, D)

        tok_embs   = self.ego_token_emb(ego_gt_tokens[:, :, :-1])  # [B, M, T-1, D]
        input_embs = torch.cat([bos.unsqueeze(2), tok_embs], dim=2)  # [B, M, T, D]

        ego_q  = input_embs + step_e + role_e + mode_e
        ego_q  = self._attn_stack(ego_q, agent_kv, bev_feat, topk_valid)
        logits = self.ego_token_head(ego_q)  # [B, M, T, V]

        token_loss = F.cross_entropy(
            logits.reshape(-1, self.ego_vocab_size),
            ego_gt_tokens.reshape(-1),
            reduction='mean',
        )

        ego_tokens = logits.argmax(-1)
        ego_pred = self._build_trajectory(ego_q, ego_gt_tokens)

        traj_loss = F.smooth_l1_loss(ego_pred[..., :2], gt_traj[..., :2], reduction='mean')
        heading_loss = F.smooth_l1_loss(ego_pred[..., 2:], gt_traj[..., 2:], reduction='mean')
        loss = (
            self.token_loss_weight * token_loss
            + self.traj_loss_weight * traj_loss
            + self.heading_loss_weight * heading_loss
        )

        return {
            'trajectory_loss': loss,
            'trajectory':      ego_pred[:, 0],
            'ego_tokens':      ego_tokens,
            'ego_logits':      logits[:, 0],   # [B, T, V] – for GRPO
            'token_loss':      token_loss.detach(),
            'traj_loss':       traj_loss.detach(),
            'heading_loss':    heading_loss.detach(),
        }

    def _forward_train_trajectory_token(
        self, ego_base, agent_kv, bev_feat, topk_valid, gt_traj,
        B, M, T, D, device,
    ):
        """Train a single discrete token that represents the full trajectory."""
        ego_gt_tokens = self.match_to_trajectory_codebook(gt_traj, self.ego_codebook.to(device))

        mode_e = self.ego_mode_emb.weight.view(1, M, 1, D)
        step_e = self.step_emb.weight[0].view(1, 1, 1, D)
        role_e = self.role_emb.weight[0].view(1, 1, 1, D)

        bos = (self.bos_emb(torch.zeros(1, dtype=torch.long, device=device))
               .view(1, 1, D) + ego_base.unsqueeze(1))
        bos = bos.expand(B, M, D)

        ego_q = bos.unsqueeze(2) + step_e + role_e + mode_e
        ego_q = self._attn_stack(ego_q, agent_kv[:, :, :1, :], bev_feat, topk_valid)
        logits = self.ego_token_head(ego_q).squeeze(2)  # [B, M, V]

        token_loss = F.cross_entropy(
            logits.reshape(-1, self.ego_vocab_size),
            ego_gt_tokens.reshape(-1),
            reduction='mean',
        )

        ego_tokens = logits.argmax(-1)
        ego_pred = self._decode_trajectory_tokens(ego_tokens)

        traj_loss = F.smooth_l1_loss(ego_pred[..., :2], gt_traj[..., :2], reduction='mean')
        heading_loss = F.smooth_l1_loss(ego_pred[..., 2:], gt_traj[..., 2:], reduction='mean')
        loss = (
            self.token_loss_weight * token_loss
            + self.traj_loss_weight * traj_loss
            + self.heading_loss_weight * heading_loss
        )

        return {
            'trajectory_loss': loss,
            'trajectory':      ego_pred[:, 0],
            'ego_tokens':      ego_tokens.unsqueeze(-1),
            'ego_logits':      logits[:, 0].unsqueeze(1),
            'token_loss':      token_loss.detach(),
            'traj_loss':       traj_loss.detach(),
            'heading_loss':    heading_loss.detach(),
        }

    def _forward_test(
        self, ego_base, agent_kv, bev_feat, topk_valid,
        B, M, T, D, device, temperature: float = 0.0,
    ):
        """Inference with AR decoding."""
        if self.codebook_mode == 'trajectory_corners':
            return self._forward_test_trajectory_token(
                ego_base, agent_kv, bev_feat, topk_valid,
                B, M, T, D, device, temperature,
            )

        mode_e = self.ego_mode_emb.weight.view(1, M, 1, D)
        step_e = self.step_emb.weight.view(1, 1, T, D)
        role_e = self.role_emb.weight[0].view(1, 1, 1, D)

        bos = (self.bos_emb(torch.zeros(1, dtype=torch.long, device=device))
               .view(1, 1, D) + ego_base.unsqueeze(1))
        bos = bos.expand(B, M, D)

        input_embs = torch.zeros(B, M, T, D, device=device)
        input_embs[:, :, 0, :] = bos

        predicted_tokens: list = []
        all_logits:       list = []

        for t in range(T):
            ego_q   = input_embs + step_e + role_e + mode_e
            ego_out = self._attn_stack(ego_q, agent_kv, bev_feat, topk_valid)
            logit_t = self.ego_token_head(ego_out[:, :, t, :])   # [B, M, V]
            all_logits.append(logit_t)

            if temperature > 0:
                probs = F.softmax(logit_t / temperature, dim=-1)
                tok_t = torch.multinomial(probs.view(-1, self.ego_vocab_size), 1).view(B, M)
            else:
                tok_t = logit_t.argmax(-1)

            predicted_tokens.append(tok_t)

            if t < T - 1:
                input_embs[:, :, t + 1, :] = self.ego_token_emb(tok_t)

        ego_tokens = torch.stack(predicted_tokens, dim=2)    # [B, M, T]
        ego_logits = torch.stack(all_logits,        dim=2)   # [B, M, T, V]
        ego_hidden = self._attn_stack(input_embs + step_e + role_e + mode_e, agent_kv, bev_feat, topk_valid)
        ego_pred_full = self._build_trajectory(ego_hidden, ego_tokens)

        return {
            'trajectory':       ego_pred_full[:, 0],
            'trajectory_modes': ego_pred_full,
            'ego_tokens':       ego_tokens,
            'ego_logits':       ego_logits[:, 0],   # [B, T, V]
        }

    def _forward_test_trajectory_token(
        self, ego_base, agent_kv, bev_feat, topk_valid,
        B, M, T, D, device, temperature: float = 0.0,
    ):
        """Inference for whole-trajectory codebook mode."""
        mode_e = self.ego_mode_emb.weight.view(1, M, 1, D)
        step_e = self.step_emb.weight[0].view(1, 1, 1, D)
        role_e = self.role_emb.weight[0].view(1, 1, 1, D)

        bos = (self.bos_emb(torch.zeros(1, dtype=torch.long, device=device))
               .view(1, 1, D) + ego_base.unsqueeze(1))
        bos = bos.expand(B, M, D)

        ego_q = bos.unsqueeze(2) + step_e + role_e + mode_e
        ego_q = self._attn_stack(ego_q, agent_kv[:, :, :1, :], bev_feat, topk_valid)
        logits = self.ego_token_head(ego_q).squeeze(2)  # [B, M, V]

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(probs.view(-1, self.ego_vocab_size), 1).view(B, M)
        else:
            tokens = logits.argmax(-1)

        ego_pred_full = self._decode_trajectory_tokens(tokens)

        return {
            'trajectory':       ego_pred_full[:, 0],
            'trajectory_modes': ego_pred_full,
            'ego_tokens':       tokens.unsqueeze(-1),
            'ego_logits':       logits[:, 0].unsqueeze(1),
        }

    # ------------------------------------------------------------------
    # NEW: Teacher-forced log prob computation (for GRPO)
    # ------------------------------------------------------------------

    def _compute_token_log_probs_tf(
        self,
        ego_base:    torch.Tensor,   # [B, D]
        agent_kv:    torch.Tensor,   # [B, K, T, D]
        bev_feat:    torch.Tensor,   # [B, P, D]
        topk_valid:  torch.Tensor,   # [B, K]
        given_tokens: torch.Tensor,  # [B, T]  token indices to evaluate
        B: int, T: int, D: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forced log probability computation for GRPO.

        Unlike _forward_test (AR mode), this conditions each step t
        on the GIVEN rollout tokens a_0,...,a_{t-1} via BOS-shifted
        teacher forcing, giving the correct:

            log π_θ(given_token_t | s, given_tokens_{<t})

        Uses mode-0 only for efficiency.

        Returns
        -------
        token_log_probs : [B, T]   log probs for each given token
        logits          : [B, T, V] raw logits
        """
        # Mode 0 embeddings only (single-mode computation)
        mode_e = self.ego_mode_emb.weight[0].view(1, 1, 1, D)   # mode 0
        step_e = self.step_emb.weight.view(1, 1, T, D)
        role_e = self.role_emb.weight[0].view(1, 1, 1, D)

        # BOS  [B, 1, D]
        bos = (self.bos_emb(torch.zeros(1, dtype=torch.long, device=device))
               .view(1, 1, D) + ego_base.unsqueeze(1))   # [B, 1, D]

        # Teacher forcing: input = BOS + token_emb(given_tokens[:-1])
        tok_embs   = self.ego_token_emb(given_tokens[:, :-1])   # [B, T-1, D]
        tok_embs   = tok_embs.unsqueeze(1)                       # [B, 1, T-1, D]
        input_embs = torch.cat([bos.unsqueeze(2), tok_embs], dim=2)  # [B, 1, T, D]

        ego_q  = input_embs + step_e + role_e + mode_e           # [B, 1, T, D]
        ego_q  = self._attn_stack(ego_q, agent_kv, bev_feat, topk_valid)
        logits = self.ego_token_head(ego_q[:, 0])                # [B, T, V]

        # Gather log probs for the given tokens
        log_probs_all = F.log_softmax(logits, dim=-1)            # [B, T, V]
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        token_log_probs = log_probs_all[b_idx, t_idx, given_tokens]  # [B, T]

        return token_log_probs, logits

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        ego_query:    torch.Tensor,   # [B, 1, D]
        agents_query: torch.Tensor,   # [B, N, D]
        bev_feature:  torch.Tensor,   # [B, D, H, W]
        agent_states: torch.Tensor,   # [B, N, state_dim]
        agent_labels: torch.Tensor,   # [B, N]
        targets: Optional[Dict[str, torch.Tensor]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        B  = ego_query.shape[0]
        M  = self.ego_fut_mode
        T  = self._num_poses
        D  = self._d_model
        device = ego_query.device

        bev_flat = bev_feature.flatten(2).permute(0, 2, 1)   # [B, H*W, D]
        bev_feat = self.bev_proj(bev_flat)

        topk_idx, topk_valid = self.select_topk_agents(agent_states, agent_labels)
        K = topk_idx.shape[1]

        tidx       = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        agent_ctx  = agents_query.gather(1, tidx)                     # [B, K, D]
        agent_encoded = self.agent_encoder(agent_ctx)                  # [B, K, D]

        step_e  = self.step_emb.weight.view(1, 1, T, D)
        role_a  = self.role_emb.weight[1].view(1, 1, 1, D)
        agent_kv = agent_encoded.unsqueeze(2).expand(-1, -1, T, -1) + step_e + role_a

        ego_ctx  = ego_query[:, 0, :]
        ego_base = self.ego_ctx_proj(ego_ctx)

        if self.training and targets is not None:
            return self._forward_train(
                ego_base, agent_kv, bev_feat, topk_valid, targets,
                B, M, T, D, device,
            )
        else:
            return self._forward_test(
                ego_base, agent_kv, bev_feat, topk_valid,
                B, M, T, D, device, temperature,
            )


# ======================================================================
# Top-level model
# ======================================================================

class V2TransfuserModelAR(nn.Module):
    """
    Transfuser with Discrete Autoregressive Trajectory Head for Ego.
    Agents use continuous representation.
    """

    def __init__(self, config: TransfuserConfig):
        super().__init__()

        self._query_splits = [1, config.num_bounding_boxes]
        self._config       = config
        self._backbone     = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8 ** 2 + 1, config.tf_d_model)
        self._query_embedding  = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        self._bev_downscale     = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding   = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels, config.bev_features_channels,
                kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels, config.num_bev_classes,
                kernel_size=(1, 1), stride=1, padding=0, bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode='bilinear', align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = DiscreteARTrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            config=config,
        )

        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1, 320),
        )

    # ------------------------------------------------------------------
    # Shared backbone helper (used by forward + compute_token_log_probs)
    # ------------------------------------------------------------------

    def _run_backbone(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict:
        """
        Run the shared backbone (image/lidar encoder + transformer decoder).

        Returns a dict with keys:
          bev_semantic_map, trajectory_query, agents_query,
          cross_bev_feature, batch_size
        """
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature:  torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        bev_spatial_shape        = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape   = bev_feature.shape[2:]

        bev_feature    = self._bev_downscale(bev_feature).flatten(-2, -1).permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval  = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:, :-1].permute(0, 2, 1).contiguous().view(
            batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1]
        )
        concat_cross_bev = F.interpolate(
            concat_cross_bev, size=bev_spatial_shape,
            mode='bilinear', align_corners=False,
        )
        cross_bev_feature = torch.cat([concat_cross_bev, bev_feature_upscale], dim=1)
        cross_bev_feature = self.bev_proj(
            cross_bev_feature.flatten(-2, -1).permute(0, 2, 1)
        )
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(
            batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1]
        )

        query     = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map              = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        return {
            'bev_semantic_map':  bev_semantic_map,
            'trajectory_query':  trajectory_query,
            'agents_query':      agents_query,
            'cross_bev_feature': cross_bev_feature,
            'batch_size':        batch_size,
        }

    # ------------------------------------------------------------------
    # Standard forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets:  Optional[Dict[str, torch.Tensor]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        bb = self._run_backbone(features)

        output: Dict[str, torch.Tensor] = {
            'bev_semantic_map': bb['bev_semantic_map'],
        }

        agents = self._agent_head(bb['agents_query'])
        output.update(agents)

        trajectory = self._trajectory_head(
            bb['trajectory_query'],
            bb['agents_query'],
            bb['cross_bev_feature'],
            agents['agent_states'],
            agents['agent_labels'],
            targets=targets,
            temperature=temperature,
        )
        output.update(trajectory)

        return output

    # ------------------------------------------------------------------
    # NEW: Teacher-forced log prob (for GRPO loss computation)
    # ------------------------------------------------------------------

    def compute_token_log_probs(
        self,
        features:     Dict[str, torch.Tensor],
        given_tokens: torch.Tensor,   # [B, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute teacher-forced log probabilities for given token sequences.

        Used in GRPOTrainer.compute_grpo_loss() to obtain
            log π_θ(a_t | s, a_{<t})
        under the CURRENT policy weights, conditioned correctly on the
        rollout tokens (not on the model's own AR predictions).

        Args
        ----
        features      : input feature dict (batch of G identical scenes)
        given_tokens  : [B, T]  token indices from rollout

        Returns
        -------
        token_log_probs : [B, T]
        logits          : [B, T, V]
        """
        bb = self._run_backbone(features)

        B      = bb['batch_size']
        T      = given_tokens.shape[1]
        D      = self._config.tf_d_model
        device = given_tokens.device

        agents = self._agent_head(bb['agents_query'])

        # Build agent_kv (same as in DiscreteARTrajectoryHead.forward)
        topk_idx, topk_valid = self._trajectory_head.select_topk_agents(
            agents['agent_states'], agents['agent_labels']
        )
        tidx       = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        agent_ctx  = bb['agents_query'].gather(1, tidx)                  # [B, K, D]
        agent_enc  = self._trajectory_head.agent_encoder(agent_ctx)      # [B, K, D]

        step_e   = self._trajectory_head.step_emb.weight.view(1, 1, T, D)
        role_a   = self._trajectory_head.role_emb.weight[1].view(1, 1, 1, D)
        agent_kv = agent_enc.unsqueeze(2).expand(-1, -1, T, -1) + step_e + role_a

        # BEV features
        bev_flat = bb['cross_bev_feature'].flatten(2).permute(0, 2, 1)  # [B, P, D]
        bev_feat = self._trajectory_head.bev_proj(bev_flat)

        # Ego context
        ego_ctx  = bb['trajectory_query'][:, 0, :]
        ego_base = self._trajectory_head.ego_ctx_proj(ego_ctx)

        return self._trajectory_head._compute_token_log_probs_tf(
            ego_base, agent_kv, bev_feat, topk_valid,
            given_tokens, B, T, D, device,
        )
