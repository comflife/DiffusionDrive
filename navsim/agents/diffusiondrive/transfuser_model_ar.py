"""
Discrete Autoregressive Transfuser Model for DiffusionDrive.

Ego planning: Discrete token-based AR
Agent representation: Continuous (no tokenization)
"""

from typing import Dict, Optional
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
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
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
        self._d_model = d_model
        self._config = config
        
        # Discrete AR parameters for EGO only
        self.ego_fut_mode = 20  # Number of planning modes
        self.agent_topk = getattr(config, 'agent_topk', 8)
        self.score_thresh = 0.05
        self.num_layers = 2
        self.num_heads = 8
        self.dropout = 0.1
        
        # Ego vocabulary
        self.ego_vocab_size = getattr(config, 'ego_vocab_size', 512)
        self.ego_vocab_path = getattr(config, 'ego_vocab_path', None)
        
        # Load ego codebook only
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
        self.step_emb = nn.Embedding(num_poses, d_model)
        self.ego_mode_emb = nn.Embedding(self.ego_fut_mode, d_model)
        self.role_emb = nn.Embedding(2, d_model)  # 0=ego, 1=agent
        
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
        
        self._init_weights()
    
    def _init_codebook(self):
        """Initialize or load ego codebook only."""
        import os
        
        if self.ego_vocab_path and os.path.isfile(self.ego_vocab_path):
            import numpy as np
            arr = np.load(self.ego_vocab_path).astype(np.float32)
            # Codebook shape: [V, T, 3] or [V, T, 2] - extract displacement (dx, dy)
            if arr.ndim == 3:
                # [V, T, 3] -> [V, T, 2] (ignore heading for token prediction)
                ego_cb = torch.from_numpy(arr[:, :, :2])
            else:
                ego_cb = torch.from_numpy(arr)
            self.ego_vocab_size = ego_cb.shape[0]
            print(f"Loaded ego codebook: {ego_cb.shape} from {self.ego_vocab_path}")
        else:
            # Random initialization fallback
            print(f"Warning: Ego codebook not found at {self.ego_vocab_path}, using random init")
            ego_cb = torch.randn(self.ego_vocab_size, self._num_poses, 2) * 0.1
        
        self.register_buffer('ego_codebook', ego_cb, persistent=False)
    
    def _init_weights(self):
        nn.init.normal_(self.step_emb.weight, std=0.02)
        nn.init.normal_(self.ego_mode_emb.weight, std=0.02)
        nn.init.normal_(self.role_emb.weight, std=0.02)
        nn.init.normal_(self.bos_emb.weight, std=0.02)
        nn.init.normal_(self.ego_token_emb.weight, mean=0.0, std=0.02)
        
        nn.init.xavier_uniform_(self.ego_token_head.weight)
        nn.init.zeros_(self.ego_token_head.bias)
    
    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    
    def select_topk_agents(
        self,
        agent_states: torch.Tensor,
        agent_labels: torch.Tensor
    ) -> tuple:
        """Select top-K agents by distance and score."""
        B, N, _ = agent_states.shape
        K = min(self.agent_topk, N)
        
        scores = agent_labels.sigmoid()
        valid = scores > self.score_thresh
        
        # Positions are first 2 dims of agent_states
        positions = agent_states[..., :2]
        distances = torch.linalg.norm(positions, dim=-1)
        distances = distances.masked_fill(~valid, float('inf'))
        
        _, topk_idx = torch.topk(distances, k=K, dim=-1, largest=False)
        topk_valid = valid.gather(1, topk_idx)
        
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
        
        indices = torch.zeros(B, M, T, dtype=torch.long, device=traj_pos.device)
        accumulated = torch.zeros(B, M, 2, device=traj_pos.device, dtype=traj_pos.dtype)
        
        for t in range(T):
            # [B, M, V, 2]: position if we pick each codebook entry at step t
            candidates = accumulated.unsqueeze(-2) + codebook.view(1, 1, V, 2)  # [V, 2]
            # Distance to target position at step t
            dist = (candidates - traj_pos[..., t:t+1, :]).pow(2).sum(-1)  # [B, M, V]
            chosen = dist.argmin(-1)  # [B, M]
            indices[..., t] = chosen
            accumulated = accumulated + codebook[chosen]  # [B, M, 2]
        
        return indices
    
    def _attn_stack(
        self,
        ego_q: torch.Tensor,
        agent_kv: torch.Tensor,
        bev_feat: torch.Tensor,
        topk_valid: torch.Tensor
    ) -> torch.Tensor:
        """AR attention stack."""
        B, M, T, D = ego_q.shape
        K = agent_kv.shape[1]
        caus = self._causal_mask(T, ego_q.device)
        inv_ag = ~topk_valid
        
        inv_ag_bm = inv_ag.unsqueeze(1).expand(-1, M, -1).reshape(B * M, K)
        all_invalid = inv_ag_bm.all(dim=-1, keepdim=True)
        inv_ag_bm = inv_ag_bm & ~all_invalid
        
        P = bev_feat.shape[1]
        
        for i in range(self.num_layers):
            # Temporal self-attention
            eg = ego_q.reshape(B * M, T, D)
            eg2, _ = self.t_attn[i](eg, eg, eg, attn_mask=caus)
            ego_q = self.t_norm[i](eg + eg2).reshape(B, M, T, D)
            
            # Ego-agent cross-attention (continuous agent features per timestep)
            new_ego = []
            for t in range(T):
                q = ego_q[:, :, t, :].reshape(B * M, 1, D)
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
            ego_q = self.bev_norm[i](eg_MT + eg2).reshape(B, M, T, D)
            
            # FFN
            ego_q = self.ffn_norm[i](ego_q + self.ffn[i](ego_q))
        
        return ego_q
    
    def forward(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        agent_states: torch.Tensor,
        agent_labels: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            ego_query: [B, 1, D]
            agents_query: [B, N, D]
            bev_feature: [B, D, H, W]
            agent_states: [B, N, state_dim]
            agent_labels: [B, N]
            targets: Optional training targets with 'trajectory' [B, T, 3]
        """
        B = ego_query.shape[0]
        M = self.ego_fut_mode
        T = self._num_poses
        D = self._d_model
        device = ego_query.device
        
        # Process BEV features
        bev_flat = bev_feature.flatten(2).permute(0, 2, 1)  # [B, H*W, D]
        bev_feat = self.bev_proj(bev_flat)
        
        # Select top-K agents
        topk_idx, topk_valid = self.select_topk_agents(agent_states, agent_labels)
        K = topk_idx.shape[1]
        
        # Encode agent features (continuous, no tokenization)
        tidx = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        agent_ctx = agents_query.gather(1, tidx)  # [B, K, D]
        agent_encoded = self.agent_encoder(agent_ctx)  # [B, K, agent_ctx_dim]
        
        # Expand to time dimension with step embedding
        step_e = self.step_emb.weight.view(1, 1, T, D)
        role_a = self.role_emb.weight[1].view(1, 1, 1, D)
        agent_kv = agent_encoded.unsqueeze(2).expand(-1, -1, T, -1) + step_e + role_a
        
        # Ego context
        ego_ctx = ego_query[:, 0, :]  # [B, D]
        ego_base = self.ego_ctx_proj(ego_ctx)
        
        if self.training and targets is not None:
            return self._forward_train(
                ego_base, agent_kv, bev_feat, topk_valid, targets, B, M, T, D, device
            )
        else:
            return self._forward_test(
                ego_base, agent_kv, bev_feat, topk_valid, B, M, T, D, device
            )
    
    def _forward_train(
        self, ego_base, agent_kv, bev_feat, topk_valid, targets,
        B, M, T, D, device
    ):
        """Training with teacher forcing."""
        gt_traj = targets['trajectory']  # [B, T, 3]
        
        # Expand to multiple modes
        if gt_traj.dim() == 3:
            gt_traj = gt_traj.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 3]
        
        gt_pos = gt_traj[..., :2]  # [B, M, T, 2]
        
        # Match to codebook
        ego_gt_tokens = self.match_to_codebook(gt_pos, self.ego_codebook.to(device))
        
        # BOS-shifted teacher forcing
        mode_e = self.ego_mode_emb.weight.view(1, M, 1, D)
        step_e = self.step_emb.weight.view(1, 1, T, D)
        role_e = self.role_emb.weight[0].view(1, 1, 1, D)
        
        bos = (self.bos_emb(torch.zeros(1, dtype=torch.long, device=device))
               .view(1, 1, D) + ego_base.unsqueeze(1))
        bos = bos.expand(B, M, D)
        
        # Shift GT tokens
        tok_embs = self.ego_token_emb(ego_gt_tokens[:, :, :-1])  # [B, M, T-1, D]
        input_embs = torch.cat([bos.unsqueeze(2), tok_embs], dim=2)  # [B, M, T, D]
        
        ego_q = input_embs + step_e + role_e + mode_e
        ego_q = self._attn_stack(ego_q, agent_kv, bev_feat, topk_valid)
        logits = self.ego_token_head(ego_q)  # [B, M, T, V]
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.ego_vocab_size),
            ego_gt_tokens.reshape(-1),
            reduction='mean'
        )
        
        # Get predictions
        ego_tokens = logits.argmax(-1)
        ego_offsets = self.ego_codebook[ego_tokens]  # [B, M, T, 2]
        ego_pred = ego_offsets.cumsum(dim=2)
        
        return {
            'trajectory_loss': loss,
            'trajectory': ego_pred[:, 0],
            'ego_tokens': ego_tokens,
        }
    
    def _forward_test(
        self, ego_base, agent_kv, bev_feat, topk_valid,
        B, M, T, D, device
    ):
        """Inference with AR decoding."""
        mode_e = self.ego_mode_emb.weight.view(1, M, 1, D)
        step_e = self.step_emb.weight.view(1, 1, T, D)
        role_e = self.role_emb.weight[0].view(1, 1, 1, D)
        
        bos = (self.bos_emb(torch.zeros(1, dtype=torch.long, device=device))
               .view(1, 1, D) + ego_base.unsqueeze(1))
        bos = bos.expand(B, M, D)
        
        input_embs = torch.zeros(B, M, T, D, device=device)
        input_embs[:, :, 0, :] = bos
        
        predicted_tokens = []
        for t in range(T):
            ego_q = input_embs + step_e + role_e + mode_e
            ego_out = self._attn_stack(ego_q, agent_kv, bev_feat, topk_valid)
            logit_t = self.ego_token_head(ego_out[:, :, t, :])
            tok_t = logit_t.argmax(-1)
            predicted_tokens.append(tok_t)
            
            if t < T - 1:
                input_embs[:, :, t + 1, :] = self.ego_token_emb(tok_t)
        
        ego_tokens = torch.stack(predicted_tokens, dim=2)
        
        # Lookup codebook: [B, M, T] -> [B, M, T, 2]
        # ego_codebook shape: [V, 2]
        ego_offsets = self.ego_codebook[ego_tokens]  # [B, M, T, 2]
        
        ego_pred = ego_offsets.cumsum(dim=2)
        
        # Add heading (zeros for now)
        heading = torch.zeros(B, M, T, 1, device=device)
        ego_pred_full = torch.cat([ego_pred, heading], dim=-1)
        
        return {
            'trajectory': ego_pred_full[:, 0],
            'trajectory_modes': ego_pred_full,
            'ego_tokens': ego_tokens,
        }


class V2TransfuserModelAR(nn.Module):
    """
    Transfuser with Discrete Autoregressive Trajectory Head for Ego.
    Agents use continuous representation.
    """

    def __init__(self, config: TransfuserConfig):
        super().__init__()
        
        self._query_splits = [1, config.num_bounding_boxes]
        self._config = config
        self._backbone = TransfuserBackbone(config)
        
        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)
        
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
        
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
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
        
        # Replace diffusion head with AR head (ego only)
        self._trajectory_head = DiscreteARTrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            config=config,
        )
        
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1, 320),
        )
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        
        batch_size = status_feature.shape[0]
        
        # Backbone
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)
        
        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]
        
        concat_cross_bev = keyval[:, :-1].permute(0, 2, 1).contiguous().view(
            batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1]
        )
        concat_cross_bev = F.interpolate(
            concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False
        )
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)
        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(
            batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1]
        )
        
        # Transformer decoder
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)
        
        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)
        
        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        
        # Agent prediction
        agents = self._agent_head(agents_query)
        output.update(agents)
        
        # Trajectory prediction with AR head (ego only)
        trajectory = self._trajectory_head(
            trajectory_query,
            agents_query,
            cross_bev_feature,
            agents["agent_states"],
            agents["agent_labels"],
            targets=targets
        )
        output.update(trajectory)
        
        return output
