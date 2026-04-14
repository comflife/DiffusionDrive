"""
Enhanced discrete autoregressive Transfuser model for DiffusionDrive.

This variant keeps the backbone and discrete-token AR planning setup, but
strengthens the decoder-side conditioning with:
  - richer BOS seed from ego query + BEV summary + agent summary + status
  - agent-state-aware context embeddings
  - step-aware agent keys/values
  - ego-conditioned gating before agent cross-attention
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_model_ar import (
    AgentHead,
    DiscreteARTrajectoryHead,
)
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln


class RichDiscreteARTrajectoryHead(DiscreteARTrajectoryHead):
    """Discrete AR head with stronger BOS and agent conditioning."""

    def __init__(
        self,
        num_poses: int,
        d_ffn: int,
        d_model: int,
        config: TransfuserConfig,
    ):
        super().__init__(num_poses=num_poses, d_ffn=d_ffn, d_model=d_model, config=config)

        # AgentHead predicts [x, y, heading, length, width] and we add score.
        self.agent_state_encoder = nn.Sequential(
            nn.Linear(6, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        self.agent_ctx_fuser = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        self.bos_fuser = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.step_agent_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        self.agent_gate_q = nn.Linear(d_model, d_model)
        self.agent_gate_k = nn.Linear(d_model, d_model)
        self.agent_gate_out = nn.Linear(d_model, 1)
        self.bos_scale = nn.Parameter(torch.tensor(1.0))

        self._init_rich_weights()

    def _init_rich_weights(self):
        for module in [
            self.agent_state_encoder,
            self.agent_ctx_fuser,
            self.bos_fuser,
            self.step_agent_proj,
        ]:
            for submodule in module:
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    nn.init.zeros_(submodule.bias)
        for module in [self.agent_gate_q, self.agent_gate_k, self.agent_gate_out]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _build_agent_kv(
        self,
        agents_query: torch.Tensor,   # [B, N, D]
        agent_states: torch.Tensor,   # [B, N, 5]
        agent_labels: torch.Tensor,   # [B, N]
        T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build richer step-aware agent memory."""
        B, _, D = agents_query.shape
        topk_idx, topk_valid = self.select_topk_agents(agent_states, agent_labels)

        tidx = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        topk_agent_q = agents_query.gather(1, tidx)  # [B, K, D]

        state_dim = agent_states.shape[-1]
        sidx = topk_idx.unsqueeze(-1).expand(-1, -1, state_dim)
        topk_agent_states = agent_states.gather(1, sidx)  # [B, K, 5]
        topk_scores = agent_labels.sigmoid().gather(1, topk_idx).unsqueeze(-1)  # [B, K, 1]
        agent_state_input = torch.cat([topk_agent_states, topk_scores], dim=-1)
        agent_state_emb = self.agent_state_encoder(agent_state_input)  # [B, K, D]

        agent_base = self.agent_ctx_fuser(torch.cat([self.agent_encoder(topk_agent_q), agent_state_emb], dim=-1))
        step_e = self.step_emb.weight.view(1, 1, T, D).expand(B, agent_base.shape[1], -1, -1)
        agent_base_t = agent_base.unsqueeze(2).expand(-1, -1, T, -1)
        agent_kv = self.step_agent_proj(torch.cat([agent_base_t, step_e], dim=-1))
        role_a = self.role_emb.weight[1].view(1, 1, 1, D)
        agent_kv = agent_kv + role_a

        return agent_kv, topk_valid, agent_base

    def _build_bos_seed(
        self,
        ego_query: torch.Tensor,      # [B, D]
        bev_feat: torch.Tensor,       # [B, P, D]
        agent_base: torch.Tensor,     # [B, K, D]
        topk_valid: torch.Tensor,     # [B, K]
        status_encoding: torch.Tensor,  # [B, D]
    ) -> torch.Tensor:
        """Build a richer BOS seed from ego, map, agent, and status summaries."""
        bev_summary = bev_feat.mean(dim=1)

        valid_mask = topk_valid.float().unsqueeze(-1)
        agent_denom = valid_mask.sum(dim=1).clamp(min=1.0)
        agent_summary = (agent_base * valid_mask).sum(dim=1) / agent_denom

        bos_input = torch.cat([ego_query, bev_summary, agent_summary, status_encoding], dim=-1)
        return self.bos_fuser(bos_input)

    def _attn_stack(
        self,
        ego_q: torch.Tensor,
        agent_kv: torch.Tensor,
        bev_feat: torch.Tensor,
        topk_valid: torch.Tensor,
    ) -> torch.Tensor:
        """AR attention stack with ego-conditioned agent gating."""
        B, M, T, D = ego_q.shape
        K = agent_kv.shape[1]
        caus = self._causal_mask(T, ego_q.device)
        inv_ag = ~topk_valid

        inv_ag_bm = inv_ag.unsqueeze(1).expand(-1, M, -1).reshape(B * M, K)
        all_invalid = inv_ag_bm.all(dim=-1, keepdim=True)
        inv_ag_bm = inv_ag_bm & ~all_invalid

        P = bev_feat.shape[1]
        for i in range(self.num_layers):
            eg = ego_q.reshape(B * M, T, D)
            eg2, _ = self.t_attn[i](eg, eg, eg, attn_mask=caus)
            ego_q = self.t_norm[i](eg + eg2).reshape(B, M, T, D)

            new_ego = []
            for t in range(T):
                q = ego_q[:, :, t, :].reshape(B * M, 1, D)
                kv = (
                    agent_kv[:, :, t, :]
                    .unsqueeze(1)
                    .expand(-1, M, -1, -1)
                    .reshape(B * M, K, D)
                )
                gate_q = self.agent_gate_q(q.expand(-1, K, -1))
                gate_k = self.agent_gate_k(kv)
                gate = torch.sigmoid(self.agent_gate_out(torch.tanh(gate_q + gate_k)))
                kv = kv * gate
                out, _ = self.e2a_attn[i](q, kv, kv, key_padding_mask=inv_ag_bm)
                new_ego.append(self.e2a_norm[i](q + out).reshape(B, M, D))
            ego_q = torch.stack(new_ego, dim=2)

            eg_mt = ego_q.reshape(B * M, T, D)
            bev_bm = bev_feat.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, P, D)
            eg2, _ = self.bev_attn[i](eg_mt, bev_bm, bev_bm)
            ego_q = self.bev_norm[i](eg_mt + eg2).reshape(B, M, T, D)
            ego_q = self.ffn_norm[i](ego_q + self.ffn[i](ego_q))

        return ego_q

    def forward(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        agent_states: torch.Tensor,
        agent_labels: torch.Tensor,
        status_encoding: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        B = ego_query.shape[0]
        M = self.ego_fut_mode
        T = self._num_poses
        D = self._d_model
        device = ego_query.device

        bev_flat = bev_feature.flatten(2).permute(0, 2, 1)
        bev_feat = self.bev_proj(bev_flat)

        agent_kv, topk_valid, agent_base = self._build_agent_kv(
            agents_query, agent_states, agent_labels, T
        )

        ego_ctx = ego_query[:, 0, :]
        ego_ctx = self.ego_ctx_proj(ego_ctx)
        ego_base = self._build_bos_seed(
            ego_ctx, bev_feat, agent_base, topk_valid, status_encoding
        ) * self.bos_scale

        if self.training and targets is not None:
            return self._forward_train(
                ego_base, agent_kv, bev_feat, topk_valid, targets,
                B, M, T, D, device,
            )
        return self._forward_test(
            ego_base, agent_kv, bev_feat, topk_valid,
            B, M, T, D, device, temperature,
        )


class V2TransfuserModelARPlus(nn.Module):
    """Enhanced AR Transfuser model with richer decoder conditioning."""

    def __init__(self, config: TransfuserConfig):
        super().__init__()

        self._query_splits = [1, config.num_bounding_boxes]
        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8 ** 2 + 1, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

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

        self._trajectory_head = RichDiscreteARTrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            config=config,
        )

        self.bev_proj = nn.Sequential(*linear_relu_ln(256, 1, 1, 320))

    def _run_backbone(self, features: Dict[str, torch.Tensor]) -> Dict:
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1).permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
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

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        return {
            "bev_semantic_map": bev_semantic_map,
            "trajectory_query": trajectory_query,
            "agents_query": agents_query,
            "cross_bev_feature": cross_bev_feature,
            "status_encoding": status_encoding,
            "batch_size": batch_size,
        }

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        bb = self._run_backbone(features)
        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bb["bev_semantic_map"]}

        agents = self._agent_head(bb["agents_query"])
        output.update(agents)

        trajectory = self._trajectory_head(
            bb["trajectory_query"],
            bb["agents_query"],
            bb["cross_bev_feature"],
            agents["agent_states"],
            agents["agent_labels"],
            bb["status_encoding"],
            targets=targets,
            temperature=temperature,
        )
        output.update(trajectory)
        return output

    def compute_token_log_probs(
        self,
        features: Dict[str, torch.Tensor],
        given_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bb = self._run_backbone(features)

        B = bb["batch_size"]
        T = given_tokens.shape[1]
        D = self._config.tf_d_model
        device = given_tokens.device

        agents = self._agent_head(bb["agents_query"])
        agent_kv, topk_valid, agent_base = self._trajectory_head._build_agent_kv(
            bb["agents_query"], agents["agent_states"], agents["agent_labels"], T
        )
        bev_flat = bb["cross_bev_feature"].flatten(2).permute(0, 2, 1)
        bev_feat = self._trajectory_head.bev_proj(bev_flat)
        ego_ctx = self._trajectory_head.ego_ctx_proj(bb["trajectory_query"][:, 0, :])
        ego_base = self._trajectory_head._build_bos_seed(
            ego_ctx, bev_feat, agent_base, topk_valid, bb["status_encoding"]
        ) * self._trajectory_head.bos_scale

        return self._trajectory_head._compute_token_log_probs_tf(
            ego_base, agent_kv, bev_feat, topk_valid, given_tokens, B, T, D, device
        )
