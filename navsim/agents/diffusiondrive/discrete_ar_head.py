"""
Discrete-token autoregressive trajectory planner for DiffusionDrive.

Based on VAD's DiscreteARPlanningHead, adapted for DiffusionDrive's Transfuser architecture.

Architecture:
  - Agent trajectories are quantized to codebook tokens (fixed context)
  - Ego planning is decoded autoregressively over future steps
  - At each future step t:
      1. Temporal self-attention (causal over ego decoded steps 0..t)
      2. Ego-agent cross-attention (ego at step t attends to agent tokens at step t)
      3. BEV cross-attention (ego attends to BEV features)
      4. FFN
  
  Only ego token logits are predicted and supervised.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def _load_npy_codebook(path: str, vocab_size: int) -> torch.Tensor:
    """Load .npy codebook → float32 tensor [V, token_dim]."""
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"Codebook file not found: '{path}'")
    arr = np.load(path).astype(np.float32)
    # Reshape to [V, D] where D is the flattened token dimension
    return torch.from_numpy(arr.reshape(arr.shape[0], -1))


def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """[T, T] bool, True → position is masked (cannot attend)."""
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class DiscreteARTrajectoryHead(nn.Module):
    """
    Discrete token-based autoregressive trajectory planner for DiffusionDrive.
    
    Replaces the diffusion-based TrajectoryHead with AR token prediction.
    
    Parameters
    ----------
    embed_dims : int
        Feature dimensionality (default: 256 for DiffusionDrive)
    num_poses : int
        Number of future poses to predict (trajectory length)
    ego_fut_mode : int
        Number of ego planning modes (e.g., 20)
    agent_topk : int
        Maximum number of agents to use as context
    score_thresh : float
        Minimum detection score for agent selection
    num_layers : int
        AR decoder depth
    num_heads : int
        Attention heads
    dropout : float
    bev_h : int
        BEV feature height
    bev_w : int
        BEV feature width
    vehicle_vocab_path : str
    vehicle_vocab_size : int
    cyclist_vocab_path : str
    cyclist_vocab_size : int
    pedestrian_vocab_path : str
    pedestrian_vocab_size : int
    ego_vocab_path : str
    ego_vocab_size : int
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_poses: int = 8,
        ego_fut_mode: int = 20,
        agent_topk: int = 8,
        score_thresh: float = 0.05,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        bev_h: int = 64,
        bev_w: int = 64,
        vehicle_vocab_path: Optional[str] = None,
        vehicle_vocab_size: int = 128,
        cyclist_vocab_path: Optional[str] = None,
        cyclist_vocab_size: int = 128,
        pedestrian_vocab_path: Optional[str] = None,
        pedestrian_vocab_size: int = 128,
        ego_vocab_path: Optional[str] = None,
        ego_vocab_size: int = 256,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_poses = num_poses
        self.ego_fut_mode = ego_fut_mode
        self.agent_topk = agent_topk
        self.score_thresh = score_thresh
        self.num_layers = num_layers
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        # Codebooks (fixed buffers, not trained)
        # Vehicle/cyclist/pedestrian codebooks are optional (not used for continuous agent features)
        # All codebooks are [V, 2] shape for single-step (dx, dy) displacements
        if vehicle_vocab_path:
            veh_cb = _load_npy_codebook(vehicle_vocab_path, vehicle_vocab_size)
        else:
            veh_cb = torch.randn(vehicle_vocab_size, 2) * 0.1
        if cyclist_vocab_path:
            cyc_cb = _load_npy_codebook(cyclist_vocab_path, cyclist_vocab_size)
        else:
            cyc_cb = torch.randn(cyclist_vocab_size, 2) * 0.1
        if pedestrian_vocab_path:
            ped_cb = _load_npy_codebook(pedestrian_vocab_path, pedestrian_vocab_size)
        else:
            ped_cb = torch.randn(pedestrian_vocab_size, 2) * 0.1
        
        # Ego codebook is required
        ego_cb = _load_npy_codebook(ego_vocab_path, ego_vocab_size)
        
        self.register_buffer('vehicle_codebook', veh_cb, persistent=False)
        self.register_buffer('cyclist_codebook', cyc_cb, persistent=False)
        self.register_buffer('pedestrian_codebook', ped_cb, persistent=False)
        self.register_buffer('ego_codebook', ego_cb, persistent=False)
        
        self.vehicle_vocab_size = vehicle_vocab_size
        self.cyclist_vocab_size = cyclist_vocab_size
        self.pedestrian_vocab_size = pedestrian_vocab_size
        self.ego_vocab_size = ego_vocab_size
        
        # Ego context projector (from query embedding)
        self.ego_ctx_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )
        
        # BEV feature projector
        self.bev_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )
        
        # Token embedding tables
        self.vehicle_token_emb = nn.Embedding(vehicle_vocab_size, embed_dims)
        self.cyclist_token_emb = nn.Embedding(cyclist_vocab_size, embed_dims)
        self.pedestrian_token_emb = nn.Embedding(pedestrian_vocab_size, embed_dims)
        self.ego_token_emb = nn.Embedding(ego_vocab_size, embed_dims)
        
        # Positional embeddings
        self.step_emb = nn.Embedding(num_poses, embed_dims)
        self.ego_mode_emb = nn.Embedding(ego_fut_mode, embed_dims)
        self.role_emb = nn.Embedding(2, embed_dims)  # 0=ego, 1=agent
        
        # AR Attention Stack
        # 1. Temporal self-attention (causal)
        self.t_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.t_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_layers)])
        
        # 2. Ego-agent cross-attention
        self.e2a_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.e2a_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_layers)])
        
        # 3. BEV cross-attention (replaces map cross-attention in VAD)
        self.bev_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.bev_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_layers)])
        
        # FFN
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dims * 4, embed_dims),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_layers)])
        
        # BOS embedding
        self.bos_emb = nn.Embedding(1, embed_dims)
        
        # Ego prediction head
        self.ego_token_head = nn.Linear(embed_dims, ego_vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize trainable modules."""
        nn.init.normal_(self.step_emb.weight, std=0.02)
        nn.init.normal_(self.ego_mode_emb.weight, std=0.02)
        nn.init.normal_(self.role_emb.weight, std=0.02)
        nn.init.normal_(self.bos_emb.weight, std=0.02)
        
        for emb in [self.vehicle_token_emb, self.cyclist_token_emb, 
                    self.pedestrian_token_emb, self.ego_token_emb]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        
        for module in [self.ego_ctx_proj, self.bev_proj]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        nn.init.xavier_uniform_(self.ego_token_head.weight)
        nn.init.zeros_(self.ego_token_head.bias)
    
    def select_topk_agents(
        self, 
        agent_states: torch.Tensor, 
        agent_labels: torch.Tensor,
        agent_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-K nearest valid agents.
        
        Args:
            agent_states: [B, N, state_dim] - predicted agent states
            agent_labels: [B, N] - agent validity scores (before sigmoid)
            agent_positions: [B, N, 2] - agent positions in ego frame
            
        Returns:
            topk_idx: [B, K]
            topk_valid: [B, K]
        """
        B, N, _ = agent_positions.shape
        K = min(self.agent_topk, N)
        
        # Get agent scores
        scores = agent_labels.sigmoid()  # [B, N]
        valid = scores > self.score_thresh  # [B, N]
        
        # Distance to ego
        distances = torch.linalg.norm(agent_positions, dim=-1)  # [B, N]
        distances = distances.masked_fill(~valid, float('inf'))
        
        # Select top-K by distance
        _, topk_idx = torch.topk(distances, k=K, dim=-1, largest=False)
        topk_valid = valid.gather(1, topk_idx)
        
        return topk_idx, topk_valid
    
    @torch.no_grad()
    def match_to_codebook(
        self, 
        traj_pos: torch.Tensor, 
        codebook: torch.Tensor
    ) -> torch.Tensor:
        """
        Greedy nearest-neighbour match in position space.
        
        Args:
            traj_pos: [B, K, T, 2] cumulative positions
            codebook: [V, 2] codebook entries (displacements)
            
        Returns: [B, K, T] token indices
        """
        B, K, T, _ = traj_pos.shape
        V = codebook.shape[0]
        
        indices = torch.zeros(B, K, T, dtype=torch.long, device=traj_pos.device)
        accumulated = torch.zeros(B, K, 2, device=traj_pos.device, dtype=traj_pos.dtype)
        
        for t in range(T):
            # [B, K, V, 2]: position if we pick each codebook entry
            candidates = accumulated.unsqueeze(-2) + codebook.view(1, 1, V, 2)
            # Distance to target position
            dist = (candidates - traj_pos[..., t:t+1, :]).pow(2).sum(-1)
            chosen = dist.argmin(-1)  # [B, K]
            indices[..., t] = chosen
            accumulated = accumulated + codebook[chosen]
        
        return indices
    
    @torch.no_grad()
    def quantize_agents(
        self,
        agent_states: torch.Tensor,
        agent_labels: torch.Tensor,
        topk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize agent trajectories to codebook tokens.
        
        For DiffusionDrive, we use the predicted agent trajectories from AgentHead.
        Simplified version: assumes we have future trajectory predictions.
        
        Args:
            agent_states: [B, N, state_dim]
            agent_labels: [B, N]
            topk_idx: [B, K]
            
        Returns:
            tok_veh, tok_cyc, tok_ped: [B, K, T] token indices for each class
            top1_traj_pos: [B, K, T, 2] cumulative positions
        """
        B, K = topk_idx.shape
        T = self.num_poses
        
        # For DiffusionDrive, we use agent query features to get trajectory
        # Simplified: generate trajectory from agent states (positions)
        # In practice, you'd use the agent's predicted future trajectory
        
        # Get top-K agent positions
        tidx = topk_idx[:, :, None].expand(-1, -1, 2)
        # For simplicity, assume static agents or use dummy trajectories
        # In full implementation, use agent motion predictions
        
        # Create dummy trajectories (static for now)
        # TODO: Replace with actual agent trajectory predictions from motion head
        agent_pos = torch.zeros(B, K, T, 2, device=topk_idx.device)
        
        # Match to codebooks (simplified - use same codebook for all)
        # In full implementation, classify agents and use appropriate codebook
        tok_veh = self.match_to_codebook(agent_pos, self.vehicle_codebook)
        tok_cyc = tok_veh.clone()
        tok_ped = tok_veh.clone()
        
        return tok_veh, tok_cyc, tok_ped, agent_pos
    
    def _attn_stack(
        self, 
        ego_q: torch.Tensor, 
        agent_kv: torch.Tensor, 
        bev_feat: torch.Tensor,
        topk_valid: torch.Tensor
    ) -> torch.Tensor:
        """
        AR attention stack.
        
        Args:
            ego_q: [B, M, T, D]
            agent_kv: [B, K, T, D] fixed agent token embeddings
            bev_feat: [B, H*W, D] BEV features
            topk_valid: [B, K]
            
        Returns: ego_q [B, M, T, D]
        """
        B, M, T, D = ego_q.shape
        K = agent_kv.shape[1]
        caus = _causal_mask(T, ego_q.device)
        inv_ag = ~topk_valid  # [B, K], True = ignore
        
        # Pre-expand invalid mask
        inv_ag_bm = inv_ag.unsqueeze(1).expand(-1, M, -1).reshape(B * M, K)
        all_invalid = inv_ag_bm.all(dim=-1, keepdim=True)
        inv_ag_bm = inv_ag_bm & ~all_invalid
        
        # BEV feature preparation
        P = bev_feat.shape[1]
        
        for i in range(self.num_layers):
            # 1. Temporal self-attention (causal)
            eg = ego_q.reshape(B * M, T, D)
            eg2, _ = self.t_attn[i](eg, eg, eg, attn_mask=caus)
            ego_q = self.t_norm[i](eg + eg2).reshape(B, M, T, D)
            
            # 2. Ego-agent cross-attention
            new_ego = []
            for t in range(T):
                q = ego_q[:, :, t, :].reshape(B * M, 1, D)
                kv = (agent_kv[:, :, t, :]
                      .unsqueeze(1).expand(-1, M, -1, -1)
                      .reshape(B * M, K, D))
                out, _ = self.e2a_attn[i](q, kv, kv, key_padding_mask=inv_ag_bm)
                new_ego.append(self.e2a_norm[i](q + out).reshape(B, M, D))
            ego_q = torch.stack(new_ego, dim=2)
            
            # 3. BEV cross-attention
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
            ego_query: [B, 1, D] trajectory query
            agents_query: [B, N, D] agent queries
            bev_feature: [B, D, H, W] BEV feature map
            agent_states: [B, N, state_dim] predicted agent states
            agent_labels: [B, N] agent validity scores
            targets: Optional training targets
            
        Returns:
            dict with trajectory predictions and optionally loss
        """
        B = ego_query.shape[0]
        M = self.ego_fut_mode
        T = self.num_poses
        D = self.embed_dims
        device = ego_query.device
        
        # Flatten BEV features
        bev_flat = bev_feature.flatten(2).permute(0, 2, 1)  # [B, H*W, D]
        bev_feat = self.bev_proj(bev_flat)
        
        # Select top-K agents
        # Agent positions are in agent_states[..., :2]
        agent_positions = agent_states[..., :2]  # [B, N, 2]
        topk_idx, topk_valid = self.select_topk_agents(
            agent_states, agent_labels, agent_positions
        )
        K = topk_idx.shape[1]
        
        # Quantize agents to tokens (simplified)
        # In full implementation, use predicted agent trajectories
        tok_veh, tok_cyc, tok_ped, agent_traj_pos = self.quantize_agents(
            agent_states, agent_labels, topk_idx
        )
        
        # Lookup token embeddings (use vehicle codebook for all agents in simplified version)
        agent_emb = self.vehicle_token_emb(tok_veh)  # [B, K, T, D]
        
        # Add step and role embeddings
        step_e = self.step_emb.weight.view(1, 1, T, D)
        role_a = self.role_emb.weight[1].view(1, 1, 1, D)
        agent_kv = agent_emb + step_e + role_a
        
        # Ego context
        ego_ctx = ego_query[:, 0, :]  # [B, D]
        ego_base = self.ego_ctx_proj(ego_ctx)
        
        if self.training and targets is not None:
            return self.forward_train(
                ego_base, agent_kv, bev_feat, topk_valid, targets
            )
        else:
            return self.forward_test(
                ego_base, agent_kv, bev_feat, topk_valid, B, M, T, D, device
            )
    
    def forward_train(
        self,
        ego_base: torch.Tensor,
        agent_kv: torch.Tensor,
        bev_feat: torch.Tensor,
        topk_valid: torch.Tensor,
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward with teacher forcing.
        
        Args:
            ego_base: [B, D]
            agent_kv: [B, K, T, D]
            bev_feat: [B, P, D]
            topk_valid: [B, K]
            targets: dict containing 'trajectory' [B, T, 3] (x, y, heading)
            
        Returns:
            dict with loss and predictions
        """
        B = targets['trajectory'].shape[0] if 'trajectory' in targets else ego_base.shape[0]
        M = self.ego_fut_mode
        T = self.num_poses
        D = self.embed_dims
        device = ego_base.device
        
        # Get GT trajectory tokens
        # Convert targets to ego coordinate displacements
        gt_traj = targets['trajectory']  # [B, T, 3] or [B, M, T, 3]
        if gt_traj.dim() == 3:
            # Expand to multiple modes for training (all modes get same GT)
            gt_traj = gt_traj.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 3]
        
        # Convert to displacements (per-step deltas)
        gt_pos = gt_traj[..., :2]  # [B, M, T, 2]
        gt_deltas = torch.zeros_like(gt_pos)
        gt_deltas[..., 0, :] = gt_pos[..., 0, :]
        gt_deltas[..., 1:, :] = gt_pos[..., 1:, :] - gt_pos[..., :-1, :]
        
        # Match GT to codebook tokens
        ego_gt_tokens = self.match_to_codebook(
            gt_pos, self.ego_codebook.to(device)
        )  # [B, M, T]
        
        # BOS-shifted teacher forcing
        mode_e = self.ego_mode_emb.weight.view(1, M, 1, D)
        step_e = self.step_emb.weight.view(1, 1, T, D)
        role_e = self.role_emb.weight[0].view(1, 1, 1, D)
        
        # BOS token
        bos = (self.bos_emb(torch.zeros(1, dtype=torch.long, device=device))
               .view(1, 1, D)
               + ego_base.unsqueeze(1))  # [B, 1, D]
        bos = bos.expand(B, M, D)  # [B, M, D]
        
        # GT token embeddings shifted
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
        
        # Get predictions for metrics
        ego_tokens = logits.argmax(-1)  # [B, M, T]
        ego_offsets = self.ego_codebook[ego_tokens]  # [B, M, T, 2]
        ego_pred = ego_offsets.cumsum(dim=2)  # [B, M, T, 2]
        
        return {
            'trajectory_loss': loss,
            'trajectory': ego_pred[:, 0],  # Best mode
            'ego_tokens': ego_tokens,
            'ego_logits': logits,
        }
    
    def forward_test(
        self,
        ego_base: torch.Tensor,
        agent_kv: torch.Tensor,
        bev_feat: torch.Tensor,
        topk_valid: torch.Tensor,
        B: int,
        M: int,
        T: int,
        D: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Inference forward with autoregressive decoding.
        """
        mode_e = self.ego_mode_emb.weight.view(1, M, 1, D)
        step_e = self.step_emb.weight.view(1, 1, T, D)
        role_e = self.role_emb.weight[0].view(1, 1, 1, D)
        
        # BOS
        bos = (self.bos_emb(torch.zeros(1, dtype=torch.long, device=device))
               .view(1, 1, D)
               + ego_base.unsqueeze(1))
        bos = bos.expand(B, M, D)
        
        input_embs = torch.zeros(B, M, T, D, device=device)
        input_embs[:, :, 0, :] = bos
        
        predicted_tokens = []
        for t in range(T):
            ego_q = input_embs + step_e + role_e + mode_e
            ego_out = self._attn_stack(ego_q, agent_kv, bev_feat, topk_valid)
            logit_t = self.ego_token_head(ego_out[:, :, t, :])  # [B, M, V]
            tok_t = logit_t.argmax(-1)  # [B, M]
            predicted_tokens.append(tok_t)
            
            if t < T - 1:
                input_embs[:, :, t + 1, :] = self.ego_token_emb(tok_t)
        
        ego_tokens = torch.stack(predicted_tokens, dim=2)  # [B, M, T]
        ego_offsets = self.ego_codebook[ego_tokens]  # [B, M, T, 2]
        ego_pred = ego_offsets.cumsum(dim=2)  # [B, M, T, 2]
        
        # Add heading (zeros for now, could predict separately)
        heading = torch.zeros(B, M, T, 1, device=device)
        ego_pred_full = torch.cat([ego_pred, heading], dim=-1)  # [B, M, T, 3]
        
        return {
            'trajectory': ego_pred_full[:, 0],  # Return best mode
            'trajectory_modes': ego_pred_full,   # All modes
            'ego_tokens': ego_tokens,
        }
