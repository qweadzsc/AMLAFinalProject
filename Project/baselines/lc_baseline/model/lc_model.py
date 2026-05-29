import argparse
import math
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LCModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        params = {
            "embedding_dim": 128,
            "sqrt_embedding_dim": 128 ** 0.5,
            "num_att_layers": 3,
            "qkv_dim": 16,
            "sqrt_qkv_dim": 16 ** 0.5,
            "num_heads": 8,
            "logit_clipping": 10,
            "ff_hidden_dim": 512,
            "eval_type": "argmax",
            "local_policy_dim": 128,
            "local_k": 10,
            "local_score_weight": 1.0,
            "global_distance_penalty": 0.5,
            "distance_k": None,
            "dar_enabled": 1,
            "dar_k": 20,
            "dar_alpha": 0.5,
            "dar_log_nearest": 1,
            "max_positional_rank": 128,
        }
        params.update(model_params)
        self.model_params = params
        self.encoder = TSP_Encoder(**params)
        self.decoder = TSP_Decoder(**params)
        self.local_policy = LocalPolicyScorer(
            hidden_dim=params["local_policy_dim"],
            local_k=params["local_k"],
            max_positional_rank=params["max_positional_rank"],
        )
        self.encoded_nodes = None
        self.coordinates = None

    def pre_forward(self, reset_state):
        self.coordinates = reset_state.coordinates
        self.encoded_nodes = self.encoder(reset_state.coordinates)
        self.decoder.set_kv(self.encoded_nodes)

    def _prepare_first_step(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        selected = torch.arange(pomo_size, device=self.encoded_nodes.device)[None, :].expand(batch_size, pomo_size)
        encoded_first_node = _get_encoding(self.encoded_nodes, selected)
        self.decoder.set_q1(encoded_first_node)
        prob = torch.ones(size=(batch_size, pomo_size), device=self.encoded_nodes.device)
        return selected, prob

    def _compute_logits(self, state) -> torch.Tensor:
        logits = compute_lc_logits(self, state)
        visited_mask = build_visited_mask_from_ninf(state.ninf_mask)
        local_score = self.local_policy(self.coordinates, state.current_node, visited_mask)
        logits = logits + self.model_params["local_score_weight"] * local_score

        distance_k = self.model_params["distance_k"] or self.model_params["local_k"]
        if self.model_params["global_distance_penalty"] != 0:
            distance_bias = compute_distance_bias(
                coordinates=self.coordinates,
                current_node=state.current_node,
                ninf_mask=state.ninf_mask,
                distance_k=distance_k,
                use_log_nearest=bool(self.model_params["dar_log_nearest"]),
            )
            logits = logits + self.model_params["global_distance_penalty"] * distance_bias

        if bool(self.model_params["dar_enabled"]):
            dar_bias = compute_distance_bias(
                coordinates=self.coordinates,
                current_node=state.current_node,
                ninf_mask=state.ninf_mask,
                distance_k=self.model_params["dar_k"],
                use_log_nearest=bool(self.model_params["dar_log_nearest"]),
            )
            logits = logits + self.model_params["dar_alpha"] * dar_bias
        return logits

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            return self._prepare_first_step(state)

        logits = self._compute_logits(state)
        probs = F.softmax(logits, dim=2)
        if self.training or self.model_params["eval_type"] == "softmax":
            while True:
                with torch.no_grad():
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1)
                    selected = selected.squeeze(dim=1).reshape(batch_size, pomo_size)
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                if (prob != 0).all():
                    break
        else:
            selected = probs.argmax(dim=2)
            prob = None
        return selected, prob


def _get_encoding(encoded_nodes: torch.Tensor, node_index_to_pick: torch.Tensor) -> torch.Tensor:
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=gathering_index)


def build_visited_mask_from_ninf(ninf_mask: torch.Tensor) -> torch.Tensor:
    return torch.isneginf(ninf_mask)


def gather_node_embeddings(encoded_nodes: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
    batch_size, pomo_size = node_index.size()
    embedding_dim = encoded_nodes.size(2)
    gather_idx = node_index[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=gather_idx)


def reshape_by_heads(qkv: Tensor, num_heads: int) -> Tensor:
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, num_heads, -1)
    return q_reshaped.transpose(1, 2)


def compute_lc_logits(model: "LCModel", state) -> torch.Tensor:
    decoder = model.decoder
    encoded_current_node = gather_node_embeddings(model.encoded_nodes, state.current_node)
    num_heads = decoder.model_params["num_heads"]
    embedding_dim = decoder.model_params["embedding_dim"]
    logit_clipping = decoder.model_params["logit_clipping"]

    q0 = reshape_by_heads(decoder.Wq_0(encoded_current_node), num_heads=num_heads)
    q = decoder.q1 + q0
    out_concat = decoder.multi_head_attention(q, decoder.k, decoder.v, rank3_ninf_mask=state.ninf_mask)
    mh_atten_out = decoder.multi_head_combine(out_concat)
    score = torch.matmul(mh_atten_out, decoder.single_head_key)
    score_scaled = score / math.sqrt(embedding_dim)
    score_clipped = logit_clipping * torch.tanh(score_scaled)
    return score_clipped + state.ninf_mask


def compute_distance_bias(
    coordinates: torch.Tensor,
    current_node: torch.Tensor,
    ninf_mask: torch.Tensor,
    distance_k: int,
    use_log_nearest: bool,
) -> torch.Tensor:
    batch_size, pomo_size = current_node.size()
    node_cnt = coordinates.size(1)

    gather_idx = current_node[:, :, None].expand(batch_size, pomo_size, 2)
    current_coords = coordinates.gather(dim=1, index=gather_idx)
    all_coords = coordinates[:, None, :, :].expand(batch_size, pomo_size, node_cnt, 2)
    dist = torch.norm(all_coords - current_coords[:, :, None, :], dim=3)
    dist = dist.clamp_min(1e-6)

    valid_mask = torch.isfinite(ninf_mask)
    masked_dist = dist.masked_fill(~valid_mask, float("inf"))
    effective_k = min(max(int(distance_k), 1), node_cnt)
    nearest_idx = masked_dist.topk(k=effective_k, dim=2, largest=False).indices
    nearest_mask = torch.zeros_like(valid_mask)
    nearest_mask.scatter_(2, nearest_idx, True)
    nearest_mask = nearest_mask & valid_mask

    nearest_bias = -torch.log(dist) if use_log_nearest else -dist
    far_bias = -dist
    bias = torch.where(nearest_mask, nearest_bias, far_bias)
    return bias.masked_fill(~valid_mask, 0.0)


class LocalPolicyScorer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        local_k: int = 10,
        max_positional_rank: int = 128,
        non_neighbor_value: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.local_k = local_k
        self.max_positional_rank = max_positional_rank
        self.non_neighbor_value = non_neighbor_value
        self.rank_embedding = nn.Embedding(max_positional_rank, hidden_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _current_coords(self, coords: torch.Tensor, current_node: torch.Tensor) -> torch.Tensor:
        batch_size, pomo_size = current_node.size()
        gather_idx = current_node[:, :, None].expand(batch_size, pomo_size, 2)
        return coords.gather(dim=1, index=gather_idx)

    def _pairwise_features(self, coords: torch.Tensor, current_node: torch.Tensor):
        batch_size, node_cnt, _ = coords.size()
        pomo_size = current_node.size(1)
        current_coords = self._current_coords(coords, current_node)
        all_coords = coords[:, None, :, :].expand(batch_size, pomo_size, node_cnt, 2)
        rel = all_coords - current_coords[:, :, None, :]
        dist = torch.norm(rel, dim=3, keepdim=True)
        return rel, dist

    def forward(self, coords: torch.Tensor, current_node: torch.Tensor, visited_mask: torch.Tensor) -> torch.Tensor:
        batch_size, node_cnt, _ = coords.size()
        pomo_size = current_node.size(1)
        rel, dist = self._pairwise_features(coords, current_node)
        dist_2d = dist.squeeze(-1)
        available_mask = ~visited_mask
        masked_dist = dist_2d.masked_fill(~available_mask, float("inf"))

        effective_k = min(max(self.local_k, 1), node_cnt)
        neighbor_idx = masked_dist.topk(k=effective_k, dim=2, largest=False).indices
        rank_positions = torch.arange(neighbor_idx.size(2), device=coords.device)
        clipped_rank_positions = rank_positions.clamp(max=self.max_positional_rank - 1)
        rank_embed = self.rank_embedding(clipped_rank_positions)[None, None, :, :].expand(batch_size, pomo_size, -1, -1)

        neighbor_gather_idx = neighbor_idx[..., None].expand(batch_size, pomo_size, neighbor_idx.size(2), 2)
        neighbor_rel = rel.gather(dim=2, index=neighbor_gather_idx)
        neighbor_dist = dist.gather(dim=2, index=neighbor_idx[..., None])
        features = torch.cat(
            [
                neighbor_rel[..., 0:1],
                neighbor_rel[..., 1:2],
                neighbor_dist,
                neighbor_dist.square(),
                1.0 / (neighbor_dist + 1e-6),
            ],
            dim=3,
        )

        hidden = self.feature_proj(features) + rank_embed
        neighbor_scores = self.score_head(hidden).squeeze(-1)
        local_score = torch.full(
            (batch_size, pomo_size, node_cnt),
            fill_value=self.non_neighbor_value,
            dtype=coords.dtype,
            device=coords.device,
        )
        local_score.scatter_(2, neighbor_idx, neighbor_scores)
        local_score = local_score.masked_fill(visited_mask, self.non_neighbor_value)
        return local_score


class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(model_params["num_att_layers"])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return h


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        num_heads = model_params["num_heads"]
        ff_hidden_dim = model_params["ff_hidden_dim"]
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm1 = nn.InstanceNorm1d(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )
        self.norm2 = nn.InstanceNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.mha(x, x, x)
        h = (x + h).transpose(1, 2)
        h = self.norm1(h).transpose(1, 2)
        h2 = self.ff(h)
        h2 = (h + h2).transpose(1, 2)
        h2 = self.norm2(h2).transpose(1, 2)
        return h2


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        num_heads = model_params["num_heads"]
        qkv_dim = model_params["qkv_dim"]
        self.Wq_0 = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wq_1 = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(num_heads * qkv_dim, embedding_dim)
        self.k = None
        self.v = None
        self.single_head_key = None
        self.q1 = None

    def set_kv(self, encoded_jobs: Tensor) -> None:
        num_heads = self.model_params["num_heads"]
        self.k = reshape_by_heads(self.Wk(encoded_jobs), num_heads=num_heads)
        self.v = reshape_by_heads(self.Wv(encoded_jobs), num_heads=num_heads)
        self.single_head_key = encoded_jobs.transpose(1, 2)

    def set_q1(self, encoded_q1: Tensor) -> None:
        num_heads = self.model_params["num_heads"]
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), num_heads=num_heads)

    def multi_head_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        rank2_ninf_mask: Union[Tensor, None] = None,
        rank3_ninf_mask: Union[Tensor, None] = None,
    ) -> Tensor:
        batch_s = q.size(0)
        n = q.size(2)
        node_cnt = k.size(2)
        num_heads = self.model_params["num_heads"]
        qkv_dim = self.model_params["qkv_dim"]
        sqrt_qkv_dim = math.sqrt(qkv_dim)
        score = torch.matmul(q, k.transpose(2, 3))
        score_scaled = score / sqrt_qkv_dim
        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, num_heads, n, node_cnt)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, num_heads, n, node_cnt)
        weights = nn.Softmax(dim=3)(score_scaled)
        out = torch.matmul(weights, v)
        out_transposed = out.transpose(1, 2)
        return out_transposed.reshape(batch_s, n, num_heads * qkv_dim)

    def forward(self, encoded_q0: Tensor, ninf_mask: Tensor) -> Tensor:
        num_heads = self.model_params["num_heads"]
        embedding_dim = self.model_params["embedding_dim"]
        logit_clipping = self.model_params["logit_clipping"]
        q0 = reshape_by_heads(self.Wq_0(encoded_q0), num_heads=num_heads)
        q = self.q1 + q0
        out_concat = self.multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)
        score = torch.matmul(mh_atten_out, self.single_head_key)
        score_scaled = score / math.sqrt(embedding_dim)
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        return F.softmax(score_masked, dim=2)
