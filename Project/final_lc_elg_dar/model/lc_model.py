import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dar_wrapper import compute_dar_bias, compute_lc_logits, logits_to_probs
from .local_policy import LocalPolicyScorer, build_visited_mask_from_ninf


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

    def _compute_logits(self, state):
        logits = compute_lc_logits(self, state)
        visited_mask = build_visited_mask_from_ninf(state.ninf_mask)
        local_score = self.local_policy(self.coordinates, state.current_node, visited_mask)
        logits = logits + self.model_params["local_score_weight"] * local_score

        distance_k = self.model_params["distance_k"] or self.model_params["local_k"]
        if self.model_params["global_distance_penalty"] != 0:
            distance_bias = compute_dar_bias(
                coordinates=self.coordinates,
                current_node=state.current_node,
                ninf_mask=state.ninf_mask,
                dar_k=distance_k,
                dar_log_nearest=bool(self.model_params["dar_log_nearest"]),
            )
            logits = logits + self.model_params["global_distance_penalty"] * distance_bias

        if bool(self.model_params["dar_enabled"]):
            dar_bias = compute_dar_bias(
                coordinates=self.coordinates,
                current_node=state.current_node,
                ninf_mask=state.ninf_mask,
                dar_k=self.model_params["dar_k"],
                dar_log_nearest=bool(self.model_params["dar_log_nearest"]),
            )
            logits = logits + self.model_params["dar_alpha"] * dar_bias
        return logits

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            return self._prepare_first_step(state)

        logits = self._compute_logits(state)
        probs = logits_to_probs(logits)
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


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(model_params["num_att_layers"])])

    def forward(self, x):
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

    def forward(self, x):
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

    def forward(self, encoded_q0: Tensor, ninf_mask: Tensor) -> Tensor:
        num_heads = self.model_params["num_heads"]
        embedding_dim = self.model_params["embedding_dim"]
        logit_clipping = self.model_params["logit_clipping"]
        q0 = reshape_by_heads(self.Wq_0(encoded_q0), num_heads=num_heads)
        q = self.q1 + q0
        out_concat = self._multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)
        score = torch.matmul(mh_atten_out, self.single_head_key)
        score_scaled = score / math.sqrt(embedding_dim)
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        return F.softmax(score_masked, dim=2)

    def _multi_head_attention(self, q: Tensor, k: Tensor, v: Tensor, rank3_ninf_mask=None) -> Tensor:
        batch_s = q.size(0)
        n = q.size(2)
        node_cnt = k.size(2)
        num_heads = self.model_params["num_heads"]
        qkv_dim = self.model_params["qkv_dim"]
        sqrt_qkv_dim = math.sqrt(qkv_dim)
        score = torch.matmul(q, k.transpose(2, 3))
        score_scaled = score / sqrt_qkv_dim
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, num_heads, n, node_cnt)
        weights = nn.Softmax(dim=3)(score_scaled)
        out = torch.matmul(weights, v)
        out_transposed = out.transpose(1, 2)
        return out_transposed.reshape(batch_s, n, num_heads * qkv_dim)


def reshape_by_heads(qkv: Tensor, num_heads: int) -> Tensor:
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, num_heads, -1)
    return q_reshaped.transpose(1, 2)
