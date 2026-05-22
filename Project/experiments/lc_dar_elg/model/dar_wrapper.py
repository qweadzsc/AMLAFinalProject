import math

import torch
import torch.nn.functional as F


def _reshape_by_heads(qkv: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size = qkv.size(0)
    steps = qkv.size(1)
    q_reshaped = qkv.reshape(batch_size, steps, num_heads, -1)
    return q_reshaped.transpose(1, 2)


def _multi_head_attention(decoder, q, k, v, rank3_ninf_mask=None):
    batch_size = q.size(0)
    num_heads = decoder.model_params["num_heads"]
    steps = q.size(2)
    node_cnt = k.size(2)
    qkv_dim = decoder.model_params["qkv_dim"]
    sqrt_qkv_dim = math.sqrt(qkv_dim)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / sqrt_qkv_dim
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_size, num_heads, steps, node_cnt)
    weights = F.softmax(score_scaled, dim=3)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    return out_transposed.reshape(batch_size, steps, num_heads * qkv_dim)


def gather_node_embeddings(encoded_nodes: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
    batch_size, pomo_size = node_index.size()
    embedding_dim = encoded_nodes.size(2)
    gather_idx = node_index[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=gather_idx)


def compute_lc_logits(model, state) -> torch.Tensor:
    decoder = model.decoder
    encoded_current_node = gather_node_embeddings(model.encoded_nodes, state.current_node)

    num_heads = decoder.model_params["num_heads"]
    embedding_dim = decoder.model_params["embedding_dim"]
    logit_clipping = decoder.model_params["logit_clipping"]

    q0 = _reshape_by_heads(decoder.Wq_0(encoded_current_node), num_heads=num_heads)
    q = decoder.q1 + q0
    out_concat = _multi_head_attention(decoder, q, decoder.k, decoder.v, rank3_ninf_mask=state.ninf_mask)
    mh_atten_out = decoder.multi_head_combine(out_concat)

    score = torch.matmul(mh_atten_out, decoder.single_head_key)
    score_scaled = score / math.sqrt(embedding_dim)
    score_clipped = logit_clipping * torch.tanh(score_scaled)
    return score_clipped + state.ninf_mask


def compute_dar_bias(
    coordinates: torch.Tensor,
    current_node: torch.Tensor,
    ninf_mask: torch.Tensor,
    dar_k: int,
    dar_log_nearest: bool,
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

    effective_k = min(max(dar_k, 1), node_cnt)
    nearest_idx = masked_dist.topk(k=effective_k, dim=2, largest=False).indices
    nearest_mask = torch.zeros_like(valid_mask)
    nearest_mask.scatter_(2, nearest_idx, True)
    nearest_mask = nearest_mask & valid_mask

    if dar_log_nearest:
        nearest_bias = -torch.log(dist)
    else:
        nearest_bias = -dist
    far_bias = -dist

    bias = torch.where(nearest_mask, nearest_bias, far_bias)
    bias = bias.masked_fill(~valid_mask, 0.0)
    return bias


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=2)


def apply_dar_to_logits(
    logits: torch.Tensor,
    coordinates: torch.Tensor,
    current_node: torch.Tensor,
    ninf_mask: torch.Tensor,
    dar_k: int,
    dar_alpha: float,
    dar_log_nearest: bool,
) -> torch.Tensor:
    bias = compute_dar_bias(
        coordinates=coordinates,
        current_node=current_node,
        ninf_mask=ninf_mask,
        dar_k=dar_k,
        dar_log_nearest=dar_log_nearest,
    )
    return logits + dar_alpha * bias
