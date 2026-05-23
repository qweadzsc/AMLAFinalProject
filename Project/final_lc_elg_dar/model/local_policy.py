import torch
import torch.nn as nn


def build_visited_mask_from_ninf(ninf_mask: torch.Tensor) -> torch.Tensor:
    return torch.isneginf(ninf_mask)


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

    def _select_local_neighbors(self, dist: torch.Tensor, visited_mask: torch.Tensor):
        batch_size, pomo_size, node_cnt, _ = dist.size()
        dist_2d = dist.squeeze(-1)
        available_mask = ~visited_mask
        masked_dist = dist_2d.masked_fill(~available_mask, float("inf"))

        effective_k = min(max(self.local_k, 1), node_cnt)
        neighbor_idx = masked_dist.topk(k=effective_k, dim=2, largest=False).indices
        neighbor_mask = torch.zeros_like(available_mask)
        neighbor_mask.scatter_(2, neighbor_idx, True)
        neighbor_mask = neighbor_mask & available_mask

        return neighbor_idx, neighbor_mask

    def forward(self, coords: torch.Tensor, current_node: torch.Tensor, visited_mask: torch.Tensor) -> torch.Tensor:
        batch_size, node_cnt, _ = coords.size()
        pomo_size = current_node.size(1)

        rel, dist = self._pairwise_features(coords, current_node)
        neighbor_idx, _ = self._select_local_neighbors(dist, visited_mask)

        rank_positions = torch.arange(neighbor_idx.size(2), device=coords.device)
        clipped_rank_positions = rank_positions.clamp(max=self.max_positional_rank - 1)
        rank_embed = self.rank_embedding(clipped_rank_positions)
        rank_embed = rank_embed[None, None, :, :].expand(batch_size, pomo_size, -1, -1)

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
