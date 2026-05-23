from dataclasses import dataclass
from typing import Any, Tuple

import torch
from torch import Tensor


@dataclass
class Reset_State:
    problems: torch.Tensor
    coordinates: torch.Tensor = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None


class TSPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.node_cnt = env_params["node_cnt"]
        self.pomo_size = env_params["pomo_size"]
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.problems = None
        self.coordinates = None
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.step_state = None

    def load_problems(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.coordinates = torch.rand(size=(batch_size, self.node_cnt, 2))
        self.problems = torch.cdist(self.coordinates, self.coordinates, p=2)

    def load_problems_manual(self, problems: Tensor, coordinates: Tensor = None) -> None:
        self.batch_size = problems.size(0)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.problems = problems
        self.coordinates = coordinates

    def reset(self) -> Tuple[Reset_State, None, Any]:
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self._create_step_state()
        return Reset_State(problems=self.problems, coordinates=self.coordinates), None, False

    def _create_step_state(self) -> None:
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.node_cnt))

    def pre_step(self) -> Tuple[Step_State, None, Any]:
        return self.step_state, None, False

    def step(self, node_idx: Tensor) -> Tuple[Reset_State, Tensor, Tensor]:
        self.selected_count += 1
        self.current_node = node_idx
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        self._update_step_state()
        done = self.selected_count == self.node_cnt
        reward = -self._get_total_distance() if done else None
        return self.step_state, reward, done

    def _update_step_state(self) -> None:
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float("-inf")

    def _get_total_distance(self) -> Tensor:
        node_from = self.selected_node_list
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.node_cnt)
        selected_cost = self.problems[batch_index, node_from, node_to]
        return selected_cost.sum(2)
