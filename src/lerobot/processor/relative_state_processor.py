# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_STATE

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry, RobotObservation


def to_relative_state(previous_state: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert state to ``previous_state - state`` for masked dimensions."""
    if previous_state.device != state.device or previous_state.dtype != state.dtype:
        previous_state = previous_state.to(device=state.device, dtype=state.dtype)

    relative_state = state.clone()
    dims = min(len(mask), state.shape[-1], previous_state.shape[-1])
    if dims:
        mask_t = torch.tensor(mask[:dims], dtype=torch.bool, device=state.device)
        relative_state[..., :dims] = torch.where(
            mask_t, previous_state[..., :dims] - state[..., :dims], state[..., :dims]
        )
    return relative_state


@dataclass
@ProcessorStepRegistry.register("relative_state_processor")
class RelativeStateProcessorStep(ObservationProcessorStep):
    """Convert state to the previous-state offset while keeping excluded joints absolute."""

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, state_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * state_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * state_dim

        mask = []
        for name in self.action_names[:state_dim]:
            state_name = str(name).lower()
            mask.append(not any(token == state_name or token in state_name for token in exclude_tokens))
        if len(mask) < state_dim:
            mask.extend([True] * (state_dim - len(mask)))
        return mask

    def observation(self, observation: RobotObservation) -> RobotObservation:
        if not self.enabled or OBS_STATE not in observation:
            return observation

        state = observation[OBS_STATE]
        if state.ndim >= 3 and state.shape[-2] == 2:
            previous_state, current_state = state[..., 0, :], state[..., 1, :]
        else:
            current_state = state
            previous_state = self._last_state
            if previous_state is None or previous_state.shape != current_state.shape:
                previous_state = current_state

        processed = dict(observation)
        processed[OBS_STATE] = to_relative_state(
            previous_state, current_state, self._build_mask(current_state.shape[-1])
        )
        self._last_state = current_state.detach().clone()
        return processed

    def reset(self) -> None:
        self._last_state = None

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
