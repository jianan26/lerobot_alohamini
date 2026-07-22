#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RelativeStateProcessorStep,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import (
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotObservation,
)
from lerobot.types import EnvTransition
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_act import ACTConfig


@ProcessorStepRegistry.register("select_right_arm_processor")
@dataclass
class SelectRightArmProcessorStep(ProcessorStep):
    """Keep selected right-arm dimensions and discard excluded observation keys."""

    state_indices: list[int] = field(default_factory=list)
    action_indices: list[int] = field(default_factory=list)
    excluded_observation_keys: list[str] = field(default_factory=list)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        processed = transition.copy()
        observation = processed.get(TransitionKey.OBSERVATION)
        if observation is not None:
            observation = dict(observation)
            for key in self.excluded_observation_keys:
                observation.pop(key, None)
            if OBS_STATE in observation:
                observation[OBS_STATE] = observation[OBS_STATE][..., self.state_indices]
            processed[TransitionKey.OBSERVATION] = observation

        action = processed.get(TransitionKey.ACTION)
        if action is not None:
            processed[TransitionKey.ACTION] = action[..., self.action_indices]
        return processed

    def get_config(self) -> dict[str, Any]:
        return {
            "state_indices": self.state_indices,
            "action_indices": self.action_indices,
            "excluded_observation_keys": self.excluded_observation_keys,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def get_act_normalization_stats(
    config: ACTConfig, dataset_stats: dict[str, dict[str, Any]] | None
) -> dict[str, dict[str, Any]] | None:
    """Select relative-space statistics for enabled ACT representations."""
    if not config.use_relative_state and not config.use_relative_actions and not config.single_arm:
        return dataset_stats
    if dataset_stats is None:
        if config.use_relative_state or config.use_relative_actions:
            raise ValueError("ACT relative representations require dataset stats from meta/stats.json.")
        return None

    stats = deepcopy(dataset_stats)
    replacements = (
        (config.use_relative_state, OBS_STATE, f"{OBS_STATE}_relative"),
        (config.use_relative_actions, ACTION, "action_relative"),
    )
    for enabled, target_key, relative_key in replacements:
        if not enabled:
            continue
        if relative_key not in stats:
            raise ValueError(
                f"ACT requires '{relative_key}' in meta/stats.json when the corresponding relative option is enabled."
            )
        stats[target_key] = stats[relative_key]
        logging.info("ACT normalizing %s with stats from '%s'.", target_key, relative_key)

    if config.single_arm:
        state_indices = getattr(config, "_single_arm_state_indices", None)
        action_indices = getattr(config, "_single_arm_action_indices", None)
        if state_indices is None or action_indices is None:
            raise ValueError("ACT single_arm preprocessing requires policy creation from dataset metadata.")
        for key, indices in ((OBS_STATE, state_indices), (ACTION, action_indices)):
            if key in stats:
                stats[key] = {
                    stat_name: value if stat_name == "count" else value[..., indices]
                    for stat_name, value in stats[key].items()
                }
    return stats


@dataclass
class ACTDataAugmentationProcessorStep(ObservationProcessorStep):
    """Apply ACT training-time image and state augmentations before normalization."""

    image_keys: list[str] = field(default_factory=list)
    augment_image: bool = False
    augment_state: bool = False
    normalizer: NormalizerProcessorStep | None = field(default=None, repr=False, compare=False)
    _training: bool = field(default=False, init=False, repr=False)

    def train(self, mode: bool = True) -> ACTDataAugmentationProcessorStep:
        self._training = mode
        return self

    @staticmethod
    def _augment_image(image: Tensor, key: str) -> Tensor:
        """Apply mild ResNet-style augmentation to a channels-first image batch in [0, 1]."""
        if "left" not in key and "right" not in key:
            height, width = image.shape[-2:]
            scale = 0.9 + torch.rand(1, device=image.device) * 0.1
            crop_height = int(height * scale)
            crop_width = int(width * scale)
            max_h = height - crop_height
            max_w = width - crop_width
            if max_h > 0 and max_w > 0:
                start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                image = image[:, :, start_h : start_h + crop_height, start_w : start_w + crop_width]
            image = F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)

            angle = torch.rand(1, device=image.device) * 10 - 5
            angle_rad = angle * torch.pi / 180.0
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            grid_x = torch.linspace(-1, 1, width, device=image.device)
            grid_y = torch.linspace(-1, 1, height, device=image.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
            grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
            grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)
            grid = torch.stack(
                [grid_x * cos_a - grid_y * sin_a, grid_x * sin_a + grid_y * cos_a], dim=-1
            )
            image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        brightness_factor = 0.8 + torch.rand(1, device=image.device) * 0.4
        image = image * brightness_factor
        contrast_factor = 0.8 + torch.rand(1, device=image.device) * 0.4
        mean = image.mean(dim=[1, 2, 3], keepdim=True)
        image = (image - mean) * contrast_factor + mean
        saturation_factor = 0.8 + torch.rand(1, device=image.device) * 0.4
        gray = image.mean(dim=1, keepdim=True)
        image = gray + (image - gray) * saturation_factor
        return torch.clamp(image, 0, 1)

    def observation(self, observation: RobotObservation) -> RobotObservation:
        if not self._training:
            return observation

        augmented = dict(observation)
        if self.augment_image:
            for key in self.image_keys:
                if key in augmented:
                    augmented[key] = self._augment_image(augmented[key], key)
        if self.augment_state and OBS_STATE in augmented:
            if self.normalizer is None:
                raise RuntimeError("ACT state augmentation requires a connected normalizer.")
            state_std = self.normalizer._tensor_stats.get(OBS_STATE, {}).get("std")
            if state_std is None:
                raise ValueError("ACT state augmentation requires dataset standard deviations for observation.state.")
            state = augmented[OBS_STATE]
            state_std = state_std.to(device=state.device, dtype=state.dtype)
            augmented[OBS_STATE] = state + torch.randn_like(state) * state_std * 0.01
        return augmented

    def get_config(self) -> dict[str, Any]:
        return {
            "image_keys": self.image_keys,
            "augment_image": self.augment_image,
            "augment_state": self.augment_state,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def reconcile_act_data_augmentation_processor(
    config: ACTConfig, preprocessor: PolicyProcessorPipeline
) -> None:
    """Ensure the ACT augmentation step is present and connected to the normalizer."""
    normalizer_index = next(
        (index for index, step in enumerate(preprocessor.steps) if isinstance(step, NormalizerProcessorStep)), None
    )
    if normalizer_index is None:
        raise ValueError("ACT preprocessor requires a NormalizerProcessorStep.")
    normalizer = preprocessor.steps[normalizer_index]
    assert isinstance(normalizer, NormalizerProcessorStep)

    augmentation_step = next(
        (step for step in preprocessor.steps if isinstance(step, ACTDataAugmentationProcessorStep)), None
    )
    if config.aug and augmentation_step is None:
        augmentation_step = ACTDataAugmentationProcessorStep()
        preprocessor.steps.insert(normalizer_index, augmentation_step)
    if augmentation_step is not None:
        augmentation_step.image_keys = list(config.image_features)
        augmentation_step.augment_image = "image" in config.aug
        augmentation_step.augment_state = "state" in config.aug
        augmentation_step.normalizer = normalizer


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )

    normalization_stats = get_act_normalization_stats(config, dataset_stats)
    normalizer = NormalizerProcessorStep(
        features={**config.input_features, **config.output_features},
        norm_map=config.normalization_mapping,
        stats=normalization_stats,
        device=config.device,
    )
    input_steps = [RenameObservationsProcessorStep(rename_map={})]
    if config.single_arm:
        state_indices = getattr(config, "_single_arm_state_indices", None)
        action_indices = getattr(config, "_single_arm_action_indices", None)
        if state_indices is None or action_indices is None:
            raise ValueError("ACT single_arm preprocessing requires policy creation from dataset metadata.")
        input_steps.append(
            SelectRightArmProcessorStep(
                state_indices=state_indices,
                action_indices=action_indices,
                excluded_observation_keys=["observation.images.wrist_left"],
            )
        )
    if config.use_relative_state:
        input_steps.append(
            RelativeStateProcessorStep(
                enabled=True,
                exclude_joints=config.relative_exclude_joints,
                action_names=config.action_feature_names,
            )
        )
    input_steps.extend(
        [
            AddBatchDimensionProcessorStep(),
            relative_step,
            DeviceProcessorStep(device=config.device),
        ]
    )
    if config.aug:
        input_steps.append(
            ACTDataAugmentationProcessorStep(
                image_keys=list(config.image_features),
                augment_image="image" in config.aug,
                augment_state="state" in config.aug,
                normalizer=normalizer,
            )
        )
    input_steps.append(normalizer)
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=normalization_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
