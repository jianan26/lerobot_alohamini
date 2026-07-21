#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for ACT policy processor."""

import tempfile
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    RelativeStateProcessorStep,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_STATE


def create_default_config():
    """Create a default ACT configuration for testing."""
    config = ACTConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    config.device = "cpu"
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(7), "std": torch.ones(7)},
        f"{OBS_STATE}_relative": {"mean": torch.zeros(7), "std": torch.ones(7)},
        ACTION: {"mean": torch.zeros(4), "std": torch.ones(4)},
        "action_relative": {"mean": torch.zeros(4), "std": torch.ones(4)},
    }


def create_relative_stats():
    stats = create_default_stats()
    stats[f"{OBS_STATE}_relative"] = {"mean": torch.ones(7), "std": torch.full((7,), 2.0)}
    stats["action_relative"] = {"mean": torch.full((4,), 3.0), "std": torch.full((4,), 4.0)}
    return stats


def test_make_act_processor_basic():
    """Test basic creation of ACT processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(config, stats)

    # Check processor names
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"

    # Check steps in preprocessor
    assert len(preprocessor.steps) == 5
    assert isinstance(preprocessor.steps[0], RenameObservationsProcessorStep)
    assert isinstance(preprocessor.steps[1], AddBatchDimensionProcessorStep)
    assert isinstance(preprocessor.steps[2], RelativeActionsProcessorStep)
    assert isinstance(preprocessor.steps[3], DeviceProcessorStep)
    assert isinstance(preprocessor.steps[4], NormalizerProcessorStep)

    # Check steps in postprocessor
    assert len(postprocessor.steps) == 3
    assert isinstance(postprocessor.steps[0], UnnormalizerProcessorStep)
    assert isinstance(postprocessor.steps[1], AbsoluteActionsProcessorStep)
    assert isinstance(postprocessor.steps[2], DeviceProcessorStep)


def test_act_relative_actions_roundtrip_with_gripper_excluded():
    """ACT converts selected action dimensions to relative values and restores absolute actions."""
    config = create_default_config()
    config.use_relative_actions = True
    config.action_feature_names = ["joint_1", "joint_2", "joint_3", "gripper"]
    preprocessor, postprocessor = make_act_pre_post_processors(config, create_default_stats())

    observation = {OBS_STATE: torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0])}
    action = torch.tensor([3.0, 5.0, 7.0, 9.0])
    batch = transition_to_batch(create_transition(observation, action))

    processed = preprocessor(batch)
    expected_relative = torch.tensor([[2.0, 3.0, 4.0, 9.0]])
    torch.testing.assert_close(processed[ACTION], expected_relative)
    torch.testing.assert_close(postprocessor(processed[ACTION]), action.unsqueeze(0))


def test_act_relative_actions_disabled_keeps_actions_absolute():
    config = create_default_config()
    config.action_feature_names = ["joint_1", "joint_2", "joint_3", "gripper"]
    preprocessor, postprocessor = make_act_pre_post_processors(config, create_default_stats())

    observation = {OBS_STATE: torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0])}
    action = torch.tensor([3.0, 5.0, 7.0, 9.0])
    processed = preprocessor(transition_to_batch(create_transition(observation, action)))

    torch.testing.assert_close(processed[ACTION], action.unsqueeze(0))
    torch.testing.assert_close(postprocessor(processed[ACTION]), action.unsqueeze(0))


def test_act_relative_actions_reconnect_after_loading(tmp_path):
    config = create_default_config()
    config.use_relative_actions = True
    config.action_feature_names = ["joint_1", "joint_2", "joint_3", "gripper"]
    preprocessor, postprocessor = make_act_pre_post_processors(config, create_default_stats())
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(config, pretrained_path=tmp_path)
    relative_step = next(step for step in loaded_preprocessor.steps if isinstance(step, RelativeActionsProcessorStep))
    absolute_step = next(step for step in loaded_postprocessor.steps if isinstance(step, AbsoluteActionsProcessorStep))
    assert absolute_step.relative_step is relative_step

    observation = {OBS_STATE: torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0])}
    action = torch.tensor([3.0, 5.0, 7.0, 9.0])
    processed = loaded_preprocessor(transition_to_batch(create_transition(observation, action)))
    torch.testing.assert_close(loaded_postprocessor(processed[ACTION]), action.unsqueeze(0))


def test_act_relative_state_uses_previous_frame_and_excludes_gripper():
    config = create_default_config()
    config.use_relative_state = True
    config.action_feature_names = ["joint_1", "joint_2", "joint_3", "gripper"]
    preprocessor, _ = make_act_pre_post_processors(config, create_relative_stats())

    state_window = torch.tensor([[[5.0, 7.0, 9.0, 11.0, 0.0, 0.0, 0.0], [2.0, 3.0, 4.0, 6.0, 0.0, 0.0, 0.0]]])
    batch = transition_to_batch(create_transition({OBS_STATE: state_window}, torch.zeros(1, 4)))

    processed = preprocessor(batch)
    expected = torch.tensor([[1.0, 1.5, 2.0, 2.5, -0.5, -0.5, -0.5]])
    torch.testing.assert_close(processed[OBS_STATE], expected)


def test_act_relative_state_uses_current_state_after_reset():
    config = create_default_config()
    config.use_relative_state = True
    config.action_feature_names = ["joint_1", "joint_2", "joint_3", "gripper"]
    preprocessor, _ = make_act_pre_post_processors(config, create_relative_stats())

    first = preprocessor(transition_to_batch(create_transition({OBS_STATE: torch.tensor([5.0, 7.0, 9.0, 11.0, 0.0, 0.0, 0.0])}, torch.zeros(4))))
    torch.testing.assert_close(first[OBS_STATE], torch.tensor([[-0.5, -0.5, -0.5, 5.0, -0.5, -0.5, -0.5]]))

    second = preprocessor(transition_to_batch(create_transition({OBS_STATE: torch.tensor([2.0, 3.0, 4.0, 6.0, 0.0, 0.0, 0.0])}, torch.zeros(4))))
    torch.testing.assert_close(second[OBS_STATE], torch.tensor([[1.0, 1.5, 2.0, 2.5, -0.5, -0.5, -0.5]]))

    preprocessor.reset()
    reset_first = preprocessor(transition_to_batch(create_transition({OBS_STATE: torch.tensor([2.0, 3.0, 4.0, 6.0, 0.0, 0.0, 0.0])}, torch.zeros(4))))
    torch.testing.assert_close(reset_first[OBS_STATE], torch.tensor([[-0.5, -0.5, -0.5, 2.5, -0.5, -0.5, -0.5]]))


def test_act_relative_representations_use_relative_stats():
    config = create_default_config()
    config.use_relative_state = True
    config.use_relative_actions = True
    preprocessor, postprocessor = make_act_pre_post_processors(config, create_relative_stats())

    normalizer = next(step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep))
    unnormalizer = next(step for step in postprocessor.steps if isinstance(step, UnnormalizerProcessorStep))
    torch.testing.assert_close(normalizer._tensor_stats[OBS_STATE]["mean"], torch.ones(7))
    torch.testing.assert_close(normalizer._tensor_stats[ACTION]["mean"], torch.full((4,), 3.0))
    torch.testing.assert_close(unnormalizer._tensor_stats[ACTION]["mean"], torch.full((4,), 3.0))


def test_act_relative_state_is_added_when_loading_an_older_processor(tmp_path):
    config = create_default_config()
    preprocessor, postprocessor = make_act_pre_post_processors(config, create_default_stats())
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    config.use_relative_state = True
    loaded_preprocessor, _ = make_pre_post_processors(
        config, pretrained_path=tmp_path, dataset_stats=create_relative_stats()
    )

    assert any(isinstance(step, RelativeStateProcessorStep) for step in loaded_preprocessor.steps)
    normalizer = next(step for step in loaded_preprocessor.steps if isinstance(step, NormalizerProcessorStep))
    torch.testing.assert_close(normalizer._tensor_stats[OBS_STATE]["mean"], torch.ones(7))


def test_act_relative_state_requests_only_a_state_window():
    config = create_default_config()
    config.use_relative_state = True
    config.chunk_size = 2
    dataset_meta = SimpleNamespace(
        features={OBS_STATE: {}, "observation.images.top": {}, ACTION: {}},
        fps=10,
    )

    assert resolve_delta_timestamps(config, dataset_meta) == {
        OBS_STATE: [-0.1, 0.0],
        ACTION: [0.0, 0.1],
    }


@pytest.mark.parametrize(
    ("relative_state", "relative_actions", "missing_key"),
    [(True, False, f"{OBS_STATE}_relative"), (False, True, "action_relative")],
)
def test_act_relative_representations_require_relative_stats(relative_state, relative_actions, missing_key):
    config = create_default_config()
    config.use_relative_state = relative_state
    config.use_relative_actions = relative_actions
    stats = create_relative_stats()
    del stats[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        make_act_pre_post_processors(config, stats)


def test_act_processor_normalization():
    """Test that ACT processor correctly normalizes and unnormalizes data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Create test data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is normalized and batched
    assert processed[OBS_STATE].shape == (1, 7)
    assert processed[TransitionKey.ACTION.value].shape == (1, 4)

    # Process action through postprocessor
    postprocessed = postprocessor(processed[TransitionKey.ACTION.value])

    # Check that action is unnormalized
    assert postprocessed.shape == (1, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_cuda():
    """Test ACT processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Create CPU data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is on CUDA
    assert processed[OBS_STATE].device.type == "cuda"
    assert processed[TransitionKey.ACTION.value].device.type == "cuda"

    # Process through postprocessor
    postprocessed = postprocessor(processed[TransitionKey.ACTION.value])

    # Check that action is back on CPU
    assert postprocessed.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_accelerate_scenario():
    """Test ACT processor in simulated Accelerate scenario (data already on GPU)."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Simulate Accelerate: data already on GPU
    device = torch.device("cuda:0")
    observation = {OBS_STATE: torch.randn(1, 7).to(device)}  # Already batched and on GPU
    action = torch.randn(1, 4).to(device)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data stays on same GPU (not moved unnecessarily)
    assert processed[OBS_STATE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_act_processor_multi_gpu():
    """Test ACT processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Simulate data on different GPU (like in multi-GPU training)
    device = torch.device("cuda:1")
    observation = {OBS_STATE: torch.randn(1, 7).to(device)}
    action = torch.randn(1, 4).to(device)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data stays on cuda:1 (not moved to cuda:0)
    assert processed[OBS_STATE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


def test_act_processor_without_stats():
    """Test ACT processor creation without dataset statistics."""
    config = create_default_config()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        dataset_stats=None,
    )

    # Should still create processors, but normalization won't have stats
    assert preprocessor is not None
    assert postprocessor is not None

    # Process should still work (but won't normalize without stats)
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed is not None


def test_act_processor_save_and_load():
    """Test saving and loading ACT processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save preprocessor
        preprocessor.save_pretrained(tmpdir)

        # Load preprocessor
        loaded_preprocessor = DataProcessorPipeline.from_pretrained(
            tmpdir, config_filename="policy_preprocessor.json"
        )

        # Test that loaded processor works
        observation = {OBS_STATE: torch.randn(7)}
        action = torch.randn(4)
        transition = create_transition(observation, action)
        batch = transition_to_batch(transition)

        processed = loaded_preprocessor(batch)
        assert processed[OBS_STATE].shape == (1, 7)
        assert processed[TransitionKey.ACTION.value].shape == (1, 4)


def test_act_processor_device_placement_preservation():
    """Test that ACT processor preserves device placement correctly."""
    config = create_default_config()
    stats = create_default_stats()

    # Test with CPU config
    config.device = "cpu"
    preprocessor, _ = make_act_pre_post_processors(
        config,
        stats,
    )

    # Process CPU data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed[OBS_STATE].device.type == "cpu"
    assert processed[TransitionKey.ACTION.value].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_mixed_precision():
    """Test ACT processor with mixed precision (float16)."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    # Modify the device processor to use float16
    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Replace DeviceProcessorStep with one that uses float16
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="float16"))
        elif isinstance(step, NormalizerProcessorStep):
            # Update normalizer to use the same device as the device processor
            norm_step = step  # Now type checker knows this is NormalizerProcessorStep
            modified_steps.append(
                NormalizerProcessorStep(
                    features=norm_step.features,
                    norm_map=norm_step.norm_map,
                    stats=norm_step.stats,
                    device=config.device,
                    dtype=torch.float16,  # Match the float16 dtype
                )
            )
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Create test data
    observation = {OBS_STATE: torch.randn(7, dtype=torch.float32)}
    action = torch.randn(4, dtype=torch.float32)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is converted to float16
    assert processed[OBS_STATE].dtype == torch.float16
    assert processed[TransitionKey.ACTION.value].dtype == torch.float16


def test_act_processor_batch_consistency():
    """Test that ACT processor handles different batch sizes correctly."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Test single sample (unbatched)
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed[OBS_STATE].shape[0] == 1  # Batched

    # Test already batched data
    observation_batched = {OBS_STATE: torch.randn(8, 7)}  # Batch of 8
    action_batched = torch.randn(8, 4)
    transition_batched = create_transition(observation_batched, action_batched)
    batch_batched = transition_to_batch(transition_batched)

    processed_batched = preprocessor(batch_batched)
    assert processed_batched[OBS_STATE].shape[0] == 8
    assert processed_batched[TransitionKey.ACTION.value].shape[0] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_bfloat16_device_float32_normalizer():
    """Test: DeviceProcessor(bfloat16) + NormalizerProcessor(float32) → output bfloat16 via automatic adaptation"""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, _ = make_act_pre_post_processors(
        config,
        stats,
    )

    # Modify the pipeline to use bfloat16 device processor with float32 normalizer
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            # Device processor converts to bfloat16
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="bfloat16"))
        elif isinstance(step, NormalizerProcessorStep):
            # Normalizer stays configured as float32 (will auto-adapt to bfloat16)
            norm_step = step  # Now type checker knows this is NormalizerProcessorStep
            modified_steps.append(
                NormalizerProcessorStep(
                    features=norm_step.features,
                    norm_map=norm_step.norm_map,
                    stats=norm_step.stats,
                    device=config.device,
                    dtype=torch.float32,  # Deliberately configured as float32
                )
            )
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Verify initial normalizer configuration
    normalizer_step = preprocessor.steps[4]  # NormalizerProcessorStep
    assert normalizer_step.dtype == torch.float32

    # Create test data
    observation = {OBS_STATE: torch.randn(7, dtype=torch.float32)}  # Start with float32
    action = torch.randn(4, dtype=torch.float32)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through full pipeline
    processed = preprocessor(batch)

    # Verify: DeviceProcessor → bfloat16, NormalizerProcessor adapts → final output is bfloat16
    assert processed[OBS_STATE].dtype == torch.bfloat16
    assert processed[TransitionKey.ACTION.value].dtype == torch.bfloat16

    # Verify normalizer automatically adapted its internal state
    assert normalizer_step.dtype == torch.bfloat16
    for stat_tensor in normalizer_step._tensor_stats[OBS_STATE].values():
        assert stat_tensor.dtype == torch.bfloat16
