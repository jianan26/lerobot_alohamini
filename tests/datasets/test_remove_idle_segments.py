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
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from examples.dataset.remove_idle_segments import (
    analyze_dataset,
    plan_episode,
    rebuild_dataset,
    select_state_dimensions,
)
from lerobot.datasets import LeRobotDataset


def test_plan_episode_trims_edges_and_compresses_middle_idle():
    states = np.asarray(
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.5, 0.5],
        dtype=np.float64,
    )[:, None]

    plan = plan_episode(
        states,
        episode_index=0,
        source_from_index=0,
        fps=10,
        scales=np.ones(1),
        idle_threshold=0.01,
        min_idle_seconds=0.2,
        smooth_window=1,
    )

    assert [segment["location"] for segment in plan.segments] == ["leading", "middle", "trailing"]
    np.testing.assert_array_equal(plan.keep_indices, [2, 3, 4, 5, 8, 9, 10])
    assert plan.segments[1]["kept_relative_indices"] == [5, 8]


def test_middle_idle_sampling_follows_joint_space_arc_length():
    states = np.asarray(
        [0.0, 0.1, 0.2, 0.3, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54, 0.64, 0.74],
        dtype=np.float64,
    )[:, None]

    plan = plan_episode(
        states,
        episode_index=0,
        source_from_index=0,
        fps=10,
        scales=np.ones(1),
        idle_threshold=0.05,
        min_idle_seconds=0.3,
        smooth_window=1,
    )

    middle = plan.segments[0]
    assert middle["location"] == "middle"
    assert middle["kept_relative_indices"] == [3, 6, 8, 9]
    selected_states = states[middle["kept_relative_indices"], 0]
    assert np.all(np.diff(selected_states) > 0)
    assert np.max(np.diff(selected_states)) <= plan.target_step * 1.3


def test_short_pause_is_not_cleaned():
    states = np.asarray([0.0, 0.1, 0.2, 0.2, 0.3, 0.4], dtype=np.float64)[:, None]
    plan = plan_episode(
        states,
        episode_index=0,
        source_from_index=0,
        fps=10,
        scales=np.ones(1),
        idle_threshold=0.01,
        min_idle_seconds=0.2,
        smooth_window=1,
    )

    np.testing.assert_array_equal(plan.keep_indices, np.arange(len(states)))
    assert plan.segments == []


def test_select_state_dimensions_by_name():
    feature = {"dtype": "float32", "shape": (5,), "names": ["arm_0", "arm_1", "x.vel", "y.vel", "lift"]}

    indices, names = select_state_dimensions(
        feature,
        dimension_regex="^arm_",
        dimension_indices=None,
    )

    np.testing.assert_array_equal(indices, [0, 1])
    assert names == ["arm_0", "arm_1"]


def test_select_state_dimensions_rejects_unmatched_regex():
    feature = {"dtype": "float32", "shape": (2,), "names": ["joint_0", "joint_1"]}
    with pytest.raises(ValueError, match="matched none"):
        select_state_dimensions(feature, dimension_regex="^arm_", dimension_indices=None)


def test_analyze_and_rebuild_dataset(tmp_path):
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["arm_0", "base.vel"]},
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["arm_0", "base.vel"],
        },
    }
    source = LeRobotDataset.create(
        repo_id="test/source",
        root=tmp_path / "source",
        fps=10,
        features=features,
        use_videos=False,
    )
    arm_states = [0.0, 0.0, 0.0, 0.2, 0.4, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8]
    for arm_state in arm_states:
        state = np.asarray([arm_state, 1.0], dtype=np.float32)
        source.add_frame({"action": state.copy(), "observation.state": state, "task": "move arm"})
    source.save_episode()
    source.finalize()

    plans, report = analyze_dataset(
        source,
        state_key="observation.state",
        dimension_regex="^arm_",
        dimension_indices=None,
        idle_threshold=0.01,
        min_idle_seconds=0.2,
        smooth_window=1,
    )
    cleaned = rebuild_dataset(
        source,
        plans,
        output_repo_id="test/cleaned",
        output_root=tmp_path / "cleaned",
    )

    assert report["source"]["frames"] == len(arm_states)
    assert report["cleaned_estimate"]["frames"] == 6
    assert cleaned.meta.total_episodes == 1
    assert cleaned.meta.total_frames == 6
    np.testing.assert_allclose(cleaned.hf_dataset["frame_index"], np.arange(6))
    np.testing.assert_allclose(cleaned.hf_dataset["timestamp"], np.arange(6) / 10)
    assert {cleaned[index]["task"] for index in range(len(cleaned))} == {"move arm"}
    assert cleaned.meta.stats["observation.state"]["mean"].shape == (2,)
