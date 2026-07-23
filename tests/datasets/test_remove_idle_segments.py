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
    _encoder_configs,
    _letterbox_resize,
    _output_features,
    analyze_dataset,
    plan_episode,
    rebuild_dataset,
    select_state_dimensions,
)
from lerobot.configs import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from tests.datasets.test_video_encoding import require_h264


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


def test_letterbox_resize_preserves_aspect_ratio_with_black_bars():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)

    resized = _letterbox_resize(image)

    assert resized.shape == (224, 224, 3)
    assert np.all(resized[:27] == 0)
    assert np.all(resized[29:195] == 255)
    assert np.all(resized[197:] == 0)


def test_output_features_resizes_all_rgb_but_not_depth():
    class Meta:
        features = {
            "cam_a": {"dtype": "video", "shape": (480, 640, 3), "info": {}},
            "cam_b": {"dtype": "image", "shape": (480, 640, 3)},
            "depth": {
                "dtype": "video",
                "shape": (480, 640, 1),
                "info": {"is_depth_map": True},
            },
        }
        depth_keys = ["depth"]

    class Source:
        meta = Meta()

    features = _output_features(Source(), resize_images=True)

    assert features["cam_a"]["shape"] == (224, 224, 3)
    assert features["cam_a"]["info"]["video.height"] == 224
    assert features["cam_b"]["shape"] == (224, 224, 3)
    assert features["depth"]["shape"] == (480, 640, 1)


@require_h264
def test_encoder_configs_support_source_and_h264_override(tmp_path):
    features = {
        "camera": {"dtype": "video", "shape": (16, 16, 3)},
        "observation.state": {"dtype": "float32", "shape": (1,)},
    }
    source = LeRobotDataset.create(
        repo_id="test/encoder-source",
        root=tmp_path / "encoder-source",
        fps=10,
        features=features,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=27, preset="slow", g=7),
    )
    for index in range(2):
        source.add_frame(
            {
                "camera": np.full((16, 16, 3), index, dtype=np.uint8),
                "observation.state": np.asarray([index], dtype=np.float32),
                "task": "test",
            }
        )
    source.save_episode()
    source.finalize()

    source_rgb, _ = _encoder_configs(
        source, output_video_codec="source", h264_crf=18, h264_preset="fast", h264_gop=30
    )
    h264_rgb, _ = _encoder_configs(
        source, output_video_codec="h264", h264_crf=18, h264_preset="fast", h264_gop=30
    )

    assert (source_rgb.crf, source_rgb.preset, source_rgb.g) == (27, "slow", 7)
    assert (h264_rgb.vcodec, h264_rgb.crf, h264_rgb.preset, h264_rgb.g) == (
        "h264",
        18,
        "fast",
        30,
    )

    plans, _ = analyze_dataset(
        source,
        state_key="observation.state",
        dimension_regex=None,
        dimension_indices=None,
        idle_threshold=0.01,
        min_idle_seconds=0.2,
        smooth_window=1,
    )
    cleaned = rebuild_dataset(
        source,
        plans,
        output_repo_id="test/encoder-cleaned",
        output_root=tmp_path / "encoder-cleaned",
        decode_batch_size=1,
        output_video_codec="h264",
        resize_images=True,
    )

    video_info = cleaned.meta.features["camera"]["info"]
    assert tuple(cleaned.meta.features["camera"]["shape"]) == (224, 224, 3)
    assert (video_info["video.codec"], video_info["video.crf"], video_info["video.g"]) == (
        "h264",
        18,
        30,
    )
    assert video_info["video.preset"] == "fast"
    assert cleaned.meta.total_frames == 2
    np.testing.assert_allclose(cleaned.hf_dataset["timestamp"], [0.0, 0.1])
    assert tuple(cleaned[0]["camera"].shape) == (3, 224, 224)


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
    second_episode_states = [1.0, 1.2, 1.4]
    for arm_state in second_episode_states:
        state = np.asarray([arm_state, 2.0], dtype=np.float32)
        source.add_frame({"action": state.copy(), "observation.state": state, "task": "return arm"})
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
        decode_batch_size=4,
    )

    assert report["source"]["frames"] == len(arm_states) + len(second_episode_states)
    assert report["cleaned_estimate"]["frames"] == 9
    assert cleaned.meta.total_episodes == 2
    assert cleaned.meta.total_frames == 9
    np.testing.assert_allclose(cleaned.hf_dataset["frame_index"], [0, 1, 2, 3, 4, 5, 0, 1, 2])
    np.testing.assert_allclose(cleaned.hf_dataset["timestamp"], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.1, 0.2])
    assert {cleaned[index]["task"] for index in range(len(cleaned))} == {"move arm", "return arm"}
    assert cleaned.meta.stats["observation.state"]["mean"].shape == (2,)


def test_rebuild_rejects_invalid_batch_size(tmp_path):
    features = {"observation.state": {"dtype": "float32", "shape": (1,)}}
    source = LeRobotDataset.create(
        repo_id="test/invalid-batch",
        root=tmp_path / "invalid-batch",
        fps=10,
        features=features,
        use_videos=False,
    )
    source.add_frame({"observation.state": np.zeros(1, dtype=np.float32), "task": "test"})
    source.save_episode()
    source.finalize()

    with pytest.raises(ValueError, match="batch size must be positive"):
        rebuild_dataset(
            source,
            [],
            output_repo_id="test/not-created",
            output_root=tmp_path / "not-created",
            decode_batch_size=0,
        )
