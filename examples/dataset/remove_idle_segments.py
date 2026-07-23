#!/usr/bin/env python

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

"""Detect and remove long idle sections from a LeRobot dataset.

The detector uses changes in selected dimensions of ``observation.state``.
Leading and trailing idle sections are trimmed. Interior idle sections are
compressed by sampling approximately uniformly in normalized joint-space arc
length, which preserves slow motion while removing repeated stationary frames.

The command is a dry run unless ``--write`` is passed. Source datasets are
never modified.

Example for AlohaMini arm-only cleaning:

    uv run python examples/dataset/remove_idle_segments.py \
        --repo-id user/source --root /data/source \
        --arm-dim-regex '^arm_' --report-path idle_report.json

    uv run python examples/dataset/remove_idle_segments.py \
        --repo-id user/source --root /data/source \
        --output-repo-id user/source_clean --output-root /data/source_clean \
        --arm-dim-regex '^arm_' --resize-images \
        --report-path idle_report.json --write
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.configs import DepthEncoderConfig, RGBEncoderConfig
from lerobot.configs.video import encoder_config_from_video_info
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES


@dataclass
class EpisodePlan:
    episode_index: int
    source_from_index: int
    source_frames: int
    keep_indices: np.ndarray
    segments: list[dict]
    target_step: float
    source_motion: np.ndarray
    cleaned_motion: np.ndarray

    @property
    def kept_frames(self) -> int:
        return int(self.keep_indices.size)


def _median_filter(states: np.ndarray, window: int) -> np.ndarray:
    if window == 1 or len(states) < 2:
        return states.copy()
    half = window // 2
    padded = np.pad(states, ((half, half), (0, 0)), mode="edge")
    return np.stack([np.median(padded[i : i + window], axis=0) for i in range(len(states))])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return true runs as half-open index ranges."""
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _motion_steps(states: np.ndarray) -> np.ndarray:
    if len(states) < 2:
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(np.diff(states, axis=0), axis=1) / math.sqrt(states.shape[1])


def _motion_summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def plan_episode(
    states: np.ndarray,
    *,
    episode_index: int,
    source_from_index: int,
    fps: int,
    scales: np.ndarray,
    idle_threshold: float,
    min_idle_seconds: float,
    smooth_window: int,
) -> EpisodePlan:
    """Build a frame-retention plan for one episode."""
    if states.ndim != 2 or states.shape[1] == 0:
        raise ValueError("states must have shape (frames, selected_dimensions)")

    normalized = states.astype(np.float64) / scales
    smoothed = _median_filter(normalized, smooth_window)
    deltas = np.abs(np.diff(smoothed, axis=0))
    idle_steps = np.max(deltas, axis=1) <= idle_threshold if len(deltas) else np.empty(0, dtype=bool)
    motion_steps = _motion_steps(smoothed)
    active_motion = motion_steps[~idle_steps]
    minimum_target = idle_threshold / math.sqrt(states.shape[1])
    target_step = (
        float(max(np.median(active_motion), minimum_target)) if active_motion.size else minimum_target
    )

    minimum_idle_steps = max(1, math.ceil(min_idle_seconds * fps))
    idle_runs = [(start, end) for start, end in _runs(idle_steps) if end - start >= minimum_idle_steps]
    keep = np.ones(len(states), dtype=bool)
    segments: list[dict] = []

    for step_start, step_end in idle_runs:
        frame_start = step_start
        frame_end = step_end + 1
        if frame_start == 0 and frame_end == len(states):
            location = "entire_episode"
            keep[:] = False
            kept_in_segment: list[int] = []
        elif frame_start == 0:
            location = "leading"
            keep[frame_start : frame_end - 1] = False
            kept_in_segment = [frame_end - 1]
        elif frame_end == len(states):
            location = "trailing"
            keep[frame_start + 1 : frame_end] = False
            kept_in_segment = [frame_start]
        else:
            location = "middle"
            keep[frame_start + 1 : frame_end - 1] = False
            kept_in_segment = [frame_start]
            accumulated = 0.0
            for frame_index in range(frame_start + 1, frame_end - 1):
                accumulated += float(
                    np.linalg.norm(smoothed[frame_index] - smoothed[frame_index - 1])
                    / math.sqrt(states.shape[1])
                )
                if accumulated >= target_step:
                    keep[frame_index] = True
                    kept_in_segment.append(frame_index)
                    accumulated -= target_step
            keep[frame_end - 1] = True
            kept_in_segment.append(frame_end - 1)

        segments.append(
            {
                "location": location,
                "start_frame": int(frame_start),
                "end_frame": int(frame_end),
                "source_frames": int(frame_end - frame_start),
                "source_duration_s": float((step_end - step_start) / fps),
                "kept_frames": len(kept_in_segment),
                "kept_relative_indices": kept_in_segment,
            }
        )

    keep_indices = np.flatnonzero(keep)
    cleaned_motion = _motion_steps(smoothed[keep_indices]) if keep_indices.size >= 2 else np.empty(0)
    return EpisodePlan(
        episode_index=episode_index,
        source_from_index=source_from_index,
        source_frames=len(states),
        keep_indices=keep_indices,
        segments=segments,
        target_step=target_step,
        source_motion=motion_steps,
        cleaned_motion=cleaned_motion,
    )


def select_state_dimensions(
    feature: dict,
    *,
    dimension_regex: str | None,
    dimension_indices: list[int] | None,
) -> tuple[np.ndarray, list[str]]:
    if len(feature["shape"]) != 1:
        raise ValueError(f"State feature must be one-dimensional, got shape={feature['shape']}")
    dimension_count = int(feature["shape"][0])
    names = feature.get("names")
    if names is not None and len(names) != dimension_count:
        raise ValueError("State feature names do not match its shape")

    if dimension_indices is not None:
        indices = np.asarray(dimension_indices, dtype=np.int64)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= dimension_count):
            raise ValueError(f"Dimension indices must be in [0, {dimension_count})")
    elif dimension_regex is not None:
        if names is None:
            raise ValueError("--arm-dim-regex requires names in the state feature metadata")
        pattern = re.compile(dimension_regex)
        indices = np.asarray([i for i, name in enumerate(names) if pattern.search(str(name))], dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"Dimension regex {dimension_regex!r} matched none of: {list(names)}")
    else:
        indices = np.arange(dimension_count, dtype=np.int64)

    selected_names = [str(names[i]) if names is not None else str(i) for i in indices]
    return indices, selected_names


def _robust_scales(dataset: LeRobotDataset, state_key: str, states: np.ndarray) -> tuple[np.ndarray, str]:
    stats = dataset.meta.stats.get(state_key, {}) if dataset.meta.stats else {}
    if "q01" in stats and "q99" in stats:
        q01 = np.asarray(stats["q01"], dtype=np.float64).reshape(-1)
        q99 = np.asarray(stats["q99"], dtype=np.float64).reshape(-1)
        source = "metadata_q01_q99"
    else:
        q01 = np.quantile(states, 0.01, axis=0)
        q99 = np.quantile(states, 0.99, axis=0)
        source = "computed_q01_q99"
    return q99 - q01, source


def analyze_dataset(
    dataset: LeRobotDataset,
    *,
    state_key: str,
    dimension_regex: str | None,
    dimension_indices: list[int] | None,
    idle_threshold: float,
    min_idle_seconds: float,
    smooth_window: int,
) -> tuple[list[EpisodePlan], dict]:
    if state_key not in dataset.meta.features:
        raise ValueError(f"State key {state_key!r} not found in dataset features")
    if idle_threshold <= 0 or min_idle_seconds <= 0:
        raise ValueError("idle threshold and minimum idle duration must be positive")
    if smooth_window < 1 or smooth_window % 2 == 0:
        raise ValueError("smooth window must be a positive odd integer")

    selected, selected_names = select_state_dimensions(
        dataset.meta.features[state_key],
        dimension_regex=dimension_regex,
        dimension_indices=dimension_indices,
    )
    all_states = np.asarray(dataset.hf_dataset.with_format(None)[state_key], dtype=np.float64)
    state_dimensions = int(dataset.meta.features[state_key]["shape"][0])
    all_states = all_states.reshape(-1, state_dimensions)
    all_scales, scale_source = _robust_scales(dataset, state_key, all_states)
    selected_scales = all_scales[selected]
    nonconstant = selected_scales > np.finfo(np.float64).eps
    excluded_names = [name for name, valid in zip(selected_names, nonconstant, strict=True) if not valid]
    if np.any(nonconstant):
        selected = selected[nonconstant]
        selected_scales = selected_scales[nonconstant]
        selected_names = [name for name, valid in zip(selected_names, nonconstant, strict=True) if valid]
    else:
        # A dataset whose selected joints never move is a valid all-idle input.
        selected_scales = np.ones_like(selected_scales)

    plans = []
    for episode_index in range(dataset.meta.total_episodes):
        episode = dataset.meta.episodes[episode_index]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        plans.append(
            plan_episode(
                all_states[start:end, selected],
                episode_index=episode_index,
                source_from_index=start,
                fps=dataset.meta.fps,
                scales=selected_scales,
                idle_threshold=idle_threshold,
                min_idle_seconds=min_idle_seconds,
                smooth_window=smooth_window,
            )
        )

    kept_plans = [plan for plan in plans if plan.kept_frames >= 2]
    source_motion_parts = [plan.source_motion for plan in plans if plan.source_motion.size]
    cleaned_motion_parts = [plan.cleaned_motion for plan in kept_plans if plan.cleaned_motion.size]
    source_motion = np.concatenate(source_motion_parts) if source_motion_parts else np.empty(0)
    cleaned_motion = np.concatenate(cleaned_motion_parts) if cleaned_motion_parts else np.empty(0)
    source_frames = sum(plan.source_frames for plan in plans)
    cleaned_frames = sum(plan.kept_frames for plan in kept_plans)
    report = {
        "config": {
            "state_key": state_key,
            "selected_dimensions": selected_names,
            "excluded_zero_range_dimensions": excluded_names,
            "scale_source": scale_source,
            "idle_threshold": idle_threshold,
            "min_idle_seconds": min_idle_seconds,
            "smooth_window": smooth_window,
        },
        "source": {
            "repo_id": dataset.repo_id,
            "root": str(dataset.root),
            "episodes": dataset.meta.total_episodes,
            "frames": source_frames,
            "duration_s": source_frames / dataset.meta.fps,
            "motion": _motion_summary(source_motion),
        },
        "cleaned_estimate": {
            "episodes": len(kept_plans),
            "frames": cleaned_frames,
            "duration_s": cleaned_frames / dataset.meta.fps,
            "retained_fraction": cleaned_frames / source_frames if source_frames else 0.0,
            "motion": _motion_summary(cleaned_motion),
        },
        "episodes": [
            {
                "episode_index": plan.episode_index,
                "source_frames": plan.source_frames,
                "kept_frames": plan.kept_frames,
                "removed_frames": plan.source_frames - plan.kept_frames,
                "skipped": plan.kept_frames < 2,
                "target_normalized_step": plan.target_step,
                "segments": plan.segments,
            }
            for plan in plans
        ],
    }
    return plans, report


def _encoder_configs(
    dataset: LeRobotDataset,
    *,
    output_video_codec: str,
    h264_crf: int,
    h264_preset: str,
    h264_gop: int,
) -> tuple[RGBEncoderConfig | None, DepthEncoderConfig | None]:
    rgb_encoder = None
    depth_encoder = None
    for key in dataset.meta.video_keys:
        config = encoder_config_from_video_info(dataset.meta.features[key].get("info"))
        if isinstance(config, DepthEncoderConfig) and depth_encoder is None:
            depth_encoder = config
        elif isinstance(config, RGBEncoderConfig) and rgb_encoder is None:
            rgb_encoder = config
    if output_video_codec == "h264" and rgb_encoder is not None:
        rgb_encoder = RGBEncoderConfig(
            vcodec="h264", pix_fmt="yuv420p", crf=h264_crf, preset=h264_preset, g=h264_gop
        )
    return rgb_encoder, depth_encoder


def _letterbox_resize(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Pad an HWC image to square with black pixels, then resize it."""
    height, width = image.shape[:2]
    square_size = max(height, width)
    canvas = np.zeros((square_size, square_size, image.shape[2]), dtype=image.dtype)
    top = (square_size - height) // 2
    left = (square_size - width) // 2
    canvas[top : top + height, left : left + width] = image
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _output_features(source: LeRobotDataset, resize_images: bool) -> dict:
    features = {
        key: deepcopy(value) for key, value in source.meta.features.items() if key not in DEFAULT_FEATURES
    }
    if not resize_images:
        return features
    for key, feature in features.items():
        if feature["dtype"] not in {"image", "video"} or key in source.meta.depth_keys:
            continue
        feature["shape"] = (224, 224, 3)
        info = feature.get("info")
        if info is not None:
            info["video.height"] = 224
            info["video.width"] = 224
    return features


def _as_hwc(value: torch.Tensor | np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 3 and value.shape == (expected_shape[2], expected_shape[0], expected_shape[1]):
        value = np.transpose(value, (1, 2, 0))
    return value


def _batch_writer_frames(
    dataset: LeRobotDataset,
    indices: list[int],
    features: dict,
    *,
    episode_index: int,
    resize_images: bool,
) -> list[dict]:
    rows = dataset.hf_dataset[indices]
    video_frames: dict[str, torch.Tensor] = {}
    if dataset.meta.video_keys:
        timestamps = [float(timestamp) for timestamp in rows["timestamp"]]
        query_timestamps = dict.fromkeys(dataset.meta.video_keys, timestamps)
        video_frames = dataset.reader._query_videos(query_timestamps, episode_index)
        video_frames = {
            key: value.unsqueeze(0) if value.ndim == 3 else value for key, value in video_frames.items()
        }

    frames = []
    for batch_index in range(len(indices)):
        task_index = int(rows["task_index"][batch_index])
        frame = {"task": dataset.meta.tasks.iloc[task_index].name}
        for key, feature in features.items():
            value = video_frames[key][batch_index] if key in video_frames else rows[key][batch_index]
            if feature["dtype"] in {"image", "video"}:
                value = _as_hwc(value, tuple(dataset.meta.features[key]["shape"]))
                if key not in dataset.meta.depth_keys and np.issubdtype(value.dtype, np.floating):
                    value = np.rint(np.clip(value, 0.0, 1.0) * 255).astype(np.uint8)
                if resize_images and key not in dataset.meta.depth_keys:
                    value = _letterbox_resize(value)
            elif isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            if feature["dtype"] not in {"image", "video"} and tuple(feature["shape"]) == (1,):
                value = np.asarray(value).reshape(1)
            frame[key] = value
        frames.append(frame)
    return frames


def rebuild_dataset(
    source: LeRobotDataset,
    plans: list[EpisodePlan],
    *,
    output_repo_id: str,
    output_root: Path,
    decode_batch_size: int = 32,
    image_writer_threads: int = 8,
    output_video_codec: str = "h264",
    h264_crf: int = 18,
    h264_preset: str = "fast",
    h264_gop: int = 30,
    resize_images: bool = False,
) -> LeRobotDataset:
    if decode_batch_size <= 0:
        raise ValueError("decode batch size must be positive")
    if image_writer_threads < 0:
        raise ValueError("image writer threads must be non-negative")
    if output_video_codec not in {"h264", "source"}:
        raise ValueError("output video codec must be 'h264' or 'source'")

    features = _output_features(source, resize_images)
    rgb_encoder, depth_encoder = _encoder_configs(
        source,
        output_video_codec=output_video_codec,
        h264_crf=h264_crf,
        h264_preset=h264_preset,
        h264_gop=h264_gop,
    )
    cleaned = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=output_root,
        fps=source.meta.fps,
        features=features,
        robot_type=source.meta.robot_type,
        use_videos=bool(source.meta.video_keys),
        image_writer_threads=image_writer_threads,
        rgb_encoder=rgb_encoder,
        depth_encoder=depth_encoder,
        data_files_size_in_mb=source.meta.data_files_size_in_mb,
        video_files_size_in_mb=source.meta.video_files_size_in_mb,
        video_backend=source._video_backend,
    )
    try:
        for plan in plans:
            if plan.kept_frames < 2:
                continue
            absolute_indices = plan.source_from_index + plan.keep_indices
            for batch_start in range(0, plan.kept_frames, decode_batch_size):
                batch_indices = absolute_indices[batch_start : batch_start + decode_batch_size].tolist()
                frames = _batch_writer_frames(
                    source,
                    batch_indices,
                    features,
                    episode_index=plan.episode_index,
                    resize_images=resize_images,
                )
                for frame in frames:
                    cleaned.add_frame(frame)
            cleaned.save_episode()
    finally:
        cleaned.finalize()
    return LeRobotDataset(
        output_repo_id, root=output_root, return_uint8=True, video_backend=source._video_backend
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Source dataset repository ID")
    parser.add_argument("--root", type=Path, help="Optional local source dataset root")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--video-backend", default="pyav", help="Video decoder used while rebuilding")
    dimensions = parser.add_mutually_exclusive_group()
    dimensions.add_argument("--arm-dim-regex", help="Regex matched against state dimension names")
    dimensions.add_argument("--arm-dim-indices", type=int, nargs="+", help="Explicit state dimension indices")
    parser.add_argument("--idle-threshold", type=float, default=0.001)
    parser.add_argument("--min-idle-seconds", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--output-video-codec", choices=("h264", "source"), default="h264")
    parser.add_argument("--h264-crf", type=int, default=18)
    parser.add_argument("--h264-preset", default="fast")
    parser.add_argument("--h264-gop", type=int, default=30)
    parser.add_argument("--resize-images", action="store_true")
    parser.add_argument("--report-path", type=Path, default=Path("idle_cleaning_report.json"))
    parser.add_argument("--write", action="store_true", help="Create the cleaned dataset")
    parser.add_argument("--output-repo-id", help="Required with --write")
    parser.add_argument("--output-root", type=Path, help="Required with --write; must not exist")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.write and (args.output_repo_id is None or args.output_root is None):
        raise ValueError("--output-repo-id and --output-root are required with --write")
    if args.write and args.output_root.exists():
        raise FileExistsError(f"Output root already exists: {args.output_root}")
    if args.decode_batch_size <= 0:
        raise ValueError("--decode-batch-size must be positive")
    if args.image_writer_threads < 0:
        raise ValueError("--image-writer-threads must be non-negative")

    source = LeRobotDataset(args.repo_id, root=args.root, return_uint8=True, video_backend=args.video_backend)
    plans, report = analyze_dataset(
        source,
        state_key=args.state_key,
        dimension_regex=args.arm_dim_regex,
        dimension_indices=args.arm_dim_indices,
        idle_threshold=args.idle_threshold,
        min_idle_seconds=args.min_idle_seconds,
        smooth_window=args.smooth_window,
    )

    report["config"].update(
        {
            "video_backend": args.video_backend,
            "decode_batch_size": args.decode_batch_size,
            "image_writer_threads": args.image_writer_threads,
            "output_video_codec": args.output_video_codec,
            "h264_crf": args.h264_crf,
            "h264_preset": args.h264_preset,
            "h264_gop": args.h264_gop,
            "resize_images": args.resize_images,
            "output_resolution": {
                key: list((224, 224) if args.resize_images else feature["shape"][:2])
                for key, feature in source.meta.features.items()
                if feature["dtype"] in {"image", "video"} and key not in source.meta.depth_keys
            },
        }
    )

    if args.write:
        write_started = time.perf_counter()
        cleaned = rebuild_dataset(
            source,
            plans,
            output_repo_id=args.output_repo_id,
            output_root=args.output_root,
            decode_batch_size=args.decode_batch_size,
            image_writer_threads=args.image_writer_threads,
            output_video_codec=args.output_video_codec,
            h264_crf=args.h264_crf,
            h264_preset=args.h264_preset,
            h264_gop=args.h264_gop,
            resize_images=args.resize_images,
        )
        write_seconds = time.perf_counter() - write_started
        report["cleaned_actual"] = {
            "repo_id": cleaned.repo_id,
            "root": str(cleaned.root),
            "episodes": cleaned.meta.total_episodes,
            "frames": cleaned.meta.total_frames,
            "duration_s": cleaned.meta.total_frames / cleaned.meta.fps,
            "write_seconds": write_seconds,
            "effective_fps": cleaned.meta.total_frames / write_seconds if write_seconds else None,
        }
        estimate = report["cleaned_estimate"]
        if (
            cleaned.meta.total_episodes != estimate["episodes"]
            or cleaned.meta.total_frames != estimate["frames"]
        ):
            raise RuntimeError("Written dataset counts do not match the cleaning plan")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    estimate = report["cleaned_estimate"]
    print(
        f"Source: {report['source']['episodes']} episodes, {report['source']['frames']} frames\n"
        f"Cleaned: {estimate['episodes']} episodes, {estimate['frames']} frames "
        f"({estimate['retained_fraction']:.1%} retained)\n"
        f"Report: {args.report_path}"
    )


if __name__ == "__main__":
    main()
