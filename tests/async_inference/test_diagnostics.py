import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from lerobot.utils import async_inference_diagnostics as diagnostics_module
from lerobot.utils.async_inference_diagnostics import DiagnosticRecorder, should_sample


def test_should_sample_first_and_interval():
    sampled = [index for index in range(1, 23) if should_sample(index, 10)]
    assert sampled == [1, 11, 21]


def test_diagnostics_disabled_does_not_create_directory(tmp_path):
    output = tmp_path / "disabled"
    recorder = DiagnosticRecorder("test", enabled=False, root_dir=output)

    recorder.record_event("ignored", tensor=torch.ones(2))
    recorder.close()

    assert not output.exists()


def test_relative_path_uses_repository_root_and_writes_json_and_jpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics_module, "repository_root", lambda: tmp_path)
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    recorder = DiagnosticRecorder(
        "policy_server",
        enabled=True,
        root_dir=Path("logs/async_diagnostics"),
        session_id="slow_motion_001",
        manifest={"tensor": torch.tensor([1.0, 2.0])},
    )

    recorder.record_event("inference", action=torch.tensor([3.0, 4.0]))
    recorder.record_observation_sample(
        7,
        state={"state": np.array([5.0, 6.0], dtype=np.float32)},
        images={"observation.images.forward": image},
    )
    recorder.close()

    component_dir = tmp_path / "logs/async_diagnostics/slow_motion_001/policy_server"
    manifest = json.loads((component_dir / "manifest.json").read_text())
    event = json.loads((component_dir / "events.jsonl").read_text().splitlines()[0])
    sample_dir = component_dir / "samples/request_00000007"
    state = json.loads((sample_dir / "state.json").read_text())
    saved_image = cv2.imread(str(sample_dir / "observation.images.forward.jpg"))

    assert manifest["tensor"] == [1.0, 2.0]
    assert event["action"] == [3.0, 4.0]
    assert state["state"] == [5.0, 6.0]
    assert saved_image.shape == image.shape
    assert (
        json.loads((tmp_path / "logs/async_diagnostics/latest_policy_server.json").read_text())["session_id"]
        == "slow_motion_001"
    )


def test_diagnostics_interval_must_be_positive():
    pytest.importorskip("grpc")
    from lerobot.async_inference.configs import PolicyServerConfig

    with pytest.raises(ValueError, match="diagnostics_sample_interval"):
        PolicyServerConfig(diagnostics_sample_interval=0)
