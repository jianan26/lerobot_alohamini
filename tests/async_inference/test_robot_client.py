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
"""Unit-tests for the `RobotClient` action-queue logic (pure Python, no gRPC).

We monkey-patch `lerobot.robots.utils.make_robot_from_config` so that
no real hardware is accessed. Only the queue-update mechanism is verified.
"""

from __future__ import annotations

import pickle  # nosec
import threading
import time
from queue import Queue

import numpy as np
import pytest
import torch

# Skip entire module if required deps are not available
pytest.importorskip("grpc")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def robot_client():
    """Fresh `RobotClient` instance for each test case (no threads started).
    Uses DummyRobot."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from tests.mocks.mock_robot import MockRobotConfig

    test_config = MockRobotConfig()

    # gRPC channel is not actually used in tests, so using a dummy address
    test_config = RobotClientConfig(
        robot=test_config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
    )

    client = RobotClient(test_config)

    # Initialize attributes that are normally set in start() method
    client.chunks_received = 0
    client.available_actions_size = []

    yield client

    if client.robot.is_connected:
        client.stop()


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_actions(start_ts: float, start_t: int, count: int):
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    from lerobot.async_inference.helpers import TimedAction

    fps = 30  # emulates most common frame-rate
    actions = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(action=action_tensor, timestep=timestep, timestamp=timestamp))
    return actions


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_update_action_queue_discards_stale(robot_client):
    """`_update_action_queue` must drop actions with `timestep` <= `latest_action`."""

    # Pretend we already executed up to action #4
    robot_client.latest_action = 4

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    robot_client._aggregate_action_queues(incoming)

    # Extract timesteps from queue
    resulting_timesteps = [a.get_timestep() for a in robot_client.action_queue.queue]

    assert resulting_timesteps == [5, 6, 7]


@pytest.mark.parametrize(
    "weight_old, weight_new",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.1, 0.9),
        (0.9, 0.1),
    ],
)
def test_aggregate_action_queues_combines_actions_in_overlap(
    robot_client, weight_old: float, weight_new: float
):
    """`_aggregate_action_queues` must combine actions on overlapping timesteps according
    to the provided aggregate_fn, here tested with multiple coefficients."""
    from lerobot.async_inference.helpers import TimedAction

    robot_client.chunks_received = 0

    # Pretend we already executed up to action #4, and queue contains actions for timesteps 5..6
    robot_client.latest_action = 4
    current_actions = _make_actions(
        start_ts=time.time(), start_t=5, count=2
    )  # actions are [torch.ones(6), torch.ones(6), ...]
    current_actions = [
        TimedAction(action=10 * a.get_action(), timestep=a.get_timestep(), timestamp=a.get_timestamp())
        for a in current_actions
    ]

    for a in current_actions:
        robot_client.action_queue.put(a)

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    overlap_timesteps = [5, 6]  # properly tested in test_aggregate_action_queues_discards_stale
    nonoverlap_timesteps = [7]

    robot_client._aggregate_action_queues(
        incoming, aggregate_fn=lambda x1, x2: weight_old * x1 + weight_new * x2
    )

    queue_overlap_actions = []
    queue_non_overlap_actions = []
    for a in robot_client.action_queue.queue:
        if a.get_timestep() in overlap_timesteps:
            queue_overlap_actions.append(a)
        elif a.get_timestep() in nonoverlap_timesteps:
            queue_non_overlap_actions.append(a)

    queue_overlap_actions = sorted(queue_overlap_actions, key=lambda x: x.get_timestep())
    queue_non_overlap_actions = sorted(queue_non_overlap_actions, key=lambda x: x.get_timestep())

    assert torch.allclose(
        queue_overlap_actions[0].get_action(),
        weight_old * current_actions[0].get_action() + weight_new * incoming[-3].get_action(),
    )
    assert torch.allclose(
        queue_overlap_actions[1].get_action(),
        weight_old * current_actions[1].get_action() + weight_new * incoming[-2].get_action(),
    )
    assert torch.allclose(queue_non_overlap_actions[0].get_action(), incoming[-1].get_action())


def test_aggregate_action_queues_preserves_source_request_ids(robot_client):
    from lerobot.async_inference.helpers import TimedAction

    old_action = TimedAction(
        timestamp=0,
        timestep=5,
        action=torch.zeros(6),
        source_request_ids=(10,),
    )
    new_action = TimedAction(
        timestamp=1,
        timestep=5,
        action=torch.ones(6),
        source_request_ids=(11,),
    )
    robot_client.action_queue.put(old_action)

    robot_client._aggregate_action_queues([new_action], lambda old, new: (old + new) / 2)

    aggregated = robot_client.action_queue.get_nowait()
    assert aggregated.source_request_ids == (10, 11)
    torch.testing.assert_close(aggregated.get_action(), torch.full((6,), 0.5))


@pytest.mark.parametrize(
    "chunk_size, queue_len, expected",
    [
        (20, 12, False),  # 12 / 20 = 0.6  > g=0.5 threshold, not ready to send
        (20, 8, True),  # 8  / 20 = 0.4 <= g=0.5, ready to send
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation(robot_client, chunk_size: int, queue_len: int, expected: bool):
    """Validate `_ready_to_send_observation` ratio logic for various sizes."""

    robot_client.action_chunk_size = chunk_size

    # Clear any existing actions then fill with `queue_len` dummy entries ----
    robot_client.action_queue = Queue()

    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


@pytest.mark.parametrize(
    "g_threshold, expected",
    [
        # The condition is `queue_size / chunk_size <= g`.
        # Here, ratio = 6 / 10 = 0.6.
        (0.0, False),  # 0.6 <= 0.0 is False
        (0.1, False),
        (0.2, False),
        (0.3, False),
        (0.4, False),
        (0.5, False),
        (0.6, True),  # 0.6 <= 0.6 is True
        (0.7, True),
        (0.8, True),
        (0.9, True),
        (1.0, True),
    ],
)
def test_ready_to_send_observation_with_varying_threshold(robot_client, g_threshold: float, expected: bool):
    """Validate `_ready_to_send_observation` with fixed sizes and varying `g`."""
    # Fixed sizes for this test: ratio = 6 / 10 = 0.6
    chunk_size = 10
    queue_len = 6

    robot_client.action_chunk_size = chunk_size
    # This is the parameter we are testing
    robot_client._chunk_size_threshold = g_threshold

    # Fill queue with dummy actions
    robot_client.action_queue = Queue()
    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


def test_relative_state_caches_unsent_control_cycle(robot_client, monkeypatch):
    observations = iter(
        [
            {"motor_1.pos": 1.0, "motor_2.pos": 2.0, "motor_3.pos": 3.0},
            {"motor_1.pos": 4.0, "motor_2.pos": 5.0, "motor_3.pos": 6.0},
        ]
    )
    sent_observations = []
    robot_client.config.use_relative_state = True
    monkeypatch.setattr(robot_client.robot, "get_observation", lambda: next(observations))
    monkeypatch.setattr(
        robot_client,
        "send_observation",
        lambda observation: sent_observations.append(observation) or True,
    )

    robot_client.control_loop_observation("task", send=False)
    current_raw_observation = robot_client.control_loop_observation("task", send=True)

    assert current_raw_observation["motor_1.pos"] == 4.0
    assert len(sent_observations) == 1
    torch.testing.assert_close(sent_observations[0].get_previous_state(), torch.tensor([[1.0, 2.0, 3.0]]))


# -----------------------------------------------------------------------------
# Regression test: robot type registry populated by robot_client imports
# -----------------------------------------------------------------------------


def test_robot_client_registers_builtin_robot_types():
    """Importing robot_client must populate RobotConfig's ChoiceRegistry.

    This is a regression test for a bug introduced in #2425, where removing
    robot module imports from robot_client.py caused RobotConfig's registry to
    be empty, breaking CLI argument parsing with:
      error: argument --robot.type: invalid choice: 'so101_follower' (choose from )

    Robot types are registered via @RobotConfig.register_subclass() decorators
    at import time, so all supported modules must be explicitly imported.
    """
    import lerobot.async_inference.robot_client  # noqa: F401
    from lerobot.robots.config import RobotConfig

    known_choices = RobotConfig.get_known_choices()

    expected_robot_types = [
        "so100_follower",
        "so101_follower",
        "koch_follower",
        "omx_follower",
        "bi_so_follower",
    ]
    for robot_type in expected_robot_types:
        assert robot_type in known_choices, (
            f"Robot type '{robot_type}' is not registered in RobotConfig's ChoiceRegistry. "
            f"Ensure the corresponding module is imported in robot_client.py. "
            f"Known choices: {sorted(known_choices)}"
        )


def _action_validation_client(action_key_sets):
    import logging
    import threading

    from lerobot.async_inference.robot_client import RobotClient

    client = object.__new__(RobotClient)
    client._action_key_sets = action_key_sets
    client._default_action_keys = ()
    client._uses_configured_action_key_sets = True
    client._active_action_dim = None
    client.action_queue = Queue()
    client.action_queue_lock = threading.Lock()
    client.shutdown_event = threading.Event()
    client.logger = logging.getLogger("test_action_validation_client")
    return client


def test_action_key_sets_support_subset_actions():
    from lerobot.async_inference.helpers import TimedAction

    action_keys = tuple(f"joint_{index}.pos" for index in range(14))
    client = _action_validation_client(
        {14: action_keys, 18: (*action_keys, "x.vel", "y.vel", "theta.vel", "lift")}
    )
    action = torch.arange(14, dtype=torch.float32)

    assert client._accept_action_chunk([TimedAction(timestamp=0, timestep=0, action=action)])
    assert client._action_tensor_to_action_dict(action) == {
        key: float(index) for index, key in enumerate(action_keys)
    }


def test_action_key_sets_reject_unknown_or_changed_dimensions():
    from lerobot.async_inference.helpers import TimedAction

    client = _action_validation_client(
        {
            14: tuple(f"joint_{index}.pos" for index in range(14)),
            18: tuple(f"key_{index}" for index in range(18)),
        }
    )

    assert client._accept_action_chunk([TimedAction(timestamp=0, timestep=0, action=torch.zeros(14))])
    assert not client._accept_action_chunk([TimedAction(timestamp=1, timestep=1, action=torch.zeros(18))])
    assert client.shutdown_event.is_set()


def test_action_key_sets_reject_unsupported_dimension():
    from lerobot.async_inference.helpers import TimedAction

    client = _action_validation_client({14: tuple(f"joint_{index}.pos" for index in range(14))})

    assert not client._accept_action_chunk([TimedAction(timestamp=0, timestep=0, action=torch.zeros(15))])
    assert client.shutdown_event.is_set()


def test_alohamini_async_config_builds_single_and_bimanual_action_key_sets():
    from lerobot.async_inference.alohamini_client import AlohaMiniAsyncClientConfig
    from lerobot.async_inference.helpers import TimedAction

    cfg = AlohaMiniAsyncClientConfig(
        policy_type="act",
        pretrained_name_or_path="checkpoint",
        actions_per_chunk=30,
        use_relative_state=True,
        use_relative_actions=True,
        diagnostics=True,
        diagnostics_session_id="slow_motion_001",
    )

    client_cfg = cfg.make_robot_client_config()

    assert len(client_cfg.action_key_sets[7]) == 7
    assert all(key.startswith("arm_right_") for key in client_cfg.action_key_sets[7])
    assert len(client_cfg.action_key_sets[14]) == 14
    assert len(client_cfg.action_key_sets[18]) == 18
    assert client_cfg.action_key_sets[18][:14] == client_cfg.action_key_sets[14]
    assert client_cfg.action_key_sets[18][-4:] == [
        "x.vel",
        "y.vel",
        "theta.vel",
        "lift_axis.height_mm",
    ]
    assert client_cfg.use_relative_state is True
    assert client_cfg.use_relative_actions is True
    assert client_cfg.background_observation_send is True
    assert client_cfg.observation_image_transport == "jpeg"
    assert client_cfg.inference_request_timeout_s == 10.0
    assert client_cfg.diagnostics is True
    assert client_cfg.diagnostics_session_id == "slow_motion_001"
    assert "127.0.0.1:18080" in " ".join(cfg.ssh_command())

    client = _action_validation_client(
        {dimension: tuple(keys) for dimension, keys in client_cfg.action_key_sets.items()}
    )
    action = torch.arange(7, dtype=torch.float32)
    assert client._accept_action_chunk([TimedAction(timestamp=0, timestep=0, action=action)])
    assert client._action_tensor_to_action_dict(action) == {
        key: float(index) for index, key in enumerate(client_cfg.action_key_sets[7])
    }


def test_regular_robot_client_keeps_synchronous_raw_defaults(robot_client):
    assert robot_client.config.background_observation_send is False
    assert robot_client.config.observation_image_transport == "raw"


def test_background_jpeg_observation_uses_raw_fallback(robot_client, monkeypatch):
    camera_a = np.zeros((8, 8, 3), dtype=np.uint8)
    camera_b = np.ones((8, 8, 3), dtype=np.uint8)
    robot_client.config.background_observation_send = True
    robot_client.config.observation_image_transport = "jpeg"
    monkeypatch.setattr(
        robot_client.robot,
        "get_observation",
        lambda: {
            "motor_1.pos": 1.0,
            "motor_2.pos": 2.0,
            "motor_3.pos": 3.0,
            "camera_a": camera_a,
            "camera_b": camera_b,
        },
    )
    monkeypatch.setattr(
        robot_client.robot,
        "get_latest_jpeg_images",
        lambda: {"camera_a": b"host-jpeg"},
        raising=False,
    )

    observation = robot_client._capture_background_observation("pick")

    assert observation.must_go is True
    assert observation.jpeg_images == {"camera_a": b"host-jpeg"}
    assert "camera_a" not in observation.observation
    assert observation.observation["camera_b"] is camera_b
    assert observation.observation["task"] == "pick"


def test_background_relative_state_samples_previous_then_current(robot_client, monkeypatch):
    observations = iter(
        [
            {"motor_1.pos": 1.0, "motor_2.pos": 2.0, "motor_3.pos": 3.0},
            {"motor_1.pos": 4.0, "motor_2.pos": 5.0, "motor_3.pos": 6.0},
        ]
    )
    robot_client.config.background_observation_send = True
    robot_client.config.use_relative_state = True
    monkeypatch.setattr(robot_client.robot, "get_observation", lambda: next(observations))
    monkeypatch.setattr(time, "sleep", lambda _: None)

    observation = robot_client._capture_background_observation("task")

    assert observation.observation["motor_1.pos"] == 4.0
    torch.testing.assert_close(observation.previous_state, torch.tensor([[1.0, 2.0, 3.0]]))


def test_refill_triggers_are_merged_while_request_is_pending(robot_client):
    robot_client.config.background_observation_send = True
    robot_client._pending_request_id = 7

    robot_client._request_refill("first")
    robot_client._request_refill("second")

    assert not robot_client.refill_event.is_set()
    assert robot_client._merged_refill_triggers == 2


def test_background_rpc_failure_safely_stops(robot_client, monkeypatch):
    from lerobot.async_inference.helpers import TimedObservation

    robot_client.config.background_observation_send = True
    robot_client.refill_event.set()
    monkeypatch.setattr(
        robot_client,
        "_capture_background_observation",
        lambda _task: TimedObservation(timestamp=time.time(), timestep=0, observation={}),
    )
    monkeypatch.setattr(robot_client, "send_observation", lambda _obs, timeout=None: False)

    robot_client.send_observations_in_background("task")

    assert robot_client.shutdown_event.is_set()
    assert robot_client.action_queue.empty()


def test_blocked_observation_rpc_does_not_block_action_dispatch(robot_client, monkeypatch):
    from lerobot.async_inference.helpers import TimedAction, TimedObservation

    rpc_started = threading.Event()
    release_rpc = threading.Event()
    robot_client.config.background_observation_send = True
    robot_client.config.inference_request_timeout_s = 1.0
    robot_client.refill_event.set()
    robot_client._active_action_dim = 6
    robot_client.action_queue.put(
        TimedAction(timestamp=time.time(), timestep=1, action=torch.ones(6))
    )
    monkeypatch.setattr(
        robot_client,
        "_capture_background_observation",
        lambda _task: TimedObservation(timestamp=time.time(), timestep=0, observation={}),
    )

    def blocked_send(_observation, timeout=None):
        rpc_started.set()
        release_rpc.wait(1)
        return True

    monkeypatch.setattr(robot_client, "send_observation", blocked_send)
    sender = threading.Thread(target=robot_client.send_observations_in_background, args=("task",))
    sender.start()
    assert rpc_started.wait(1)

    performed = robot_client.control_loop_action()

    assert performed is not None
    release_rpc.set()
    robot_client.shutdown_event.set()
    robot_client._pending_request_id = None
    robot_client._pending_chunk_received.set()
    sender.join(1)
    assert not sender.is_alive()


def test_background_receiver_only_accepts_matching_request_id(robot_client):
    from lerobot.async_inference.helpers import TimedAction
    from lerobot.transport import services_pb2

    wrong = [
        TimedAction(
            timestamp=time.time(),
            timestep=1,
            action=torch.ones(6),
            source_request_ids=(1,),
        )
    ]
    matching = [
        TimedAction(
            timestamp=time.time(),
            timestep=2,
            action=torch.full((6,), 2.0),
            source_request_ids=(2,),
        )
    ]

    class Stub:
        def __init__(self):
            self.responses = iter((wrong, matching))

        def GetActions(self, _request):  # noqa: N802
            try:
                actions = next(self.responses)
            except StopIteration:
                robot_client.shutdown_event.set()
                return services_pb2.Actions()
            return services_pb2.Actions(data=pickle.dumps(actions))

    robot_client.config.background_observation_send = True
    robot_client.stub = Stub()
    robot_client._pending_request_id = 2
    robot_client._pending_request_started_at = time.monotonic()
    receiver = threading.Thread(target=robot_client.receive_actions)
    receiver.start()
    robot_client.start_barrier.wait()
    receiver.join(1)

    assert not receiver.is_alive()
    assert robot_client._pending_request_id is None
    assert [action.source_request_ids for action in robot_client.action_queue.queue] == [(2,)]


def test_background_request_timeout_sends_once_and_safely_stops(robot_client, monkeypatch):
    from lerobot.async_inference.helpers import TimedObservation

    send_count = 0
    robot_client.config.background_observation_send = True
    robot_client.config.inference_request_timeout_s = 0.01
    robot_client.refill_event.set()
    monkeypatch.setattr(
        robot_client,
        "_capture_background_observation",
        lambda _task: TimedObservation(timestamp=time.time(), timestep=0, observation={}),
    )

    def send(_observation, timeout=None):
        nonlocal send_count
        send_count += 1
        robot_client._request_refill("duplicate-threshold")
        return True

    monkeypatch.setattr(robot_client, "send_observation", send)

    robot_client.send_observations_in_background("task")

    assert send_count == 1
    assert robot_client.shutdown_event.is_set()
    assert robot_client.action_queue.empty()


def test_background_stop_joins_workers_before_robot_disconnect(robot_client, monkeypatch):
    order = []
    original_disconnect = robot_client.robot.disconnect
    robot_client.config.background_observation_send = True

    class Channel:
        def close(self):
            order.append("channel_closed")

    robot_client.channel = Channel()

    def worker():
        robot_client.shutdown_event.wait()
        order.append("worker_exited")

    thread = threading.Thread(target=worker)
    robot_client._worker_threads = [thread]
    thread.start()

    def disconnect():
        order.append("robot_disconnected")
        original_disconnect()

    monkeypatch.setattr(robot_client.robot, "disconnect", disconnect)

    robot_client.stop()

    assert order == ["channel_closed", "worker_exited", "robot_disconnected"]
