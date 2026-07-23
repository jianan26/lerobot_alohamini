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

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )

    diagnostics: bool = field(default=False, metadata={"help": "Enable diagnostic recording"})
    diagnostics_dir: Path = field(
        default=Path("logs/async_diagnostics"),
        metadata={"help": "Diagnostic path relative to repository root"},
    )
    diagnostics_session_id: str | None = field(
        default=None, metadata={"help": "Shared diagnostic session identifier"}
    )
    diagnostics_sample_interval: int = field(
        default=10, metadata={"help": "Save one full observation for every N received requests"}
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

        if self.diagnostics_sample_interval <= 0:
            raise ValueError(
                f"diagnostics_sample_interval must be positive, got {self.diagnostics_sample_interval}"
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
            "diagnostics": self.diagnostics,
            "diagnostics_dir": str(self.diagnostics_dir),
            "diagnostics_session_id": self.diagnostics_session_id,
            "diagnostics_sample_interval": self.diagnostics_sample_interval,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    use_relative_state: bool = field(
        default=False, metadata={"help": "Send consecutive states for ACT relative-state inference"}
    )
    use_relative_actions: bool = field(
        default=False, metadata={"help": "Convert ACT relative action predictions back to absolute actions"}
    )

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )
    diagnostics: bool = field(default=False, metadata={"help": "Enable diagnostic recording"})
    diagnostics_dir: Path = field(
        default=Path("logs/async_diagnostics"),
        metadata={"help": "Diagnostic path relative to repository root"},
    )
    diagnostics_session_id: str | None = field(
        default=None, metadata={"help": "Shared diagnostic session identifier"}
    )

    # Optional action schemas keyed by action-vector dimension. This is useful for
    # robots whose policy controls only a subset of their available actuators.
    action_key_sets: dict[int, list[str]] | None = field(
        default=None,
        metadata={"help": "Optional action dimension to ordered robot-action-key mapping"},
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.action_key_sets is not None:
            for action_dim, keys in self.action_key_sets.items():
                if action_dim <= 0:
                    raise ValueError(f"action key-set dimension must be positive, got {action_dim}")
                if len(keys) != action_dim:
                    raise ValueError(f"action key-set for dimension {action_dim} has {len(keys)} keys")
                if len(set(keys)) != len(keys):
                    raise ValueError(f"action key-set for dimension {action_dim} contains duplicate keys")

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "client_device": self.client_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "use_relative_state": self.use_relative_state,
            "use_relative_actions": self.use_relative_actions,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "diagnostics": self.diagnostics,
            "diagnostics_dir": str(self.diagnostics_dir),
            "diagnostics_session_id": self.diagnostics_session_id,
            "aggregate_fn_name": self.aggregate_fn_name,
            "action_key_sets": self.action_key_sets,
        }
