# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def alohamini_cameras_config() -> dict[str, CameraConfig]:
    return {
        "forward": OpenCVCameraConfig(
            index_or_path="/dev/video6", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
        # "backward": OpenCVCameraConfig(
        #     index_or_path="/dev/am_camera_backward", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
        # "chest": OpenCVCameraConfig(
        #     index_or_path="/dev/am_camera_chest", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
        "wrist_left": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
        "wrist_right": OpenCVCameraConfig(
            index_or_path="/dev/video2", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
    }


@RobotConfig.register_subclass("alohamini")
@dataclass
class AlohaMiniConfig(RobotConfig):
    left_port: str = "/dev/ttyACM0"  # port to connect to the bus
    right_port: str = "/dev/ttyACM1"  # port to connect to the bus
    disable_torque_on_disconnect: bool = True
    # robot_model drives the whole-robot hardware specs: follower arm profile, base motors,
    # lift motor, and lead screw pitch.
    # alohamini1   – so-arm-5dof,          base sts3215, lift sts3215, lead=84 mm/rev
    # alohamini2   – am-follower-6dof,     base sts3215, lift sts3095, lead=131 mm/rev
    # alohamini2pro– am-follower-6dof-hd,  base sts3250, lift sts3095, lead=131 mm/rev
    robot_model: str = "alohamini2pro"

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=alohamini_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # When True, skip follower arms entirely (only base and lift operate).
    # Use together with --no_leader on the teleoperate side for base-only teleoperation.
    no_follower: bool = False




@dataclass
class AlohaMiniHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    observation_request_window: int = 3

    # Duration of the application
    connection_time_s: int = 6000

    # Watchdog: stop the robot if no command is received for over 1 second.
    watchdog_timeout_ms: int = 1000

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30




@RobotConfig.register_subclass("alohamini_client")
@dataclass
class AlohaMiniClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    observation_request_window: int = 3

    # Must match the robot_model used on the host side so that _state_ft keys are consistent.
    # alohamini1   – so-arm-5dof (6 joints per arm, no wrist_yaw)
    # alohamini2   – am-follower-6dof (7 joints per arm, includes wrist_yaw)
    # alohamini2pro– am-follower-6dof-hd (7 joints per arm, includes wrist_yaw)
    robot_model: str = "alohamini1"

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "z",
            "right": "x",
            "rotate_left": "a",
            "rotate_right": "d",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # Z axis
            "lift_up": "u",
            "lift_down": "j",
            # quit teleop
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=alohamini_cameras_config)

    # Must exceed one host cycle at 30 Hz for request/reply observation transport.
    polling_timeout_ms: int = 200
    connect_timeout_s: int = 5
