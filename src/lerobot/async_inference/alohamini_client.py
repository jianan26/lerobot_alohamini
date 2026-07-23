"""Run AlohaMini asynchronous inference through an SSH tunnel."""

import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field

import draccus

from lerobot.robots.alohamini import AlohaMiniClientConfig
from lerobot.robots.alohamini.model_specs import arm_state_keys_for_robot_model

from .configs import RobotClientConfig
from .robot_client import RobotClient, run_robot_client


@dataclass
class AlohaMiniAsyncClientConfig:
    policy_type: str = field(metadata={"help": "Policy type running on the inference server"})
    pretrained_name_or_path: str = field(metadata={"help": "Checkpoint path on the inference server"})
    actions_per_chunk: int = field(metadata={"help": "Number of actions requested per inference call"})
    task: str = field(default="", metadata={"help": "Policy task instruction"})
    policy_device: str = field(default="cuda", metadata={"help": "Inference-server policy device"})
    fps: int = field(default=30, metadata={"help": "Pi control-loop frequency"})
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Action queue refill threshold"})
    aggregate_fn_name: str = field(
        default="weighted_average", metadata={"help": "Overlapping action aggregation"}
    )
    use_relative_state: bool = field(
        default=False, metadata={"help": "Use ACT state[t-1] - state[t] inputs"}
    )
    use_relative_actions: bool = field(
        default=False, metadata={"help": "Convert ACT relative predictions to absolute actions"}
    )

    robot_model: str = field(default="alohamini2pro", metadata={"help": "Must match the Pi Host model"})
    robot_id: str = field(default="alohamini", metadata={"help": "Local robot identifier"})
    robot_host: str = field(default="127.0.0.1", metadata={"help": "Pi Host ZMQ address"})
    robot_cmd_port: int = field(default=5555, metadata={"help": "Pi Host ZMQ command port"})
    robot_observation_port: int = field(default=5556, metadata={"help": "Pi Host ZMQ observation port"})

    ssh_host: str = field(default="183.230.224.121", metadata={"help": "Inference-server SSH host"})
    ssh_port: int = field(default=50210, metadata={"help": "Inference-server SSH port"})
    ssh_user: str = field(default="jianan", metadata={"help": "Inference-server SSH user"})
    server_grpc_port: int = field(default=8080, metadata={"help": "PolicyServer loopback port"})
    local_tunnel_port: int = field(default=18080, metadata={"help": "Pi loopback port for the SSH tunnel"})

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    def make_robot_client_config(self) -> RobotClientConfig:
        left_arm_keys, right_arm_keys = arm_state_keys_for_robot_model(self.robot_model)
        arm_action_keys = [*left_arm_keys, *right_arm_keys]
        full_action_keys = [*arm_action_keys, "x.vel", "y.vel", "theta.vel", "lift_axis.height_mm"]
        robot_config = AlohaMiniClientConfig(
            remote_ip=self.robot_host,
            port_zmq_cmd=self.robot_cmd_port,
            port_zmq_observations=self.robot_observation_port,
            robot_model=self.robot_model,
            id=self.robot_id,
        )
        return RobotClientConfig(
            policy_type=self.policy_type,
            pretrained_name_or_path=self.pretrained_name_or_path,
            robot=robot_config,
            actions_per_chunk=self.actions_per_chunk,
            task=self.task,
            server_address=f"127.0.0.1:{self.local_tunnel_port}",
            policy_device=self.policy_device,
            chunk_size_threshold=self.chunk_size_threshold,
            fps=self.fps,
            aggregate_fn_name=self.aggregate_fn_name,
            use_relative_state=self.use_relative_state,
            use_relative_actions=self.use_relative_actions,
            action_key_sets={
                len(right_arm_keys): list(right_arm_keys),
                len(arm_action_keys): arm_action_keys,
                len(full_action_keys): full_action_keys,
            },
        )

    def ssh_command(self) -> list[str]:
        return [
            "ssh",
            "-N",
            "-p",
            str(self.ssh_port),
            "-L",
            f"127.0.0.1:{self.local_tunnel_port}:127.0.0.1:{self.server_grpc_port}",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            f"{self.ssh_user}@{self.ssh_host}",
        ]


def _stop_tunnel(tunnel: subprocess.Popen[bytes]) -> None:
    if tunnel.poll() is not None:
        return
    tunnel.terminate()
    try:
        tunnel.wait(timeout=5)
    except subprocess.TimeoutExpired:
        tunnel.kill()
        tunnel.wait()


def _wait_for_tunnel(tunnel: subprocess.Popen[bytes], local_port: int, timeout_s: float = 10) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if tunnel.poll() is not None:
            raise RuntimeError(f"SSH tunnel exited with status {tunnel.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)

    raise TimeoutError(f"SSH tunnel did not become ready on 127.0.0.1:{local_port}")


@draccus.wrap()
def main(cfg: AlohaMiniAsyncClientConfig) -> None:
    tunnel = subprocess.Popen(cfg.ssh_command())
    try:
        _wait_for_tunnel(tunnel, cfg.local_tunnel_port)

        def watch_tunnel(client: RobotClient) -> None:
            def wait_for_tunnel() -> None:
                tunnel.wait()
                if client.running:
                    client.logger.error("SSH tunnel exited; stopping the robot client")
                    client.shutdown_event.set()

            threading.Thread(target=wait_for_tunnel, daemon=True).start()

        run_robot_client(cfg.make_robot_client_config(), on_started=watch_tunnel)
    finally:
        _stop_tunnel(tunnel)


if __name__ == "__main__":
    main()
