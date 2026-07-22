from types import SimpleNamespace

import pytest
import torch

import lerobot.policies.factory as policy_factory
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

RIGHT_ARM_NAMES = [
    "arm_right_shoulder_pan.pos",
    "arm_right_shoulder_lift.pos",
    "arm_right_elbow_flex.pos",
    "arm_right_wrist_flex.pos",
    "arm_right_wrist_roll.pos",
    "arm_right_gripper.pos",
    "arm_right_extra.pos",
]


class DummyACTPolicy(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config


def make_metadata(right_arm_names: list[str] = RIGHT_ARM_NAMES):
    left_arm_names = [name.replace("arm_right_", "arm_left_") for name in RIGHT_ARM_NAMES]
    state_names = [*left_arm_names, *right_arm_names, "x.vel", "y.vel", "theta.vel", "lift_axis.height_mm"]
    return SimpleNamespace(
        features={
            OBS_STATE: {"dtype": "float32", "shape": (18,), "names": state_names},
            ACTION: {"dtype": "float32", "shape": (18,), "names": state_names},
            f"{OBS_IMAGES}.forward": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            f"{OBS_IMAGES}.wrist_left": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            f"{OBS_IMAGES}.wrist_right": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
        },
        stats={},
    )


def test_act_single_arm_defaults_to_disabled():
    assert ACTConfig().single_arm is False


def test_act_single_arm_configures_right_arm_features(monkeypatch):
    monkeypatch.setattr(policy_factory, "get_policy_class", lambda _: DummyACTPolicy)
    config = ACTConfig(single_arm=True, device="cpu")

    policy = policy_factory.make_policy(config, ds_meta=make_metadata())

    assert policy.config.input_features[OBS_STATE].shape == (7,)
    assert policy.config.output_features[ACTION].shape == (7,)
    assert policy.config.action_feature_names == RIGHT_ARM_NAMES
    assert policy.config._single_arm_state_indices == list(range(7, 14))
    assert policy.config._single_arm_action_indices == list(range(7, 14))
    assert set(policy.config.image_features) == {f"{OBS_IMAGES}.forward", f"{OBS_IMAGES}.wrist_right"}


def test_act_single_arm_requires_seven_right_arm_dimensions(monkeypatch):
    monkeypatch.setattr(policy_factory, "get_policy_class", lambda _: DummyACTPolicy)
    config = ACTConfig(single_arm=True, device="cpu")

    with pytest.raises(ValueError, match="exactly 7 right-arm dimensions"):
        policy_factory.make_policy(config, ds_meta=make_metadata(RIGHT_ARM_NAMES[:-1]))
