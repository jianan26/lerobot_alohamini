import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import ACTDataAugmentationProcessorStep, make_act_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def make_config(aug: list[str]) -> ACTConfig:
    return ACTConfig(
        aug=aug,
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            f"{OBS_IMAGES}.forward": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 40, 40)),
            f"{OBS_IMAGES}.wrist_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 40, 40)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )


def test_act_augmentation_config():
    assert ACTConfig().aug == []
    assert ACTConfig().normalization_mapping["VISUAL"] is NormalizationMode.MEAN_STD
    assert make_config(["image"]).normalization_mapping["VISUAL"] is NormalizationMode.MEAN_STD


def test_act_augmentation_config_rejects_unknown_value():
    try:
        ACTConfig(aug=["image", "audio"])
    except ValueError as exc:
        assert "audio" in str(exc)
    else:
        raise AssertionError("Expected unsupported augmentation to be rejected")


def test_image_augmentation_distinguishes_front_and_wrist_cameras():
    image = torch.linspace(0, 1, 3 * 40 * 40).reshape(1, 3, 40, 40)

    torch.manual_seed(0)
    front = ACTDataAugmentationProcessorStep._augment_image(image, "observation.images.forward")
    torch.manual_seed(0)
    wrist = ACTDataAugmentationProcessorStep._augment_image(image, "observation.images.wrist_left")

    assert front.shape == wrist.shape == image.shape
    assert front.min() >= 0 and front.max() <= 1
    assert wrist.min() >= 0 and wrist.max() <= 1
    assert not torch.allclose(front, wrist)


def test_state_augmentation_scales_noise_by_dataset_std():
    config = make_config(["state"])
    stats = {
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.tensor([1.0, 2.0, 3.0])},
        ACTION: {"mean": torch.zeros(2), "std": torch.ones(2)},
    }
    preprocessor, _ = make_act_pre_post_processors(config, stats)
    augmentation_step = next(
        step for step in preprocessor.steps if isinstance(step, ACTDataAugmentationProcessorStep)
    )
    state = torch.zeros(2, 3)

    preprocessor.train()
    torch.manual_seed(0)
    augmented = augmentation_step.observation({OBS_STATE: state})[OBS_STATE]
    torch.manual_seed(0)
    expected = torch.randn_like(state) * torch.tensor([0.01, 0.02, 0.03])

    torch.testing.assert_close(augmented, expected)


def test_act_augmentation_only_applies_during_training_without_mutating_input():
    config = make_config(["image", "state"])
    stats = {
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.tensor([1.0, 2.0, 3.0])},
        ACTION: {"mean": torch.zeros(2), "std": torch.ones(2)},
        f"{OBS_IMAGES}.forward": {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        f"{OBS_IMAGES}.wrist_left": {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
    }
    preprocessor, _ = make_act_pre_post_processors(config, stats)
    batch = {
        OBS_STATE: torch.zeros(2, 3),
        f"{OBS_IMAGES}.forward": torch.linspace(0, 1, 2 * 3 * 40 * 40).reshape(2, 3, 40, 40),
        f"{OBS_IMAGES}.wrist_left": torch.linspace(0, 1, 2 * 3 * 40 * 40).reshape(2, 3, 40, 40),
    }
    original = {key: value.clone() for key, value in batch.items()}

    preprocessor.train()
    torch.manual_seed(0)
    augmented = preprocessor(batch)

    for key, value in batch.items():
        torch.testing.assert_close(value, original[key])
    assert not torch.allclose(augmented[OBS_STATE], original[OBS_STATE])
    assert not torch.allclose(augmented[f"{OBS_IMAGES}.forward"], original[f"{OBS_IMAGES}.forward"])
    assert not torch.allclose(augmented[f"{OBS_IMAGES}.wrist_left"], original[f"{OBS_IMAGES}.wrist_left"])

    preprocessor.eval()
    unaugmented = preprocessor(batch)
    for key, value in batch.items():
        torch.testing.assert_close(unaugmented[key], value)
