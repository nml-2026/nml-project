from __future__ import annotations

from nba_forecasting.config import load_config


def test_mart_aug_config_overrides() -> None:
    both, _ = load_config("mart")
    none, _ = load_config("mart_aug_none")
    reflect, _ = load_config("mart_aug_reflect")
    perm, _ = load_config("mart_aug_perm")

    assert both.features.reflection.x_prob == 0.5
    assert both.features.reflection.y_prob == 0.5
    assert both.features.augmentation.agent_permutation_prob == 0.5

    assert none.features.reflection.x_prob == 0.0
    assert none.features.reflection.y_prob == 0.0
    assert none.features.augmentation.agent_permutation_prob == 0.0

    assert reflect.features.reflection.x_prob == 0.5
    assert reflect.features.reflection.y_prob == 0.5
    assert reflect.features.augmentation.agent_permutation_prob == 0.0

    assert perm.features.reflection.x_prob == 0.0
    assert perm.features.reflection.y_prob == 0.0
    assert perm.features.augmentation.agent_permutation_prob == 0.5
