"""``align_obs_normalizer`` follows the checkpoint, not the task's current rl config."""

from __future__ import annotations

import pytest

from mjswan.mjlab.runner import align_obs_normalizer

torch = pytest.importorskip("torch")
EmpiricalNormalization = pytest.importorskip("rsl_rl.modules").EmpiricalNormalization


class _Policy(torch.nn.Module):
    def __init__(self, normalized: bool) -> None:
        super().__init__()
        self.obs_dim = 4
        self.obs_normalization = normalized
        self.obs_normalizer = (
            EmpiricalNormalization(self.obs_dim) if normalized else torch.nn.Identity()
        )
        self.mlp = torch.nn.Linear(self.obs_dim, 2)


class _Runner:
    def __init__(self, policy: _Policy) -> None:
        self.alg = self
        self._policy = policy

    def get_policy(self) -> _Policy:
        return self._policy


NORMALIZED_CKPT = {
    "actor_state_dict": {"obs_normalizer._mean": None, "mlp.0.weight": None}
}
PLAIN_CKPT = {"actor_state_dict": {"mlp.0.weight": None}}
LEGACY_NORMALIZED_CKPT = {
    "model_state_dict": {"actor_obs_normalizer._mean": None, "actor.0.weight": None}
}
LEGACY_PLAIN_CKPT = {
    # Only the critic normalizes — the actor's normalizer must stay off.
    "model_state_dict": {"critic_obs_normalizer._mean": None, "actor.0.weight": None}
}


@pytest.mark.parametrize(
    ("checkpoint", "normalized"),
    [
        (NORMALIZED_CKPT, True),
        (PLAIN_CKPT, False),
        (LEGACY_NORMALIZED_CKPT, True),
        (LEGACY_PLAIN_CKPT, False),
    ],
)
@pytest.mark.parametrize("built_normalized", [True, False])
def test_normalizer_follows_checkpoint(checkpoint, normalized, built_normalized):
    policy = _Policy(built_normalized)
    align_obs_normalizer(_Runner(policy), checkpoint)

    assert policy.obs_normalization is normalized
    expected = EmpiricalNormalization if normalized else torch.nn.Identity
    assert isinstance(policy.obs_normalizer, expected)
    # What the strict load actually compares.
    has_norm_keys = any(k.startswith("obs_normalizer.") for k in policy.state_dict())
    assert has_norm_keys is normalized


def test_untrained_normalizer_is_not_the_identity():
    """Why the swap is needed rather than a lenient load: eps still rescales."""
    obs = torch.ones(1, 4)
    assert not torch.allclose(EmpiricalNormalization(4)(obs), obs)
