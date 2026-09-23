"""A missing source backend names the extra that supplies it.

Both clients are optional dependencies imported inside a function, so the first thing a
user without them sees is this message. A bare ``ModuleNotFoundError`` would leave them
guessing which extra to install.
"""

from __future__ import annotations

import builtins

import pytest

from mjswan.source.hf import _hub
from mjswan.source.wandb import _wandb


@pytest.fixture
def without(monkeypatch):
    """Make ``import <name>`` fail, however the module is spelled elsewhere."""

    def hide(name: str) -> None:
        real = builtins.__import__

        def fake(module, *args, **kwargs):
            if module == name or module.startswith(f"{name}."):
                raise ImportError(f"No module named {name!r}")
            return real(module, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)

    return hide


class TestMissingBackend:
    def test_wandb_names_its_extra(self, without):
        without("wandb")

        with pytest.raises(ImportError, match=r"mjswan\[wandb\]"):
            _wandb()

    def test_huggingface_hub_names_its_extra(self, without):
        without("huggingface_hub")

        with pytest.raises(ImportError, match=r"mjswan\[hf\]"):
            _hub()

    def test_the_original_failure_is_kept_as_the_cause(self, without):
        without("wandb")

        with pytest.raises(ImportError) as excinfo:
            _wandb()

        assert isinstance(excinfo.value.__cause__, ImportError)


class TestBackendPresent:
    def test_the_helper_returns_the_module(self):
        pytest.importorskip("wandb")

        assert _wandb().__name__ == "wandb"
