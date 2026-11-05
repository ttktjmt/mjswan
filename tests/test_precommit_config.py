"""The eslint pre-commit hook runs the template's locked eslint, not a registry install."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CONFIG = Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml"


def test_eslint_hook_runs_the_template_eslint_by_path():
    repos = yaml.safe_load(CONFIG.read_text())["repos"]
    (eslint,) = [h for repo in repos for h in repo["hooks"] if h["id"] == "eslint"]

    assert eslint["language"] == "system"
    assert eslint["entry"].split()[0] == "src/mjswan/template/node_modules/.bin/eslint"
