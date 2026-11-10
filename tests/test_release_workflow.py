"""The release's publish jobs build the commit the release job checked, not a ref name."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/release.yml"


def test_publish_jobs_check_out_the_release_commit_by_sha():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]

    assert jobs["release"]["outputs"]["sha"] == "${{ steps.tag.outputs.sha }}"
    for name in ("publish-pypi", "publish-npm"):
        assert jobs[name]["with"]["ref"] == "${{ needs.release.outputs.sha }}"
