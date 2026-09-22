"""The `mjswan demo` surface: what it lists, what it runs, and what it refuses.

The last test here is the point of the file. `_DEMOS` maps a name to a module path as a
string, so a renamed or deleted example leaves a command that fails only when someone
types it, which is how `mjswan demo mjlab` outlived `examples/mjlab/defaults/main.py`.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mjswan.cli import _DEMOS, app

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Modules the command would have run. `_run_module` otherwise spawns and exits."""
    calls: list[str] = []
    monkeypatch.setattr("mjswan.cli._run_module", calls.append)
    return calls


class TestDemoCommand:
    def test_bare_command_lists_and_runs_nothing(
        self, runner: CliRunner, ran: list[str]
    ):
        """Every demo downloads, so the argumentless command must not pick one."""
        result = runner.invoke(app, ["demo"])

        assert result.exit_code == 0
        assert ran == []
        for name in _DEMOS:
            assert name in result.stdout

    @pytest.mark.parametrize("name", sorted(_DEMOS))
    def test_a_name_runs_its_module(self, runner: CliRunner, ran: list[str], name: str):
        result = runner.invoke(app, ["demo", name])

        assert result.exit_code == 0
        assert ran == [_DEMOS[name][0]]

    def test_unknown_name_fails_and_says_where_to_look(
        self, runner: CliRunner, ran: list[str]
    ):
        # `mjlab` specifically: it was a demo until its example moved out of this repo.
        result = runner.invoke(app, ["demo", "mjlab"])

        assert result.exit_code == 1
        assert ran == []
        assert "Unknown demo" in result.stdout

    @pytest.mark.parametrize("name", sorted(_DEMOS))
    def test_the_module_it_names_is_in_the_tree(self, name: str):
        module, _ = _DEMOS[name]
        path = REPO_ROOT / f"{module.replace('.', '/')}.py"

        assert path.is_file(), f"'{name}' names {module}, which is not in the tree"

    @pytest.mark.parametrize("name", sorted(_DEMOS))
    def test_the_module_it_names_is_runnable_as_one(self, name: str):
        """`_run_module` invokes `python -m`, which needs an entry point to reach."""
        module, _ = _DEMOS[name]
        source = (REPO_ROOT / f"{module.replace('.', '/')}.py").read_text()

        assert 'if __name__ == "__main__":' in source
