import pytest
from typer.testing import CliRunner

from mjswan import __version__
from mjswan.cli import app


@pytest.mark.parametrize("args", [["version"], ["--version"]])
def test_version_prints_the_package_version(args: list[str]):
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"mjswan {__version__}"
