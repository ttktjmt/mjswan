"""Tests for mjswan._cli — unified CLI entry point.

L1 — pure Python, no MuJoCo/ONNX required (safe for pre-commit).
"""

import argparse
import shutil as _shutil_module
import subprocess as _subprocess_module
import sys
from unittest.mock import MagicMock

import pytest

import mjswan._cli as cli_module
from mjswan._cli import (
    _cmd_cf_pages,
    _cmd_gh_pages,
    _cmd_info,
    _cmd_serve,
    _resolve_app_dir,
    mjswan_cli,
)


def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


# ---------------------------------------------------------------------------
# _resolve_app_dir
# ---------------------------------------------------------------------------


class TestResolveAppDir:
    def test_explicit_path_returns_resolved(self, tmp_path):
        assert _resolve_app_dir(str(tmp_path)) == tmp_path.resolve()

    def test_none_uses_dist_in_cwd(self, tmp_path, monkeypatch):
        dist = tmp_path / "dist"
        dist.mkdir()
        monkeypatch.chdir(tmp_path)
        assert _resolve_app_dir(None) == dist.resolve()

    def test_none_exits_when_dist_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            _resolve_app_dir(None)

    def test_none_error_mentions_dist(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            _resolve_app_dir(None)
        assert "dist" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _cmd_serve
# ---------------------------------------------------------------------------


class TestCmdServe:
    @pytest.fixture
    def app_dir(self, tmp_path):
        d = tmp_path / "dist"
        d.mkdir()
        return d

    @pytest.fixture
    def mock_app_cls(self, monkeypatch):
        instance = MagicMock()
        cls = MagicMock(return_value=instance)
        monkeypatch.setattr("mjswan.app.mjswanApp", cls)
        return cls, instance

    def test_launches_with_explicit_path(self, app_dir, mock_app_cls):
        cls, instance = mock_app_cls
        _cmd_serve(_ns(path=str(app_dir)))
        cls.assert_called_once_with(app_dir.resolve())
        instance.launch.assert_called_once()

    def test_launches_with_default_dist(self, tmp_path, monkeypatch, mock_app_cls):
        dist = tmp_path / "dist"
        dist.mkdir()
        monkeypatch.chdir(tmp_path)
        cls, instance = mock_app_cls
        _cmd_serve(_ns(path=None))
        cls.assert_called_once_with(dist.resolve())
        instance.launch.assert_called_once()


# ---------------------------------------------------------------------------
# _cmd_cf_pages
# ---------------------------------------------------------------------------


class TestCmdCfPages:
    @pytest.fixture
    def app_dir(self, tmp_path):
        d = tmp_path / "myproject" / "dist"
        d.mkdir(parents=True)
        return d

    @pytest.fixture
    def mock_run(self, monkeypatch):
        result = MagicMock(returncode=0)
        mock = MagicMock(return_value=result)
        monkeypatch.setattr(_subprocess_module, "run", mock)
        return mock

    def test_uses_wrangler_when_found(self, app_dir, monkeypatch, mock_run):
        monkeypatch.setattr(
            _shutil_module,
            "which",
            lambda n: "/usr/bin/wrangler" if n == "wrangler" else None,
        )
        with pytest.raises(SystemExit):
            _cmd_cf_pages(_ns(path=str(app_dir), name="my-app"))
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/wrangler"

    def test_falls_back_to_npx_when_wrangler_missing(
        self, app_dir, monkeypatch, mock_run
    ):
        monkeypatch.setattr(
            _shutil_module, "which", lambda n: "/usr/bin/npx" if n == "npx" else None
        )
        with pytest.raises(SystemExit):
            _cmd_cf_pages(_ns(path=str(app_dir), name="my-app"))
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/npx"
        assert "wrangler" in cmd

    def test_exits_with_error_when_neither_found(self, app_dir, monkeypatch, capsys):
        monkeypatch.setattr(_shutil_module, "which", lambda n: None)
        with pytest.raises(SystemExit) as exc_info:
            _cmd_cf_pages(_ns(path=str(app_dir), name="my-app"))
        assert exc_info.value.code != 0
        assert "wrangler" in capsys.readouterr().err

    def test_deploy_command_includes_project_name(self, app_dir, monkeypatch, mock_run):
        monkeypatch.setattr(
            _shutil_module,
            "which",
            lambda n: "/usr/bin/wrangler" if n == "wrangler" else None,
        )
        with pytest.raises(SystemExit):
            _cmd_cf_pages(_ns(path=str(app_dir), name="custom-name"))
        cmd = mock_run.call_args[0][0]
        assert "--project-name" in cmd
        assert "custom-name" in cmd

    def test_deploy_command_includes_app_dir(self, app_dir, monkeypatch, mock_run):
        monkeypatch.setattr(
            _shutil_module,
            "which",
            lambda n: "/usr/bin/wrangler" if n == "wrangler" else None,
        )
        with pytest.raises(SystemExit):
            _cmd_cf_pages(_ns(path=str(app_dir), name="my-app"))
        cmd = mock_run.call_args[0][0]
        assert str(app_dir.resolve()) in cmd

    def test_derives_name_from_parent_dir_when_omitted(
        self, app_dir, monkeypatch, mock_run
    ):
        monkeypatch.setattr(
            _shutil_module,
            "which",
            lambda n: "/usr/bin/wrangler" if n == "wrangler" else None,
        )
        with pytest.raises(SystemExit):
            _cmd_cf_pages(_ns(path=str(app_dir), name=None))
        cmd = mock_run.call_args[0][0]
        assert "myproject" in cmd

    def test_exits_zero_on_success(self, app_dir, monkeypatch, mock_run):
        monkeypatch.setattr(_shutil_module, "which", lambda n: "/usr/bin/wrangler")
        mock_run.return_value.returncode = 0
        with pytest.raises(SystemExit) as exc_info:
            _cmd_cf_pages(_ns(path=str(app_dir), name="my-app"))
        assert exc_info.value.code == 0

    def test_exits_nonzero_on_failure(self, app_dir, monkeypatch, mock_run):
        monkeypatch.setattr(_shutil_module, "which", lambda n: "/usr/bin/wrangler")
        mock_run.return_value.returncode = 1
        with pytest.raises(SystemExit) as exc_info:
            _cmd_cf_pages(_ns(path=str(app_dir), name="my-app"))
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _cmd_gh_pages
# ---------------------------------------------------------------------------


class TestCmdGhPages:
    @pytest.fixture
    def app_dir(self, tmp_path):
        d = tmp_path / "dist"
        d.mkdir()
        return d

    @pytest.fixture
    def mock_run(self, monkeypatch):
        result = MagicMock(returncode=0)
        mock = MagicMock(return_value=result)
        monkeypatch.setattr(_subprocess_module, "run", mock)
        return mock

    def test_calls_ghp_import_with_push_and_force(self, app_dir, monkeypatch, mock_run):
        monkeypatch.setattr(
            _shutil_module,
            "which",
            lambda n: "/usr/bin/ghp-import" if n == "ghp-import" else None,
        )
        with pytest.raises(SystemExit):
            _cmd_gh_pages(_ns(path=str(app_dir)))
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/ghp-import"
        assert "-p" in cmd
        assert "-f" in cmd

    def test_passes_app_dir_to_ghp_import(self, app_dir, monkeypatch, mock_run):
        monkeypatch.setattr(
            _shutil_module,
            "which",
            lambda n: "/usr/bin/ghp-import" if n == "ghp-import" else None,
        )
        with pytest.raises(SystemExit):
            _cmd_gh_pages(_ns(path=str(app_dir)))
        cmd = mock_run.call_args[0][0]
        assert str(app_dir.resolve()) in cmd

    def test_exits_with_error_when_ghp_import_missing(
        self, app_dir, monkeypatch, capsys
    ):
        monkeypatch.setattr(_shutil_module, "which", lambda n: None)
        with pytest.raises(SystemExit) as exc_info:
            _cmd_gh_pages(_ns(path=str(app_dir)))
        assert exc_info.value.code != 0
        assert "ghp-import" in capsys.readouterr().err

    def test_exits_zero_on_success(self, app_dir, monkeypatch, mock_run):
        monkeypatch.setattr(_shutil_module, "which", lambda n: "/usr/bin/ghp-import")
        mock_run.return_value.returncode = 0
        with pytest.raises(SystemExit) as exc_info:
            _cmd_gh_pages(_ns(path=str(app_dir)))
        assert exc_info.value.code == 0

    def test_exits_nonzero_on_failure(self, app_dir, monkeypatch, mock_run):
        monkeypatch.setattr(_shutil_module, "which", lambda n: "/usr/bin/ghp-import")
        mock_run.return_value.returncode = 1
        with pytest.raises(SystemExit) as exc_info:
            _cmd_gh_pages(_ns(path=str(app_dir)))
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _cmd_info
# ---------------------------------------------------------------------------


class TestCmdInfo:
    @pytest.fixture
    def app_dir(self, tmp_path):
        d = tmp_path / "dist"
        d.mkdir()
        return d

    @pytest.fixture
    def mock_version(self, monkeypatch):
        import importlib.metadata as _meta

        monkeypatch.setattr(_meta, "version", lambda _: "1.2.3")

    def test_prints_app_dir_path(self, app_dir, mock_version, capsys):
        _cmd_info(_ns(path=str(app_dir)))
        assert str(app_dir.resolve()) in capsys.readouterr().out

    def test_prints_file_count_zero_for_empty_dir(self, app_dir, mock_version, capsys):
        _cmd_info(_ns(path=str(app_dir)))
        assert "0" in capsys.readouterr().out

    def test_prints_file_count(self, app_dir, mock_version, capsys):
        (app_dir / "a.js").write_bytes(b"x" * 10)
        (app_dir / "b.css").write_bytes(b"y" * 20)
        _cmd_info(_ns(path=str(app_dir)))
        assert "2" in capsys.readouterr().out

    def test_size_reported_in_bytes(self, app_dir, mock_version, capsys):
        (app_dir / "tiny.txt").write_bytes(b"x" * 500)
        _cmd_info(_ns(path=str(app_dir)))
        assert "B" in capsys.readouterr().out

    def test_size_reported_in_kilobytes(self, app_dir, mock_version, capsys):
        (app_dir / "mid.bin").write_bytes(b"x" * 2048)
        _cmd_info(_ns(path=str(app_dir)))
        assert "KB" in capsys.readouterr().out

    def test_size_reported_in_megabytes(self, app_dir, mock_version, capsys):
        (app_dir / "big.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        _cmd_info(_ns(path=str(app_dir)))
        assert "MB" in capsys.readouterr().out

    def test_prints_mjswan_version(self, app_dir, mock_version, capsys):
        _cmd_info(_ns(path=str(app_dir)))
        assert "1.2.3" in capsys.readouterr().out

    def test_uses_default_dist_path(self, tmp_path, monkeypatch, mock_version, capsys):
        dist = tmp_path / "dist"
        dist.mkdir()
        monkeypatch.chdir(tmp_path)
        _cmd_info(_ns(path=None))
        assert str(dist.resolve()) in capsys.readouterr().out


# ---------------------------------------------------------------------------
# mjswan_cli (argparse dispatch)
# ---------------------------------------------------------------------------


class TestMjswanCli:
    def test_version_flag_exits_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["mjswan", "--version"])
        with pytest.raises(SystemExit) as exc_info:
            mjswan_cli()
        assert exc_info.value.code == 0

    def test_version_flag_prints_version_string(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["mjswan", "--version"])
        with pytest.raises(SystemExit):
            mjswan_cli()
        out = capsys.readouterr().out
        assert "mjswan" in out

    def test_no_subcommand_exits_nonzero(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["mjswan"])
        with pytest.raises(SystemExit) as exc_info:
            mjswan_cli()
        assert exc_info.value.code != 0

    def test_dispatches_serve(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cli_module, "_cmd_serve", lambda args: calls.append(args))
        monkeypatch.setattr(sys, "argv", ["mjswan", "serve"])
        mjswan_cli()
        assert len(calls) == 1

    def test_serve_passes_path_argument(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cli_module, "_cmd_serve", lambda args: calls.append(args))
        monkeypatch.setattr(sys, "argv", ["mjswan", "serve", "/some/path"])
        mjswan_cli()
        assert calls[0].path == "/some/path"

    def test_dispatches_cf_pages(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            cli_module, "_cmd_cf_pages", lambda args: calls.append(args)
        )
        monkeypatch.setattr(sys, "argv", ["mjswan", "cf-pages"])
        mjswan_cli()
        assert len(calls) == 1

    def test_cf_pages_passes_name_argument(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            cli_module, "_cmd_cf_pages", lambda args: calls.append(args)
        )
        monkeypatch.setattr(sys, "argv", ["mjswan", "cf-pages", "--name", "my-site"])
        mjswan_cli()
        assert calls[0].name == "my-site"

    def test_cf_pages_name_defaults_to_none(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            cli_module, "_cmd_cf_pages", lambda args: calls.append(args)
        )
        monkeypatch.setattr(sys, "argv", ["mjswan", "cf-pages"])
        mjswan_cli()
        assert calls[0].name is None

    def test_dispatches_gh_pages(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            cli_module, "_cmd_gh_pages", lambda args: calls.append(args)
        )
        monkeypatch.setattr(sys, "argv", ["mjswan", "gh-pages"])
        mjswan_cli()
        assert len(calls) == 1

    def test_dispatches_info(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cli_module, "_cmd_info", lambda args: calls.append(args))
        monkeypatch.setattr(sys, "argv", ["mjswan", "info"])
        mjswan_cli()
        assert len(calls) == 1

    def test_info_passes_path_argument(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cli_module, "_cmd_info", lambda args: calls.append(args))
        monkeypatch.setattr(sys, "argv", ["mjswan", "info", "/some/path"])
        mjswan_cli()
        assert calls[0].path == "/some/path"
