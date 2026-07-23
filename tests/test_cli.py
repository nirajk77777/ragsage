"""Tests for the ragsage CLI — the library's standalone driver."""

import pytest
from ragsage.cli import main


def test_info_reports_engine_version(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["info"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "ragsage 0.1.0" in out
    assert "local" in out


def test_no_command_prints_help_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "usage: ragsage" in out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "ragsage 0.1.0" in capsys.readouterr().out
