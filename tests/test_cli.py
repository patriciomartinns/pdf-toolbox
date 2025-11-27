from __future__ import annotations

import argparse
from typing import Any, Sequence

import pytest

from mcp_pdf_reader import cli


def test_start_mcp_server_runs_with_quiet(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> None:
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr(cli.mcp, "run", fake_run)
    cli.start_mcp_server(["--quiet"])
    assert called["args"] == ()
    assert called["kwargs"] == {}


def test_start_mcp_server_prints_banner(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(cli.mcp, "run", lambda: None)
    cli.start_mcp_server([])
    captured = capsys.readouterr()
    assert "Starting MCP PDF Reader" in captured.out


def test_start_mcp_server_quiet_suppresses_banner(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(cli.mcp, "run", lambda: None)
    cli.start_mcp_server(["--quiet"])
    captured = capsys.readouterr()
    assert captured.out == ""


def test_start_mcp_server_rejects_non_stdio(monkeypatch: Any) -> None:
    class DummyParser:
        def parse_args(self, argv: Sequence[str] | None) -> argparse.Namespace:
            return argparse.Namespace(transport="tcp", quiet=True)

        def error(self, message: str) -> None:
            raise RuntimeError(message)

    monkeypatch.setattr(cli, "_build_parser", lambda: DummyParser())
    monkeypatch.setattr(cli.mcp, "run", lambda: None)
    with pytest.raises(RuntimeError, match="Only the stdio"):
        cli.start_mcp_server([])

