from __future__ import annotations

import argparse
from typing import Any, Sequence

import pytest
from typer.testing import CliRunner

from mcp_pdf_reader import cli
from mcp_pdf_reader.schemas import PDFPage, PDFReadResponse

runner = CliRunner()


def _dummy_read_response() -> PDFReadResponse:
    return PDFReadResponse(
        path="dummy.pdf",
        page_count=1,
        pages=[PDFPage(page_number=1, text="content")],
    )



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


def test_read_pdf_cli_invokes_service(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_read_pdf(**kwargs: Any) -> PDFReadResponse:
        captured.update(kwargs)
        return _dummy_read_response()

    monkeypatch.setattr(cli.pdf_service, "read_pdf", fake_read_pdf)
    result = runner.invoke(
        cli.tool_app,
        [
            "read-pdf",
            "manual.pdf",
            "--start-page",
            "2",
            "--end-page",
            "5",
            "--max-pages",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert captured == {"path": "manual.pdf", "start_page": 2, "end_page": 5, "max_pages": 7}
    assert "PDF read" in result.output


def test_search_pdf_cli_handles_error(monkeypatch: Any) -> None:
    def fake_search(**kwargs: Any) -> None:
        raise ValueError("falhou geral")

    monkeypatch.setattr(cli.pdf_service, "search_pdf", fake_search)
    result = runner.invoke(
        cli.tool_app,
        [
            "search-pdf",
            "manual.pdf",
            "chunking",
        ],
    )

    assert result.exit_code == 1
    assert "failed" in result.output


def test_run_tool_cli_returns_exit_code(monkeypatch: Any) -> None:
    def fake_read_pdf(**kwargs: Any) -> PDFReadResponse:
        return _dummy_read_response()

    monkeypatch.setattr(cli.pdf_service, "read_pdf", fake_read_pdf)
    exit_code = cli.run_tool_cli(["read-pdf", "manual.pdf"])
    assert exit_code == 0

