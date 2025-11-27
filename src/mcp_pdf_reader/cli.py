from __future__ import annotations

import argparse
from typing import Any, Callable, Sequence

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

from .server import mcp
from .services import pdf_reader as pdf_service

console = Console()
tool_app = typer.Typer(
    name="pdf-reader",
    add_completion=False,
    help="Run the MCP PDF tools locally (reading, search, chunk description, defaults).",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-pdf-reader",
        description="Run the MCP PDF Reader server over STDIO.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Transport to use. Only stdio is supported for now.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output before the server starts.",
    )
    return parser


def _print_payload(payload: Any) -> None:
    """Pretty-print tool responses in JSON when possible."""

    if payload is None:
        return
    if isinstance(payload, BaseModel):
        console.print_json(data=payload.model_dump())
        return
    if isinstance(payload, (dict, list)):
        console.print_json(data=payload)
        return
    console.print(payload)


def _run_command(action: str, call: Callable[[], Any]) -> None:
    try:
        payload = call()
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        console.print(Panel(str(exc), title=f"{action} failed", border_style="red"))
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - defensive safety net
        console.print(
            Panel(
                f"Unexpected error: {exc}",
                title=f"{action} failed",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc

    console.print(Panel(f"{action} finished successfully.", border_style="green"))
    _print_payload(payload)


@tool_app.command("read-pdf")
def read_pdf_command(
    path: str = typer.Argument(..., help="Path to the PDF file (relative or absolute)."),
    start_page: int = typer.Option(1, "--start-page", "-s", min=1, help="Start page (1-based)."),
    end_page: int | None = typer.Option(
        None,
        "--end-page",
        "-e",
        help="Inclusive end page. Defaults to the document end or configured limit.",
    ),
    max_pages: int | None = typer.Option(
        None,
        "--max-pages",
        "-m",
        min=1,
        help="Maximum number of pages for this call (overrides the default).",
    ),
) -> None:
    """Extract ordered pages from a PDF."""

    _run_command(
        "PDF read",
        lambda: pdf_service.read_pdf(
            path=path,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
        ),
    )


@tool_app.command("search-pdf")
def search_pdf_command(
    path: str = typer.Argument(..., help="Path to the indexed PDF."),
    query: str = typer.Argument(..., help="Search text for semantic similarity."),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, help="Max number of results."),
    min_score: float = typer.Option(
        0.25,
        "--min-score",
        "-t",
        help="Minimum cosine similarity to keep a hit.",
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        help="Chunk size override for this run only.",
    ),
    chunk_overlap: int | None = typer.Option(
        None,
        "--chunk-overlap",
        help="Chunk overlap override for this run only.",
    ),
) -> None:
    """Run semantic search over the PDF."""

    _run_command(
        "Semantic search",
        lambda: pdf_service.search_pdf(
            path=path,
            query=query,
            top_k=top_k,
            min_score=min_score,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    )


@tool_app.command("describe-pdf-sections")
def describe_pdf_sections_command(
    path: str = typer.Argument(..., help="Path to the PDF."),
    max_chunks: int = typer.Option(
        20,
        "--max-chunks",
        "-n",
        min=1,
        help="Maximum number of chunks to describe.",
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        help="Chunk size override for this call.",
    ),
    chunk_overlap: int | None = typer.Option(
        None,
        "--chunk-overlap",
        help="Chunk overlap override for this call.",
    ),
) -> None:
    """Produce deterministic chunks with offsets and metadata."""

    _run_command(
        "Chunk description",
        lambda: pdf_service.describe_pdf_sections(
            path=path,
            max_chunks=max_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    )


@tool_app.command("configure-pdf-defaults")
def configure_pdf_defaults_command(
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        help="New default chunk size (minimum 100).",
    ),
    chunk_overlap: int | None = typer.Option(
        None,
        "--chunk-overlap",
        help="New default overlap (0 up to half the chunk).",
    ),
    max_pages: int | None = typer.Option(
        None,
        "--max-pages",
        help="New default max pages per read call.",
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        help="SentenceTransformer model name to use by default.",
    ),
) -> None:
    """Update runtime defaults for the MCP tools."""

    _run_command(
        "Default configuration",
        lambda: pdf_service.configure_pdf_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_pages=max_pages,
            embedding_model=embedding_model,
        ),
    )


def run_tool_cli(argv: Sequence[str] | None = None) -> int:
    """Entrada principal para o comando `pdf-reader`."""

    args = list(argv) if argv is not None else None
    try:
        tool_app(prog_name="pdf-reader", args=args, standalone_mode=False)
    except typer.Exit as exc:
        return exc.exit_code
    return 0


def start_mcp_server(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.transport != "stdio":
        parser.error("Only the stdio transport is supported at the moment.")

    if not args.quiet:
        print(
            "Starting MCP PDF Reader (transport=stdio). "
            "Tip: `uvx --from <repo> mcp-pdf-reader` for remote execution.",
            flush=True,
        )

    mcp.run()

