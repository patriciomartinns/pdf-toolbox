from __future__ import annotations

import argparse
from typing import Sequence

from .server import mcp


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

