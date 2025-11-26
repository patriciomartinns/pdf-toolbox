from __future__ import annotations

from mcp_pdf_reader.cli import start_mcp_server
from mcp_pdf_reader.server import app, mcp


def main() -> None:
    start_mcp_server()


if __name__ == "__main__":
    main()


__all__ = ["app", "mcp", "main"]

