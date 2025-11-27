from __future__ import annotations

from pdf_toolbox.cli import start_mcp_server
from pdf_toolbox.server import app, mcp


def main() -> None:
    start_mcp_server()


if __name__ == "__main__":
    main()


__all__ = ["app", "mcp", "main"]

