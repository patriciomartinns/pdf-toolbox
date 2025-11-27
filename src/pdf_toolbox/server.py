from __future__ import annotations

from fastmcp import FastMCP

from .tools.pdf_reader import register_pdf_tools

mcp = FastMCP(
    "PDF Toolbox",
    version="1.1.0",
    instructions=(
        "Local PDF ingestion helpers for LLM agents. Provide the PDF path (relative to the allowed "
        "base directory when set) and tune pagination or semantic search parameters as needed."
    ),
)

register_pdf_tools(mcp)

app = mcp

__all__ = ["mcp", "app"]

