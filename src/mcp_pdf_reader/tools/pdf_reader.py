from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..schemas import PDFChunkResponse, PDFReadResponse, PDFSearchResponse
from ..services import pdf_reader as pdf_service


def register_pdf_tools(mcp: FastMCP[Any]) -> None:
    @mcp.tool(
        description=(
            "Extracts plain text from a PDF page range while keeping the payload under control."
        )
    )
    def read_pdf(
        path: str,
        start_page: int = 1,
        end_page: int | None = None,
        max_pages: int = pdf_service.DEFAULT_MAX_PAGES,
    ) -> PDFReadResponse:
        """Return ordered pages for quick inspection."""

        return pdf_service.read_pdf(
            path=path,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
        )

    @mcp.tool(
        description=(
            "Runs semantic search over embedded PDF chunks and returns the highest scoring matches."
        )
    )
    def search_pdf(
        path: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.25,
    ) -> PDFSearchResponse:
        """Rank PDF segments using cosine similarity."""

        return pdf_service.search_pdf(
            path=path,
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

    @mcp.tool(
        description=(
            "Generates sequential chunks enriched with offsets and metadata for LLM workflows."
        )
    )
    def describe_pdf_sections(
        path: str,
        max_chunks: int = 20,
        chunk_size: int | None = None,
    ) -> PDFChunkResponse:
        """Expose deterministic chunks for structured ingestion."""

        return pdf_service.describe_pdf_sections(
            path=path,
            max_chunks=max_chunks,
            chunk_size=chunk_size,
        )


__all__ = ["register_pdf_tools"]

