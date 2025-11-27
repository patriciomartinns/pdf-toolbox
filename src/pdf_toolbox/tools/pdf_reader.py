from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..schemas import PDFChunkResponse, PDFConfigResponse, PDFReadResponse, PDFSearchResponse
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
        max_pages: int | None = None,
    ) -> PDFReadResponse:
        """Return ordered pages for quick inspection."""

        return pdf_service.read_pdf(
            path=path,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
        )
    _ = read_pdf

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
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> PDFSearchResponse:
        """Rank PDF segments using cosine similarity."""

        return pdf_service.search_pdf(
            path=path,
            query=query,
            top_k=top_k,
            min_score=min_score,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    _ = search_pdf

    @mcp.tool(
        description=(
            "Generates sequential chunks enriched with offsets and metadata for LLM workflows."
        )
    )
    def describe_pdf_sections(
        path: str,
        max_chunks: int = 20,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> PDFChunkResponse:
        """Expose deterministic chunks for structured ingestion."""

        return pdf_service.describe_pdf_sections(
            path=path,
            max_chunks=max_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    _ = describe_pdf_sections

    @mcp.tool(
        description=(
            "Updates the default PDF processing parameters so subsequent calls inherit them."
        )
    )
    def configure_pdf_defaults(
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_pages: int | None = None,
        embedding_model: str | None = None,
    ) -> PDFConfigResponse:
        """Adjust runtime defaults for pagination, chunking, and embedding model."""

        return pdf_service.configure_pdf_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_pages=max_pages,
            embedding_model=embedding_model,
        )
    _ = configure_pdf_defaults


__all__ = ["register_pdf_tools"]

