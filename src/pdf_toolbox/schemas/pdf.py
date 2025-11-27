from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class PDFPage(BaseModel):
    page_number: int = Field(..., ge=1, description="1-based page number.")
    text: str = Field(..., description="Plain text extracted from the page.")

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class PDFReadResponse(BaseModel):
    path: str
    page_count: int
    pages: list[PDFPage]

    @model_validator(mode="after")
    def _ensure_pages(self) -> "PDFReadResponse":
        if not self.pages:
            raise ValueError("No pages were found for the requested range.")
        return self


class PDFSearchHit(BaseModel):
    chunk_id: str
    page_number: int
    score: float = Field(..., ge=-1.0, le=1.0)
    text: str


class PDFSearchResponse(BaseModel):
    path: str
    query: str
    results: list[PDFSearchHit]


class PDFChunkInfo(BaseModel):

    chunk_id: str
    page_number: int
    start_char: int
    end_char: int
    text: str
    embedding_model: str = Field(
        ...,
        description="Embedding model used when generating the chunk index.",
    )


class PDFChunkResponse(BaseModel):
    path: str
    chunk_count: int
    chunks: list[PDFChunkInfo]


class PDFConfigResponse(BaseModel):
    chunk_size: int = Field(..., ge=100)
    chunk_overlap: int = Field(..., ge=0)
    max_pages: int = Field(..., ge=1)
    embedding_model: str


__all__ = [
    "PDFChunkInfo",
    "PDFChunkResponse",
    "PDFConfigResponse",
    "PDFPage",
    "PDFReadResponse",
    "PDFSearchHit",
    "PDFSearchResponse",
]

