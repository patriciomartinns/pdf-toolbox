from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

import fitz  # type: ignore[import]
import numpy as np

from ..schemas import (
    PDFChunkInfo,
    PDFChunkResponse,
    PDFPage,
    PDFReadResponse,
    PDFSearchHit,
    PDFSearchResponse,
)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 750
DEFAULT_CHUNK_OVERLAP = 75
DEFAULT_MAX_PAGES = 25
_DEFAULT_BASE_PATH = Path.cwd().resolve()
_base_path: Path | None = _DEFAULT_BASE_PATH

_model_lock = threading.Lock()
_embedding_model: Any | None = None

if TYPE_CHECKING:  # pragma: no cover - typing helper only
    from sentence_transformers import SentenceTransformer as _SentenceTransformer


def _load_sentence_transformer(model_name: str) -> Any:
    """Import SentenceTransformer lazily to avoid heavy startup cost when unused."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@dataclass(slots=True)
class _DocumentPages:
    path: Path
    mtime: float
    page_count: int
    pages: list[str]


@dataclass(slots=True)
class PDFChunk:
    chunk_id: str
    page_number: int
    start_char: int
    end_char: int
    text: str


@dataclass(slots=True)
class _PDFIndex:
    path: Path
    chunk_size: int
    chunk_overlap: int
    mtime: float
    chunks: list[PDFChunk]
    embedding_chunks: list[PDFChunk]
    embeddings: np.ndarray


_DOCUMENT_CACHE: dict[Path, _DocumentPages] = {}
_INDEX_CACHE: dict[tuple[Path, int, int], _PDFIndex] = {}


def set_base_path(path: str | Path | None) -> None:
    global _base_path
    if path is None:
        _base_path = None
    else:
        _base_path = Path(path).resolve()


def reset_base_path() -> None:
    set_base_path(_DEFAULT_BASE_PATH)


def set_embedding_model(model: Any | None) -> None:
    global _embedding_model
    with _model_lock:
        _embedding_model = model


def get_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                _embedding_model = _load_sentence_transformer(DEFAULT_MODEL_NAME)
    return _embedding_model


def resolve_pdf_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    base_for_relative = _base_path or Path.cwd()
    if not candidate.is_absolute():
        candidate = (base_for_relative / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate.suffix.lower() != ".pdf":
        raise ValueError("Only .pdf files are supported.")

    if not candidate.exists():
        raise FileNotFoundError(f"File not found: {candidate}")

    if _base_path and _base_path.exists():
        try:
            candidate.relative_to(_base_path)
        except ValueError as exc:
            raise PermissionError(
                f"The file must reside under the allowed base path: {_base_path}"
            ) from exc

    return candidate


def read_pdf(
    path: str,
    start_page: int = 1,
    end_page: int | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> PDFReadResponse:
    pdf_path = resolve_pdf_path(path)
    start_page = max(1, start_page)

    with cast(Any, fitz.open(pdf_path)) as doc:
        total_pages = int(doc.page_count)
        last_page = end_page if end_page is not None else total_pages
        last_page = min(last_page, total_pages)

        if start_page > last_page:
            raise ValueError("Invalid page range.")

        allowed_last = min(last_page, start_page + max_pages - 1)
        pages: list[PDFPage] = []
        for idx in range(start_page - 1, allowed_last):
            page_text = cast(str, doc.load_page(idx).get_text("text"))
            pages.append(PDFPage(page_number=idx + 1, text=page_text))

    return PDFReadResponse(path=str(pdf_path), page_count=total_pages, pages=pages)


def search_pdf(
    path: str,
    query: str,
    top_k: int = 5,
    min_score: float = 0.25,
) -> PDFSearchResponse:
    pdf_path = resolve_pdf_path(path)
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    index: _PDFIndex = _get_index(pdf_path, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    if not index.embedding_chunks or not index.embeddings.size:
        return PDFSearchResponse(path=str(pdf_path), query=query, results=[])

    model = get_embedding_model()
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)

    scores = np.dot(index.embeddings, query_vec)
    order = np.argsort(-scores)[: max(1, top_k)]

    results: list[PDFSearchHit] = []
    for idx in order:
        score = float(scores[idx])
        if score < min_score:
            continue
        chunk = cast(PDFChunk, index.embedding_chunks[idx])
        results.append(
            PDFSearchHit(
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                score=score,
                text=chunk.text.strip(),
            )
        )

    return PDFSearchResponse(path=str(pdf_path), query=query, results=results)


def describe_pdf_sections(
    path: str,
    max_chunks: int = 20,
    chunk_size: int | None = None,
) -> PDFChunkResponse:
    pdf_path = resolve_pdf_path(path)
    chunk_len = chunk_size or DEFAULT_CHUNK_SIZE
    doc = _get_document(pdf_path)
    chunks = _build_chunks(doc.pages, chunk_len, DEFAULT_CHUNK_OVERLAP, pdf_path.name)
    selected = chunks[: max_chunks]

    chunk_infos = [
        PDFChunkInfo(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            text=chunk.text.strip(),
            embedding_model=DEFAULT_MODEL_NAME,
        )
        for chunk in selected
    ]

    return PDFChunkResponse(path=str(pdf_path), chunk_count=len(chunk_infos), chunks=chunk_infos)


def _get_document(path: Path) -> _DocumentPages:
    cached = _DOCUMENT_CACHE.get(path)
    mtime = path.stat().st_mtime
    if cached and cached.mtime == mtime:
        return cached

    with cast(Any, fitz.open(path)) as doc:
        page_count = int(doc.page_count)
        pages: list[str] = [
            cast(str, doc.load_page(i).get_text("text")) for i in range(page_count)
        ]
        cached = _DocumentPages(
            path=path,
            mtime=mtime,
            page_count=page_count,
            pages=pages,
        )
        _DOCUMENT_CACHE[path] = cached
        return cached


def _get_index(path: Path, chunk_size: int, chunk_overlap: int) -> _PDFIndex:
    key = (path, chunk_size, chunk_overlap)
    cached = _INDEX_CACHE.get(key)
    mtime = path.stat().st_mtime
    if cached and cached.mtime == mtime:
        return cached

    doc = _get_document(path)
    chunks = _build_chunks(doc.pages, chunk_size, chunk_overlap, path.name)
    embeddable: list[PDFChunk] = [chunk for chunk in chunks if chunk.text.strip()]

    if embeddable:
        model = get_embedding_model()
        chunk_texts: list[str] = [chunk.text for chunk in embeddable]
        embeddings = model.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)

    cached = _PDFIndex(
        path=path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        mtime=mtime,
        chunks=chunks,
        embedding_chunks=embeddable,
        embeddings=embeddings,
    )
    _INDEX_CACHE[key] = cached
    return cached


def _build_chunks(
    pages: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
    file_stem: str,
) -> list[PDFChunk]:
    chunk_size = max(100, chunk_size)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    chunks: list[PDFChunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        text = page_text.strip()
        if not text:
            continue
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            chunk_id = f"{file_stem}-p{page_number}-c{chunk_idx}"
            chunks.append(
                PDFChunk(
                    chunk_id=chunk_id,
                    page_number=page_number,
                    start_char=start,
                    end_char=end,
                    text=chunk_text,
                )
            )
            chunk_idx += 1
            if end == len(text):
                break
            start = end - chunk_overlap

    return chunks


__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MAX_PAGES",
    "DEFAULT_MODEL_NAME",
    "PDFChunk",
    "describe_pdf_sections",
    "get_embedding_model",
    "read_pdf",
    "reset_base_path",
    "resolve_pdf_path",
    "search_pdf",
    "set_base_path",
    "set_embedding_model",
]

