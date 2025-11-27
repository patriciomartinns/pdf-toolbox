from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast

with warnings.catch_warnings():
    # Same upstream SWIG warning as documented in
    # https://github.com/pymupdf/PyMuPDF/issues/3931 — silence it just for the
    # PyMuPDF import so normal user warnings still surface elsewhere.
    warnings.filterwarnings(
        "ignore",
        message=r".*Swig.* has no __module__ attribute",
        category=DeprecationWarning,
    )
    import fitz  # type: ignore[import]

import numpy as np

from ..schemas import (
    PDFChunkInfo,
    PDFChunkResponse,
    PDFConfigResponse,
    PDFPage,
    PDFReadResponse,
    PDFSearchHit,
    PDFSearchResponse,
    PDFTable,
    PDFTableCell,
    PDFTableRow,
)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 750
DEFAULT_CHUNK_OVERLAP = 75
DEFAULT_MAX_PAGES = 25
_DEFAULT_BASE_PATH = Path.cwd().resolve()
_base_path: Path | None = _DEFAULT_BASE_PATH
MAX_DOCUMENT_CACHE = 16
MAX_INDEX_CACHE = 16

_chunk_size_default = DEFAULT_CHUNK_SIZE
_chunk_overlap_default = DEFAULT_CHUNK_OVERLAP
_max_pages_default = DEFAULT_MAX_PAGES
_model_name = DEFAULT_MODEL_NAME
_model_lock = threading.Lock()
_embedding_model: Any | None = None
_cache_lock = threading.RLock()
_embedding_epoch = 0

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


_DOCUMENT_CACHE: OrderedDict[Path, _DocumentPages] = OrderedDict()
_INDEX_CACHE: OrderedDict[tuple[Path, int, int, int], _PDFIndex] = OrderedDict()


def _clear_document_cache() -> None:
    with _cache_lock:
        _DOCUMENT_CACHE.clear()


def _clear_index_cache(*, bump_epoch: bool = False) -> None:
    global _embedding_epoch
    with _cache_lock:
        _INDEX_CACHE.clear()
        if bump_epoch:
            _embedding_epoch += 1


def _remember_document(path: Path, entry: _DocumentPages) -> None:
    with _cache_lock:
        _DOCUMENT_CACHE[path] = entry
        _DOCUMENT_CACHE.move_to_end(path)
        if len(_DOCUMENT_CACHE) > MAX_DOCUMENT_CACHE:
            _DOCUMENT_CACHE.popitem(last=False)


def _remember_index(key: tuple[Path, int, int, int], entry: _PDFIndex) -> None:
    with _cache_lock:
        _INDEX_CACHE[key] = entry
        _INDEX_CACHE.move_to_end(key)
        if len(_INDEX_CACHE) > MAX_INDEX_CACHE:
            _INDEX_CACHE.popitem(last=False)


def _normalize_chunk_params(
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> tuple[int, int]:
    size = chunk_size if chunk_size is not None else _chunk_size_default
    size = max(100, size)
    overlap = chunk_overlap if chunk_overlap is not None else _chunk_overlap_default
    overlap = max(0, min(overlap, size // 2))
    return size, overlap


def _build_config_response() -> PDFConfigResponse:
    return PDFConfigResponse(
        chunk_size=_chunk_size_default,
        chunk_overlap=_chunk_overlap_default,
        max_pages=_max_pages_default,
        embedding_model=_model_name,
    )


def set_base_path(path: str | Path | None) -> None:
    global _base_path
    if path is None:
        _base_path = None
    else:
        _base_path = Path(path).resolve()
    _clear_document_cache()
    _clear_index_cache()


def reset_base_path() -> None:
    set_base_path(_DEFAULT_BASE_PATH)


def configure_pdf_defaults(
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    max_pages: int | None = None,
    embedding_model: str | None = None,
) -> PDFConfigResponse:
    global _chunk_size_default, _chunk_overlap_default, _max_pages_default
    global _model_name, _embedding_model
    invalidate_index = False

    if chunk_size is not None or chunk_overlap is not None:
        new_size, new_overlap = _normalize_chunk_params(chunk_size, chunk_overlap)
        if new_size != _chunk_size_default or new_overlap != _chunk_overlap_default:
            _chunk_size_default = new_size
            _chunk_overlap_default = new_overlap
            invalidate_index = True

    if max_pages is not None:
        _max_pages_default = max(1, max_pages)

    if embedding_model is not None:
        model_name = embedding_model.strip()
        if not model_name:
            raise ValueError("embedding_model cannot be empty.")
        with _model_lock:
            if model_name != _model_name:
                _model_name = model_name
                _embedding_model = None
                invalidate_index = True

    if invalidate_index:
        _clear_index_cache(bump_epoch=True)

    return _build_config_response()


def set_embedding_model(model: Any | None) -> None:
    global _embedding_model
    with _model_lock:
        if _embedding_model is model:
            return
        _embedding_model = model
    _clear_index_cache(bump_epoch=True)


def get_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                _embedding_model = _load_sentence_transformer(_model_name)
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
    max_pages: int | None = None,
) -> PDFReadResponse:
    pdf_path = resolve_pdf_path(path)
    start_page = max(1, start_page)
    page_window = max_pages if max_pages is not None else _max_pages_default

    with cast(Any, fitz.open(pdf_path)) as doc:
        total_pages = int(doc.page_count)
        last_page = end_page if end_page is not None else total_pages
        last_page = min(last_page, total_pages)

        if start_page > last_page:
            raise ValueError("Invalid page range.")

        allowed_last = min(last_page, start_page + page_window - 1)
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
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> PDFSearchResponse:
    pdf_path = resolve_pdf_path(path)
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    chunk_len, overlap = _normalize_chunk_params(chunk_size, chunk_overlap)
    index: _PDFIndex = _get_index(pdf_path, chunk_len, overlap)
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
    chunk_overlap: int | None = None,
    mode: Literal["chunks", "tables"] = "chunks",
) -> PDFChunkResponse:
    pdf_path = resolve_pdf_path(path)
    if mode not in {"chunks", "tables"}:
        raise ValueError("mode must be 'chunks' or 'tables'.")

    if mode == "tables":
        tables = _build_tables(pdf_path)
        selected_tables = tables[: max_chunks]
        return PDFChunkResponse(
            path=str(pdf_path),
            chunk_count=0,
            table_count=len(selected_tables),
            chunks=[],
            tables=selected_tables,
            mode="tables",
        )

    chunk_len, overlap = _normalize_chunk_params(chunk_size, chunk_overlap)
    doc = _get_document(pdf_path)
    chunks = _build_chunks(doc.pages, chunk_len, overlap, pdf_path.name)
    selected = chunks[: max_chunks]

    chunk_infos = [
        PDFChunkInfo(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            text=chunk.text.strip(),
            embedding_model=_model_name,
        )
        for chunk in selected
    ]

    return PDFChunkResponse(
        path=str(pdf_path),
        chunk_count=len(chunk_infos),
        table_count=0,
        chunks=chunk_infos,
        mode="chunks",
    )


def _get_document(path: Path) -> _DocumentPages:
    mtime = path.stat().st_mtime
    with _cache_lock:
        cached = _DOCUMENT_CACHE.get(path)
        if cached and cached.mtime == mtime:
            _DOCUMENT_CACHE.move_to_end(path)
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
        _remember_document(path, cached)
        return cached


def _get_index(path: Path, chunk_size: int, chunk_overlap: int) -> _PDFIndex:
    mtime = path.stat().st_mtime
    with _cache_lock:
        epoch = _embedding_epoch
        key = (path, chunk_size, chunk_overlap, epoch)
        cached = _INDEX_CACHE.get(key)
        if cached and cached.mtime == mtime:
            _INDEX_CACHE.move_to_end(key)
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
    _remember_index(key, cached)
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


def _build_tables(path: Path) -> list[PDFTable]:
    tables: list[PDFTable] = []
    with cast(Any, fitz.open(path)) as doc:
        for page_index in range(int(doc.page_count)):
            page = doc.load_page(page_index)
            finder = page.find_tables()
            detected_attr = getattr(finder, "tables", None)
            detected = list(detected_attr) if detected_attr is not None else []
            if not detected:
                try:
                    detected = list(finder)
                except TypeError:
                    detected = []
            if not detected:
                continue
            for table_index, table in enumerate(detected):
                rows, row_total, col_total = _convert_table_rows(table)
                header_names = _extract_header_names(table)
                bbox = _rect_to_tuple(getattr(table, "bbox", None)) or (0.0, 0.0, 0.0, 0.0)
                table_id = f"{path.stem}-p{page_index + 1}-t{table_index}"
                tables.append(
                    PDFTable(
                        table_id=table_id,
                        page_number=page_index + 1,
                        bbox=bbox,
                        headers=header_names,
                        rows=rows,
                        row_count=row_total,
                        column_count=col_total,
                    )
                )
    return tables


def _ensure_row_matrix(raw: Any) -> list[list[Any]]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        seq_raw: Sequence[Any] = cast(Sequence[Any], raw)
        return [_coerce_row_sequence(row) for row in seq_raw]
    return [_coerce_row_sequence(raw)]


def _coerce_row_sequence(row: Any) -> list[Any]:
    if isinstance(row, (list, tuple)):
        seq_row: Sequence[Any] = cast(Sequence[Any], row)
        return list(seq_row)
    if row is None:
        return []
    return [row]


def _convert_table_rows(table: Any) -> tuple[list[PDFTableRow], int, int]:
    extracted = table.extract()
    raw_rows = _ensure_row_matrix(extracted)
    row_total = int(getattr(table, "row_count", len(raw_rows)))
    row_total = max(0, row_total)
    column_total = int(
        getattr(
            table,
            "col_count",
            len(raw_rows[0]) if raw_rows else 0,
        )
    )
    column_total = max(0, column_total)
    cell_boxes = _build_cell_boxes(table, row_total, column_total)
    normalized_rows = _normalize_row_text(raw_rows, row_total, column_total)

    rows: list[PDFTableRow] = []
    for row_idx in range(row_total):
        row_cells: list[PDFTableCell] = []
        if row_idx < len(normalized_rows):
            text_row = normalized_rows[row_idx]
        else:
            text_row = [""] * column_total

        if row_idx < len(cell_boxes):
            box_row = cell_boxes[row_idx]
        else:
            box_row = [None] * column_total
        for col_idx in range(column_total):
            bbox = _rect_to_tuple(box_row[col_idx]) if col_idx < len(box_row) else None
            text_value = text_row[col_idx] if col_idx < len(text_row) else ""
            row_cells.append(
                PDFTableCell(
                    row=row_idx,
                    column=col_idx,
                    bbox=bbox,
                    text=text_value,
                )
            )
        rows.append(PDFTableRow(row=row_idx, cells=row_cells))
    return rows, row_total, column_total


def _build_cell_boxes(table: Any, row_count: int, column_count: int) -> list[list[Any | None]]:
    rows: list[list[Any | None]] = [
        [None for _ in range(column_count)] for _ in range(row_count)
    ]
    cells_attr = getattr(table, "cells", None)
    if cells_attr is None:
        return rows

    if isinstance(cells_attr, (list, tuple)):
        seq_cells: Sequence[Any | None] = cast(Sequence[Any | None], cells_attr)
        cell_list: list[Any | None] = list(seq_cells)
    elif isinstance(cells_attr, Iterable) and not isinstance(cells_attr, (str, bytes)):
        cell_list = list(cast(Iterable[Any | None], cells_attr))
    else:
        return rows
    if not cell_list:
        return rows

    first: Any | None = cell_list[0] if cell_list else None
    has_rect_attr = first is not None and hasattr(first, "x0")
    tuple_candidate: tuple[Any, ...] = (
        cast(tuple[Any, ...], first) if isinstance(first, tuple) else ()
    )
    is_rect_sequence = has_rect_attr or len(tuple_candidate) == 4
    if not is_rect_sequence and isinstance(first, (list, tuple)):
        for row_idx in range(row_count):
            source = cell_list[row_idx] if row_idx < len(cell_list) else None
            row_cells: list[Any | None]
            if isinstance(source, (list, tuple)):
                seq_source: Sequence[Any | None] = cast(Sequence[Any | None], source)
                row_cells = list(seq_source)
            elif source is None:
                row_cells = []
            else:
                row_cells = [source]
            for col_idx in range(column_count):
                if col_idx < len(row_cells):
                    rows[row_idx][col_idx] = row_cells[col_idx]
        return rows

    for row_idx in range(row_count):
        start = row_idx * column_count
        for col_idx in range(column_count):
            flat_index = start + col_idx
            if flat_index < len(cell_list):
                rows[row_idx][col_idx] = cell_list[flat_index]
    return rows


def _normalize_row_text(
    raw_rows: list[list[Any]],
    row_count: int,
    column_count: int,
) -> list[list[str]]:
    normalized: list[list[str]] = []
    for row in raw_rows[:row_count]:
        cell_values: list[Any] = list(row)
        clean_values = [_clean_cell_text(value) for value in cell_values[:column_count]]
        while len(clean_values) < column_count:
            clean_values.append("")
        normalized.append(clean_values)

    while len(normalized) < row_count:
        normalized.append([""] * column_count)
    return normalized


def _clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_header_names(table: Any) -> list[str]:
    header = getattr(table, "header", None)
    if not header:
        return []
    names = getattr(header, "names", None)
    if not names:
        return []
    result: list[str] = []
    for name in names:
        if not name:
            continue
        stripped = str(name).strip()
        if stripped:
            result.append(stripped)
    return result


def _rect_to_tuple(rect: Any | None) -> tuple[float, float, float, float] | None:
    if rect is None:
        return None
    try:
        return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    except AttributeError:
        try:
            x0, y0, x1, y1 = rect
            return (float(x0), float(y0), float(x1), float(y1))
        except (TypeError, ValueError):
            return None


__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MAX_PAGES",
    "DEFAULT_MODEL_NAME",
    "PDFChunk",
    "configure_pdf_defaults",
    "describe_pdf_sections",
    "get_embedding_model",
    "read_pdf",
    "reset_base_path",
    "resolve_pdf_path",
    "search_pdf",
    "set_base_path",
    "set_embedding_model",
]
