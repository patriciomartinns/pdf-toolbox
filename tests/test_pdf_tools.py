from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Sequence, cast

import fitz  # type: ignore[import]
import numpy as np
import numpy.typing as npt
import pytest

from pdf_toolbox import pdf_tools
from pdf_toolbox.services import pdf_reader as pdf_service

VectorArray = npt.NDArray[np.float32]


class _DeterministicModel:
    def __init__(self, bias: float = 0.0) -> None:
        self.bias = bias

    def encode(
        self,
        texts: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> VectorArray:
        vectors: list[VectorArray] = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % 997
            value = np.clip(seed / 997.0 + self.bias, -1.0, 1.0)
            vectors.append(np.array([value], dtype=np.float32))
        return np.vstack(vectors)


@pytest.fixture(autouse=True)
def isolate_base_path(tmp_path: Path):
    pdf_tools.set_base_path(tmp_path)
    yield
    pdf_tools.reset_base_path()


@pytest.fixture(autouse=True)
def reset_defaults():
    pdf_tools.configure_pdf_defaults(
        chunk_size=pdf_tools.DEFAULT_CHUNK_SIZE,
        chunk_overlap=pdf_tools.DEFAULT_CHUNK_OVERLAP,
        max_pages=pdf_tools.DEFAULT_MAX_PAGES,
        embedding_model=pdf_tools.DEFAULT_MODEL_NAME,
    )
    pdf_tools.set_embedding_model(None)
    yield
    pdf_tools.configure_pdf_defaults(
        chunk_size=pdf_tools.DEFAULT_CHUNK_SIZE,
        chunk_overlap=pdf_tools.DEFAULT_CHUNK_OVERLAP,
        max_pages=pdf_tools.DEFAULT_MAX_PAGES,
        embedding_model=pdf_tools.DEFAULT_MODEL_NAME,
    )
    pdf_tools.set_embedding_model(None)


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    doc = cast(Any, fitz.open())  # type: ignore[call-arg]
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Machine learning insights and embeddings.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Chunking text allows semantic PDF search.")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture()
def table_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "table.pdf"
    doc = cast(Any, fitz.open())  # type: ignore[call-arg]
    page = doc.new_page()
    table_rect = fitz.Rect(72, 72, 300, 200)
    page.draw_rect(table_rect, color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(72, 136), fitz.Point(300, 136), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(186, 72), fitz.Point(186, 200), color=(0, 0, 0), width=1)
    page.insert_text((82, 92), "Quarter")
    page.insert_text((200, 92), "Revenue")
    page.insert_text((82, 158), "Q1")
    page.insert_text((200, 158), "$10M")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _create_pdf(tmp_path: Path, name: str, text: str | None) -> Path:
    pdf_path = tmp_path / name
    doc = cast(Any, fitz.open())  # type: ignore[call-arg]
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_read_pdf_returns_expected_pages(sample_pdf: Path) -> None:
    response = pdf_tools.read_pdf(str(sample_pdf))
    assert response.page_count == 2
    assert len(response.pages) == 2
    assert "Machine learning" in response.pages[0].text


def test_search_pdf_uses_embeddings(sample_pdf: Path) -> None:
    pdf_tools.set_embedding_model(_DeterministicModel())
    response = pdf_tools.search_pdf(
        path=str(sample_pdf),
        query="semantic chunking",
        top_k=2,
        min_score=-1.0,
    )
    assert response.results
    assert response.results[0].score <= 1.0
    assert response.results[0].page_number in (1, 2)


def test_describe_pdf_sections_limits_chunks(sample_pdf: Path) -> None:
    result = pdf_tools.describe_pdf_sections(
        path=str(sample_pdf),
        max_chunks=1,
        chunk_size=200,
        chunk_overlap=50,
    )
    assert result.chunk_count == 1
    assert result.chunks[0].start_char == 0
    assert result.chunks[0].page_number == 1


def test_search_pdf_rebuilds_index_when_model_changes(sample_pdf: Path) -> None:
    pdf_tools.set_embedding_model(_DeterministicModel(bias=0.0))
    response_initial = pdf_tools.search_pdf(
        path=str(sample_pdf),
        query="semantic chunking",
        top_k=1,
        min_score=-1.0,
    )
    initial_score = response_initial.results[0].score

    pdf_tools.set_embedding_model(_DeterministicModel(bias=0.5))
    response_updated = pdf_tools.search_pdf(
        path=str(sample_pdf),
        query="semantic chunking",
        top_k=1,
        min_score=-1.0,
    )
    updated_score = response_updated.results[0].score
    pdf_tools.set_embedding_model(None)

    assert initial_score != updated_score


def test_configure_pdf_defaults_updates_values() -> None:
    response = pdf_tools.configure_pdf_defaults(
        chunk_size=320,
        chunk_overlap=80,
        max_pages=5,
        embedding_model=pdf_tools.DEFAULT_MODEL_NAME,
    )
    assert response.chunk_size == 320
    assert response.chunk_overlap == 80
    assert response.max_pages == 5
    assert response.embedding_model == pdf_tools.DEFAULT_MODEL_NAME

    second = pdf_tools.configure_pdf_defaults()
    assert second.chunk_size == 320
    assert second.chunk_overlap == 80


def test_describe_pdf_sections_respects_runtime_defaults(sample_pdf: Path) -> None:
    pdf_tools.configure_pdf_defaults(chunk_size=150, chunk_overlap=30)
    result = pdf_tools.describe_pdf_sections(path=str(sample_pdf), max_chunks=2)
    assert result.chunks
    first_chunk = result.chunks[0]
    assert first_chunk.end_char - first_chunk.start_char <= 150


def test_describe_pdf_sections_chunk_params_affect_output(tmp_path: Path) -> None:
    pdf_path = tmp_path / "chunking.pdf"
    doc = cast(Any, fitz.open())  # type: ignore[call-arg]
    page = doc.new_page()
    line = "Chunk parameters should impact chunking output. " * 4
    for idx in range(60):
        page.insert_text((72, 72 + idx * 12), line)
    doc.save(pdf_path)
    doc.close()

    tight = pdf_tools.describe_pdf_sections(
        str(pdf_path), chunk_size=120, chunk_overlap=10, max_chunks=25
    )
    wide = pdf_tools.describe_pdf_sections(
        str(pdf_path), chunk_size=600, chunk_overlap=10, max_chunks=25
    )

    assert tight.chunk_count >= 2
    assert tight.chunk_count > wide.chunk_count
    assert tight.chunks[0].end_char - tight.chunks[0].start_char <= 120


def test_describe_pdf_sections_table_mode(table_pdf: Path) -> None:
    result = pdf_tools.describe_pdf_sections(
        path=str(table_pdf),
        max_chunks=3,
        mode="tables",
    )
    assert result.mode == "tables"
    assert result.table_count >= 1
    assert result.tables
    first_table = result.tables[0]
    assert first_table.rows
    first_row = first_table.rows[0]
    assert first_row.cells
    assert "Quarter" in first_row.cells[0].text or "Revenue" in first_row.cells[0].text


def test_read_pdf_invalid_page_range_raises(sample_pdf: Path) -> None:
    with pytest.raises(ValueError, match="Invalid page range"):
        pdf_tools.read_pdf(str(sample_pdf), start_page=5, end_page=1)


def test_resolve_pdf_path_rejects_non_pdf(tmp_path: Path) -> None:
    bad_file = tmp_path / "note.txt"
    bad_file.write_text("hello")
    with pytest.raises(ValueError, match="Only .pdf files"):
        pdf_tools.resolve_pdf_path(str(bad_file))


def test_resolve_pdf_path_permission_error(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    pdf_file = outside / "doc.pdf"
    pdf_file.write_text("fake pdf")
    pdf_tools.set_base_path(str(allowed))
    try:
        with pytest.raises(PermissionError):
            pdf_tools.resolve_pdf_path(str(pdf_file))
    finally:
        pdf_tools.reset_base_path()


def test_configure_pdf_defaults_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        pdf_tools.configure_pdf_defaults(embedding_model="  ")


def test_get_embedding_model_loads_custom(monkeypatch: Any) -> None:
    sentinel = object()

    def fake_loader(model_name: str) -> object:
        return (model_name, sentinel)

    monkeypatch.setattr(pdf_service, "_load_sentence_transformer", fake_loader)  # pyright: ignore[reportPrivateUsage]
    pdf_tools.configure_pdf_defaults(embedding_model="sentence-transformers/test-model")
    model = pdf_tools.get_embedding_model()
    assert model == ("sentence-transformers/test-model", sentinel)


def test_load_sentence_transformer_invokes_sentence_transformers(monkeypatch: Any) -> None:
    class DummyModel:
        def __init__(self, name: str) -> None:
            self.name = name

    dummy_module = types.SimpleNamespace(SentenceTransformer=DummyModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy_module)
    pdf_tools.configure_pdf_defaults(embedding_model="custom-model")
    pdf_tools.set_embedding_model(None)
    model = pdf_tools.get_embedding_model()
    assert isinstance(model, DummyModel)
    assert model.name == "custom-model"


def test_search_pdf_requires_non_empty_query(sample_pdf: Path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        pdf_tools.search_pdf(path=str(sample_pdf), query="   ")


def test_search_pdf_returns_empty_for_blank_document(tmp_path: Path) -> None:
    _create_pdf(tmp_path, "blank.pdf", text=None)
    pdf_tools.set_base_path(tmp_path)
    try:
        response = pdf_tools.search_pdf(path="blank.pdf", query="anything")
        assert response.results == []
    finally:
        pdf_tools.reset_base_path()


def test_document_cache_eviction(monkeypatch: Any, tmp_path: Path) -> None:
    pdf_service._DOCUMENT_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(pdf_service, "MAX_DOCUMENT_CACHE", 1)
    first = _create_pdf(tmp_path, "first.pdf", "first")
    second = _create_pdf(tmp_path, "second.pdf", "second")
    pdf_tools.describe_pdf_sections(str(first))
    pdf_tools.describe_pdf_sections(str(second))
    assert len(pdf_service._DOCUMENT_CACHE) == 1  # pyright: ignore[reportPrivateUsage]
    cached_path = next(iter(pdf_service._DOCUMENT_CACHE.keys()))  # pyright: ignore[reportPrivateUsage]
    assert cached_path.name == "second.pdf"


def test_index_cache_eviction(monkeypatch: Any, tmp_path: Path) -> None:
    pdf_service._INDEX_CACHE.clear()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(pdf_service, "MAX_INDEX_CACHE", 1)
    pdf_tools.set_embedding_model(_DeterministicModel())
    first = _create_pdf(tmp_path, "first.pdf", "first text")
    second = _create_pdf(tmp_path, "second.pdf", "second text")
    pdf_tools.search_pdf(str(first), query="first", min_score=-1.0)
    pdf_tools.search_pdf(str(second), query="second", min_score=-1.0)
    assert len(pdf_service._INDEX_CACHE) == 1  # pyright: ignore[reportPrivateUsage]
    cached_key = next(iter(pdf_service._INDEX_CACHE.keys()))  # pyright: ignore[reportPrivateUsage]
    assert cached_key[0].name == "second.pdf"
    pdf_tools.set_embedding_model(None)


def test_describe_pdf_sections_skips_blank_pages(tmp_path: Path) -> None:
    blank_pdf = _create_pdf(tmp_path, "blank.pdf", text=None)
    result = pdf_tools.describe_pdf_sections(str(blank_pdf))
    assert result.chunk_count == 0


def test_build_chunks_overlap_behavior() -> None:
    chunks = pdf_service._build_chunks(["A" * 220], 120, 40, "file")  # pyright: ignore[reportPrivateUsage]
    assert len(chunks) >= 2
    assert chunks[1].start_char == chunks[0].end_char - 40


def test_resolve_pdf_path_missing_file(tmp_path: Path) -> None:
    pdf_tools.set_base_path(tmp_path)
    with pytest.raises(FileNotFoundError):
        pdf_tools.resolve_pdf_path("missing.pdf")


def test_set_base_path_none_allows_absolute_paths(tmp_path: Path) -> None:
    outside_root = tmp_path.parent / f"{tmp_path.name}-outside"
    outside_root.mkdir(exist_ok=True)
    pdf_path = _create_pdf(outside_root, "loose.pdf", "loose text")

    pdf_tools.set_base_path(None)
    resolved = pdf_tools.resolve_pdf_path(str(pdf_path))
    assert resolved == pdf_path.resolve()


def test_search_pdf_respects_min_score(sample_pdf: Path) -> None:
    pdf_tools.set_embedding_model(_DeterministicModel(bias=-0.5))
    response = pdf_tools.search_pdf(
        path=str(sample_pdf),
        query="semantic chunking",
        top_k=5,
        min_score=0.99,
    )
    assert response.results == []
    pdf_tools.set_embedding_model(None)


def test_get_index_uses_cache(tmp_path: Path) -> None:
    pdf_tools.set_embedding_model(_DeterministicModel())
    pdf_path = _create_pdf(tmp_path, "cache.pdf", "cache coverage")
    index_first = pdf_service._get_index(pdf_path, 120, 40)  # pyright: ignore[reportPrivateUsage]
    index_second = pdf_service._get_index(pdf_path, 120, 40)  # pyright: ignore[reportPrivateUsage]
    assert index_first is index_second
    pdf_tools.set_embedding_model(None)
