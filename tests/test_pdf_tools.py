from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

import fitz  # type: ignore[import]
import numpy as np
import numpy.typing as npt
import pytest

from mcp_pdf_reader import pdf_tools

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


def test_search_pdf_accepts_chunk_parameters(sample_pdf: Path) -> None:
    pdf_tools.set_embedding_model(_DeterministicModel())
    response = pdf_tools.search_pdf(
        path=str(sample_pdf),
        query="semantic chunking",
        top_k=1,
        min_score=-1.0,
        chunk_size=150,
        chunk_overlap=40,
    )
    pdf_tools.set_embedding_model(None)
    assert response.results

