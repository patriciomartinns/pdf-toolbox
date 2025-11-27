from __future__ import annotations

from typing import Any, Callable, Dict, cast

import pytest

from pdf_toolbox.services import pdf_reader as service_module
from pdf_toolbox.tools import pdf_reader as tools_module


class _DummyMCP:
    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}

    def tool(self, description: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[func.__name__] = {"func": func, "description": description}
            return func

        return decorator


def _wrap(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {name: payload}


def test_register_pdf_tools_wires_service_functions(monkeypatch: Any) -> None:
    dummy = _DummyMCP()

    def fake_read(**kwargs: Any) -> Dict[str, Any]:
        return _wrap("read", kwargs)

    def fake_search(**kwargs: Any) -> Dict[str, Any]:
        return _wrap("search", kwargs)

    def fake_describe(**kwargs: Any) -> Dict[str, Any]:
        return _wrap("describe", kwargs)

    def fake_config(**kwargs: Any) -> Dict[str, Any]:
        return _wrap("config", kwargs)

    monkeypatch.setattr(service_module, "read_pdf", fake_read)
    monkeypatch.setattr(service_module, "search_pdf", fake_search)
    monkeypatch.setattr(service_module, "describe_pdf_sections", fake_describe)
    monkeypatch.setattr(service_module, "configure_pdf_defaults", fake_config)

    tools_module.register_pdf_tools(cast(Any, dummy))

    assert {"read_pdf", "search_pdf", "describe_pdf_sections", "configure_pdf_defaults"} <= set(
        dummy.tools.keys()
    )

    assert dummy.tools["read_pdf"]["func"](path="doc.pdf")["read"]["path"] == "doc.pdf"
    assert (
        dummy.tools["search_pdf"]["func"](path="doc.pdf", query="foo")["search"]["query"] == "foo"
    )
    assert (
        dummy.tools["describe_pdf_sections"]["func"](path="doc.pdf")["describe"]["path"]
        == "doc.pdf"
    )
    assert (
        dummy.tools["configure_pdf_defaults"]["func"](chunk_size=123)["config"]["chunk_size"]
        == 123
    )


def test_register_pdf_tools_propagates_errors(monkeypatch: Any) -> None:
    dummy = _DummyMCP()

    def boom(**kwargs: Any) -> Dict[str, Any]:
        raise ValueError("kaboom")

    monkeypatch.setattr(service_module, "read_pdf", boom)
    tools_module.register_pdf_tools(cast(Any, dummy))

    with pytest.raises(ValueError, match="kaboom"):
        dummy.tools["read_pdf"]["func"](path="doc.pdf")

