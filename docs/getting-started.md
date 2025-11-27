# PDF Toolbox

Official usage guide for PDF Toolbox. Inside you will find how to:

- Install both the MCP server and the `pdf-reader` CLI using `uv`.
- Run the tools locally from the terminal (see [CLI Reference](cli-reference.md)).
- Connect the server to Cursor, VS Code, Claude, and other MCP clients (see [MCP Integration](mcp-integration.md)).

## Feature Overview

- **read_pdf** – extract ordered text with an optional page window.
- **search_pdf** – run semantic search via SentenceTransformers (`all-MiniLM-L6-v2` by default).
- **describe_pdf_sections** – produce deterministic chunks with offsets for RAG workflows.
- **configure_pdf_defaults** – adjust chunking/page limits/model defaults at runtime.

## Recommended Installation (uv)

```bash
# Install the CLI locally
uv tool install --from git+https://github.com/patriciomartinns/pdf-toolbox pdf-reader

# Run ad-hoc without installing
uvx --from git+https://github.com/patriciomartinns/pdf-toolbox -- pdf-reader --help
```

Start the MCP server (STDIO transport):

```bash
uvx --from git+https://github.com/patriciomartinns/pdf-toolbox -- pdf-toolbox --quiet
```

## Next Steps

- [CLI Reference](cli-reference.md) – commands, parameters, and examples.
- [MCP Integration](mcp-integration.md) – snippets for popular clients.
- [Advanced Options](advanced-options.md) – table-detection mode and other power-user tweaks.
- [GitHub Repository](https://github.com/patriciomartinns/pdf-toolbox) – issues, releases, roadmap.

## Known Warnings (PyMuPDF / SWIG)

PyMuPDF currently emits `DeprecationWarning: builtin type SwigPy* has no __module__ attribute` on Python 3.11+ because its macOS wheels are still built with SWIG 4.3.1. The underlying SWIG bug is fixed upstream (see [swig/swig#2881](https://github.com/swig/swig/issues/2881)), but PyMuPDF cannot yet consume SWIG 4.4.0 on macOS due to [swig/swig#3279](https://github.com/swig/swig/issues/3279) (see maintainer comments in [pymupdf/PyMuPDF#3931](https://github.com/pymupdf/PyMuPDF/issues/3931#issuecomment-3533106306)).  

Until Artifex ships PyMuPDF builds linked against the fixed SWIG toolchain on every platform, we suppress the warning locally so logs stay clean; nothing functional is impacted.
