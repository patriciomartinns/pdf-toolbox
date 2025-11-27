<div align="center">

# MCP PDF Reader

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/github/license/patriciomartinns/mcp-pdf-reader)
![Status](https://img.shields.io/badge/status-active-success)

Expose local PDFs to MCP-compatible agents with deterministic chunking, semantic embeddings, and reproducible ingestion pipelines.

</div>

---

### Table of Contents

1. [Highlights](#highlights)
2. [Requirements](#requirements)
3. [Quick Start](#quick-start)
4. [Client Configuration](#client-configuration)
5. [MCP Tools](#mcp-tools)
6. [Usage Recipes](#usage-recipes)
7. [Advanced Parameters](#advanced-parameters)
8. [Development](#development)
9. [Testing & Quality](#testing--quality)
10. [Troubleshooting](#troubleshooting)
11. [Support](#support)

---

### Highlights

- FastMCP transport for zero-boilerplate STDIO integration.
- PyMuPDF-based extraction with strict `.pdf` validation and base-path sandboxing.
- SentenceTransformers embeddings (`all-MiniLM-L6-v2` by default) + cosine similarity search.
- Deterministic chunk generator for downstream RAG pipelines.
- Distributed as a single `uvx` command—no local environment setup required for consumers.

### Requirements

- Python ≥ **3.13** (enforced in `pyproject.toml`)
- [`uv`](https://github.com/astral-sh/uv) CLI to install/run via `uvx`
- Access to the PDFs you plan to expose (optionally confined to a base directory)

### Quick Start

```bash
uvx --from git+https://github.com/patriciomartinns/mcp-pdf-reader -- mcp-pdf-reader --quiet
# inspect options / banner
uvx --from git+https://github.com/patriciomartinns/mcp-pdf-reader -- mcp-pdf-reader --help
```

> The standalone `--` separates `uvx` flags from CLI flags. `--quiet` hides the banner only—the transport stays STDIO.

### Client Configuration

All MCP clients can reuse the same server definition.

#### Cursor / VS Code (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "PdfReader": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/patriciomartinns/mcp-pdf-reader",
        "--",
        "mcp-pdf-reader",
        "--quiet"
      ]
    }
  }
}
```

> Cursor expects an `mcpServers` object at the top level; each key is a server definition. Swap `--quiet` for `--help` (or remove it) if you want the banner back.

#### Claude Desktop / Claude Code

1. Open **MCP Integrations**.
2. Add a server with `uvx --from git+https://github.com/patriciomartinns/mcp-pdf-reader -- mcp-pdf-reader`.
3. Append `--quiet` after `mcp-pdf-reader` if you prefer silent startups.

#### CLI Agents (Serena, Cline, OpenHands, etc.)

```json
{
  "name": "PdfReader",
  "command": "uvx",
  "args": [
    "--from",
    "git+https://github.com/patriciomartinns/mcp-pdf-reader",
    "--",
    "mcp-pdf-reader"
  ]
}
```

Any STDIO-aware MCP client can reuse the snippet verbatim.

### MCP Tools

| Tool | Purpose | Key Parameters / Defaults |
|------|---------|---------------------------|
| `read_pdf` | Extract ordered text for a bounded page range (default max 25 pages). | `path`, `start_page=1`, `end_page=None`, `max_pages=25` |
| `search_pdf` | Run semantic similarity over cached embeddings and return ranked hits. | `path`, `query`, `top_k=5`, `min_score=0.25`, `chunk_size=None`, `chunk_overlap=None` |
| `describe_pdf_sections` | Generate deterministic chunks with offsets for downstream ingestion. | `path`, `max_chunks=20`, `chunk_size=None`, `chunk_overlap=None` |
| `configure_pdf_defaults` | Update runtime defaults for pagination, chunking, and the embedding model. | `chunk_size=None`, `chunk_overlap=None`, `max_pages=None`, `embedding_model=None` |

All tools enforce `.pdf` extensions, resolve relative paths against the configured base directory, and throw permission errors if a file escapes that sandbox.

### Usage Recipes

- **Read specific pages**

  ```text
  Tool: read_pdf
  Args: { "path": "docs/whitepaper.pdf", "start_page": 3, "end_page": 5 }
  ```

  Returns ordered pages plus the total page count so the agent can paginate follow-up calls.

- **Configure runtime defaults**

  ```text
  Tool: configure_pdf_defaults
  Args: {
    "chunk_size": 600,
    "chunk_overlap": 150,
    "max_pages": 10,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
  ```

  Useful when you want subsequent calls to honor a different chunk window, overlap, or page cap without repeating parameters.

- **Semantic search**

  ```text
  Tool: search_pdf
  Args: { "path": "manual.pdf", "query": "rate limiting strategy", "top_k": 8 }
  ```

  Scores are cosine similarities in `[0,1]`. Values below `min_score` are automatically dropped.

- **Chunk description**

  ```text
  Tool: describe_pdf_sections
  Args: { "path": "~/Reports/Q4.pdf", "max_chunks": 10 }
  ```

  Responses list `chunk_id`, page number, character offsets, and the embedding model used (`all-MiniLM-L6-v2` by default).

### Advanced Parameters

- In Cursor, press ⌃⇧J (Ctrl+Shift+J) to open **Settings → Features → Model Context Protocol**, then pick the `PdfReader` server to edit the JSON arguments before invoking each tool.
- All optional parameters work even without a dedicated UI; simply add the fields to the JSON payload.
- Examples:

  ```text
  Tool: search_pdf
  Args: {
    "path": "learning/Cookbook.pdf",
    "query": "show service",
    "top_k": 8,
    "min_score": 0.2,
    "chunk_size": 400,
    "chunk_overlap": 120
  }
  ```

- Use `configure_pdf_defaults` to set new defaults once per session instead of repeating the same arguments on every call. The server returns the current state and applies validation.

  ```text
  Tool: describe_pdf_sections
  Args: {
    "path": "learning/Treinamento.pdf",
    "max_chunks": 5,
    "chunk_size": 600,
    "chunk_overlap": 150
  }
  ```

- If no advanced values are provided, the server sticks to the safe defaults (750 characters per chunk, 75 characters of overlap, 25 pages per call).

### Development

1. Clone the repository.
2. Sync dependencies with `uv sync` (installs runtime + dev groups).
3. Run the server locally: `uv run mcp-pdf-reader --help`.

Packaged via Hatchling (`pyproject.toml`) and exposes the CLI entry-point `mcp_pdf_reader.cli:start_mcp_server`.

### Testing & Quality

```bash
uv run pytest            # unit tests (see tests/test_pdf_tools.py)
uv run ruff check .      # lint: E/F/I plus import sorting
uv run pyright           # strict type checking (py314 target)
uv run bandit -r src     # static security audit
uv run check             # run the full pipeline above in sequence
```

`uv run ruff format` keeps the style consistent (line length 100, double quotes).

### Troubleshooting

- **File not found / permission errors**: make sure the PDF lives under the allowed base path (defaults to the working directory when the server starts).
- **Empty search results**: ensure the PDF contains extractable text; images-only documents require OCR first.
- **Slow first query**: the SentenceTransformer model loads lazily on the first semantic call and is cached afterward.

### Support

Open an issue or discussion at [github.com/patriciomartinns/mcp-pdf-reader](https://github.com/patriciomartinns/mcp-pdf-reader) for questions, roadmap requests, or bug reports.

### License & Contributing

- Licensed under the [MIT License](LICENSE).
- Contributions welcome—see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, workflow, and Conventional Commits guidelines.