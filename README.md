## MCP PDF Reader

Expose local PDFs to LLM agents with deterministic chunking, semantic embeddings, and reproducible ingestion pipelines over Machine Control Protocol (MCP).

### Highlights

- Built on FastMCP for zero-boilerplate STDIO transport
- PyMuPDF-powered text extraction with strict `.pdf` validation and base-path sandboxing
- SentenceTransformers embeddings (`all-MiniLM-L6-v2` by default) with cosine-similarity ranking
- Deterministic chunk generator for downstream RAG workflows
- Distributed as a single `uvx` command—no local clone required to consume the server

### Requirements

- Python ≥ **3.13** (enforced em `pyproject.toml`)
- [`uv`](https://github.com/astral-sh/uv) CLI to install and run via `uvx`
- Access to the PDFs you want to expose (optionally confined to a base directory)

### Quick Start (remote execution via `uvx`)

```bash
uvx --from git+https://github.com/patriciomartinns/mcp-pdf-reader -- mcp-pdf-reader --quiet
# inspect options / banner
uvx --from git+https://github.com/patriciomartinns/mcp-pdf-reader -- mcp-pdf-reader --help
```

The standalone `--` separates `uvx` flags from CLI flags. `--quiet` hides the banner only—the transport stays STDIO.

### Client Configuration

All MCP clients should execute the same `uvx` command. Below are ready-to-copy snippets.

#### Cursor / VS Code (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "mcp-pdf": {
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

> Cursor espera um objeto `mcpServers` no topo; cada chave é um servidor. O exemplo acima usa `--quiet`, mas você pode trocar por `--help` ou remover totalmente.

#### Claude Desktop / Claude Code

1. Open *MCP Integrations*.
2. Add a server with `uvx --from git+https://github.com/patriciomartinns/mcp-pdf-reader -- mcp-pdf-reader`.
3. Optionally append `--quiet` (after `mcp-pdf-reader`) for silent startups.

#### CLI Agents (Serena, Cline, OpenHands, etc.)

Use the same `uvx` invocation in your `mcp.json`:

```json
{
  "name": "mcp-pdf",
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
| `read_pdf` | Extract ordered text for a bounded page range (max 25 pages per call). | `path`, `start_page=1`, `end_page=None`, `max_pages=25` |
| `search_pdf` | Run semantic similarity over cached embeddings and return ranked hits. | `path`, `query`, `top_k=5`, `min_score=0.25`, `chunk_size=None`, `chunk_overlap=None` |
| `describe_pdf_sections` | Generate deterministic chunks with offsets for downstream ingestion. | `path`, `max_chunks=20`, `chunk_size=None`, `chunk_overlap=None` |

All tools enforce `.pdf` extensions, resolve relative paths against the configured base directory, and will throw permission errors if a file escapes that sandbox.

### Usage Examples

- **Read specific pages**

  ```text
  Tool: read_pdf
  Args: { "path": "docs/whitepaper.pdf", "start_page": 3, "end_page": 5 }
  ```

  Returns ordered pages plus the total page count so the agent can paginate follow-up calls.

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

### Advanced Parameters (Cursor, VS Code, etc.)

- Ao abrir o painel MCP do Cursor (⌘⇧C → *Servers* → `mcp-pdf`), edite o JSON de argumentos antes de executar a ferramenta.
- Todos os parâmetros extras estão disponíveis mesmo sem suporte visual dedicado; basta incluir o campo no objeto enviado.
- Exemplos úteis:

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

  ```text
  Tool: describe_pdf_sections
  Args: {
    "path": "learning/Treinamento.pdf",
    "max_chunks": 5,
    "chunk_size": 600,
    "chunk_overlap": 150
  }
  ```

- Caso nenhum valor avançado seja informado, o servidor mantém os defaults seguros (750 caracteres por chunk e 75 de overlap).

### Development

1. Clone the repository and move into it.
2. Sync dependencies with `uv sync` (installs runtime + dev groups).
3. Run the server locally: `uv run mcp-pdf-reader --help`.

The project is packaged via Hatchling (`pyproject.toml`) and exposes the CLI entry-point `mcp_pdf_reader.cli:start_mcp_server`.

### Testing & Quality

```bash
uv run pytest            # unit tests (see tests/test_pdf_tools.py)
uv run ruff check .      # lint: E/F/I plus import sorting
uv run pyright           # strict type checking (py314 target)
```

`uv run ruff format` follows the repo’s style (line length 100, double quotes).

### Troubleshooting

- **File not found / permission errors**: ensure the PDF lives under the allowed base path (defaults to the current working directory when the server starts).
- **Empty search results**: make sure the PDF has extractable text; images-only PDFs require OCR before use.
- **Slow first query**: the SentenceTransformer model loads lazily on the first semantic call and is cached afterward.

### Support

Open an issue or discussion em [github.com/patriciomartinns/mcp-pdf-reader](https://github.com/patriciomartinns/mcp-pdf-reader) para dúvidas, roadmap ou bugs.