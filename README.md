<div align="center">

# PDF Toolbox

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/github/license/patriciomartinns/pdf-toolbox)
![Status](https://img.shields.io/badge/status-active-success)

Expose local PDFs to MCP-compatible agents or run the standalone `pdf-reader` CLI with deterministic chunking, semantic search, and configurable defaults.

</div>

---

## Highlights

- FastMCP/STDIO server ready for Cursor, VS Code, Claude, and other MCP clients.
- Typer/Click/Rich CLI (`pdf-reader`) prints JSON for easy piping.
- PyMuPDF extraction plus SentenceTransformers embeddings.
- Strict `.pdf` validation, sandboxed base path, and aggressive caching.

## Documentation

- [Getting started](docs/getting-started.md)
- [CLI reference](docs/cli-reference.md)
- [MCP integration](docs/mcp-integration.md)

## Quick install (uv)
# Run the MCP server directly

```bash
# Run the MCP server directly
uvx --from git+https://github.com/patriciomartinns/pdf-toolbox -- pdf-toolbox --quiet

# Install/run the CLI
uv tool install --from git+https://github.com/patriciomartinns/pdf-toolbox pdf-reader
pdf-reader --help
```

> **Note:** If you had the old `mcp-pdf-reader` CLI installed via `uv tool install`, run `uv tool uninstall mcp-pdf-reader` before installing `pdf-reader` to avoid conflicts.

## CLI quick tour

| Command | Purpose | Example |
| --- | --- | --- |
| `pdf-reader read-pdf` | Extract ordered text for a bounded page range. | `pdf-reader read-pdf reports/Q1.pdf --start-page 3 --end-page 5` |
| `pdf-reader search-pdf` | Run semantic similarity search over cached embeddings. | `pdf-reader search-pdf reports/Q1.pdf "rate limiting" --top-k 8` |
| `pdf-reader describe-pdf-sections` | List deterministic chunks with offsets for RAG pipelines. | `pdf-reader describe-pdf-sections reports/Q1.pdf --max-chunks 5` |
| `pdf-reader configure-pdf-defaults` | Update runtime defaults for chunk size/overlap/page window/model. | `pdf-reader configure-pdf-defaults --chunk-size 600 --chunk-overlap 120 --max-pages 10` |

> **Tip:** the first `search-pdf` invocation on a new document downloads the SentenceTransformers model and builds embeddings, so it can take longer once per model/PDF combo. Subsequent searches reuse the cache.

See the [`docs/`](docs) folder for full recipes covering both CLI commands and MCP client configuration. Questions or ideas? Open an issue on [github.com/patriciomartinns/pdf-toolbox](https://github.com/patriciomartinns/pdf-toolbox).