<div align="center">

# MCP PDF Reader

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/github/license/patriciomartinns/mcp-pdf-reader)
![Status](https://img.shields.io/badge/status-active-success)

Expose local PDFs to MCP-compatible agents with deterministic chunking, semantic search, and configurable defaults.

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

```bash
# Run the MCP server directly
uvx --from git+https://github.com/patriciomartinns/mcp-pdf-reader -- mcp-pdf-reader --quiet

# Install/run the CLI
uv tool install --from git+https://github.com/patriciomartinns/mcp-pdf-reader pdf-reader
pdf-reader --help
```

See the [`docs/`](docs) folder for full examples covering CLI commands and MCP client configuration. Questions or ideas? Open an issue on [github.com/patriciomartinns/mcp-pdf-reader](https://github.com/patriciomartinns/mcp-pdf-reader).