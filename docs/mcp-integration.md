# MCP server `mcp-pdf-reader`

The server exposes the same tools over STDIO transport—ideal for Cursor, VS Code, Claude, and any MCP-compatible agent.

## Running the server

```bash
uvx --from git+https://github.com/patriciomartinns/mcp-pdf-reader -- mcp-pdf-reader --quiet
```

> Remove `--quiet` if you want to see the startup banner. Only STDIO transport is supported.

## Client configuration

### Cursor / VS Code (`.cursor/mcp.json`)

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

### Claude Desktop / Claude Code

1. Open **Settings → MCP Integrations**.  
2. Add a server with: `uvx --from git+https://github.com/patriciomartinns/mcp-pdf-reader -- mcp-pdf-reader`.  
3. Append `--quiet` if you prefer silent startups.

### CLI agents (Serena, Cline, OpenHands)

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

## Operational tips

- **Safe base path**: by default the server restricts access to the current directory. Use the tools `configure_pdf_defaults` and `set_base_path` to adjust sandboxing and chunking defaults.  
- **Embedding models**: provide the fully qualified SentenceTransformer name in `configure_pdf_defaults --embedding-model ...`.  

> [!TIP]
> The first semantic call (`search_pdf`) loads the SentenceTransformers model and builds embeddings for the PDF. Expect a longer delay once per model/PDF combo; subsequent searches reuse the cache and return much faster (the CLI and MCP server share the same behavior).

> [!NOTE]
> When running the MCP server through official PyMuPDF wheels on macOS, you might see `DeprecationWarning: builtin type SwigPy* has no __module__ attribute`. This stems from upstream SWIG builds (see [pymupdf/PyMuPDF#3931](https://github.com/pymupdf/PyMuPDF/issues/3931)) and is safe to ignore. Our local server suppresses it, and future PyMuPDF releases linked against SWIG ≥ 4.4.0 will remove the warning entirely.

