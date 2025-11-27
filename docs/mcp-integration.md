# MCP server `pdf-toolbox`

The server exposes the same tools over STDIO transport—ideal for Cursor, VS Code, Claude, and any MCP-compatible agent.

## Running the server

```bash
uvx --from git+https://github.com/patriciomartinns/pdf-toolbox -- pdf-toolbox --quiet
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
        "git+https://github.com/patriciomartinns/pdf-toolbox",
        "--",
        "pdf-toolbox",
        "--quiet"
      ]
    }
  }
}
```

### Claude Desktop / Claude Code

1. Open **Settings → MCP Integrations**.  
2. Add a server with: `uvx --from git+https://github.com/patriciomartinns/pdf-toolbox -- pdf-toolbox`.  
3. Append `--quiet` if you prefer silent startups.

### CLI agents (Cline, OpenCode)

```json
{
  "name": "PdfReader",
  "command": "uvx",
  "args": [
    "--from",
    "git+https://github.com/patriciomartinns/pdf-toolbox",
    "--",
    "pdf-toolbox"
  ]
}
```

## Operational tips

- **Safe base path**: by default the server restricts access to the current directory. Use the tools `configure_pdf_defaults` and `set_base_path` to adjust sandboxing and chunking defaults.  
- **Embedding models**: provide the fully qualified SentenceTransformer name in `configure_pdf_defaults --embedding-model ...`.  

> [!TIP]
> The first semantic call (`search_pdf`) loads the SentenceTransformers model and builds embeddings for the PDF. Expect a longer delay once per model/PDF combo; subsequent searches reuse the cache and return much faster (the CLI and MCP server share the same behavior).

> [!NOTE]
> On certain macOS builds the underlying PDF backend can emit `DeprecationWarning: builtin type SwigPy* has no __module__ attribute`. The warning is benign, and the server suppresses it so clients receive clean logs.
