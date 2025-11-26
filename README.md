## MCP PDF Reader

Expose local PDFs to LLM agents with direct reading, deterministic chunking, and semantic retrieval over MCP.

### Quick Start

Always launch the MCP via `uvx` pointing to the public repository—no local setup required:

```bash
uvx --from git+https://github.com/patriciomartinns/pdf-reader mcp-pdf-reader -- --quiet
# inspect options
uvx --from git+https://github.com/patriciomartinns/pdf-reader mcp-pdf-reader -- --help
```

`--quiet` only hides the startup banner; the transport stays STDIO.

### Configuring Your Client

All MCP clients should call the exact command above. The snippets below show how to embed it in different environments.

#### Cursor / VS Code (MCP Settings)

Add an entry to `.cursor/mcp.json` (or VS Code’s MCP configuration):

```json
{
  "name": "mcp-pdf",
  "transport": {
    "stdio": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/patriciomartinns/pdf-reader",
        "mcp-pdf-reader"
      ]
    }
  }
}
```

#### Claude Desktop / Claude Code

1. Open the MCP integrations panel.
2. Add a new server pointing to `uvx --from git+https://github.com/patriciomartinns/pdf-reader mcp-pdf-reader`.
3. (Optional) Append `--quiet` to reduce console noise when Claude starts the agent.

#### CLI agents (Serena, Cline, OpenHands, etc.)

Reference the same command when defining a server entry. Example for Serena’s `mcp.json`:

```json
{
  "name": "mcp-pdf",
  "command": "uvx",
  "args": [
    "--from",
    "git+https://github.com/patriciomartinns/pdf-reader",
    "mcp-pdf-reader"
  ]
}
```

Any CLI agent that supports STDIO MCP servers can reuse this snippet verbatim.

### Technologies

- FastMCP for the MCP server runtime
- PyMuPDF (`pymupdf`) for PDF parsing
- SentenceTransformers for semantic embeddings
- Pydantic for tool payload schemas
- `uv` for packaging and execution via `uvx`

