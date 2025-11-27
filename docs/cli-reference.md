# `pdf-reader` CLI

The CLI exposes each MCP tool as a Typer subcommand and prints JSON-formatted output via Rich.

## Installation

```bash
uv tool install --from git+https://github.com/patriciomartinns/pdf-toolbox pdf-reader
```

> [!NOTE]
> If you still have the old `mcp-pdf-reader` CLI installed (`uv tool install mcp-pdf-reader`), remove it first with `uv tool uninstall mcp-pdf-reader` to prevent binary name conflicts.

Run without installing:

```bash
uvx --from git+https://github.com/patriciomartinns/pdf-toolbox -- pdf-reader --help
```

## Available Commands

| Command | Description | Key Parameters |
| --- | --- | --- |
| `read-pdf` | Extract paginated text. | `path`, `--start-page`, `--end-page`, `--max-pages` |
| `search-pdf` | Perform semantic search using embeddings. | `path`, `query`, `--top-k`, `--min-score`, `--chunk-size`, `--chunk-overlap` |
| `describe-pdf-sections` | List deterministic chunks with offsets. | `path`, `--max-chunks`, `--chunk-size`, `--chunk-overlap` |
| `configure-pdf-defaults` | Update default parameters for future calls. | `--chunk-size`, `--chunk-overlap`, `--max-pages`, `--embedding-model` |

## Examples

### Read specific pages

```bash
pdf-reader read-pdf learning/manual.pdf --start-page 3 --end-page 6 --max-pages 10
```

### Run semantic search

```bash
pdf-reader search-pdf learning/manual.pdf "rate limiting" --top-k 8 --min-score 0.2
```

### Describe chunks

```bash
pdf-reader describe-pdf-sections ~/Reports/Q4.pdf --max-chunks 5 --chunk-size 600 --chunk-overlap 120
```

### Adjust defaults

```bash
pdf-reader configure-pdf-defaults --chunk-size 600 --chunk-overlap 100 --max-pages 15
```

## Tips

- Paths can be relative; the server enforces `.pdf` extensions and optional sandboxing.
- Errors are displayed using Rich panels and return exit code 1.
- Responses are printed as JSON, making it easy to pipe into `jq` or other tools.

> [!TIP]
> The first `search-pdf` run on a new document can take longer because it downloads the SentenceTransformers model and builds the embedding index. Repeated calls reuse the cached model/index and become much faster.

> [!NOTE]
> When using the `pdf-reader` binary installed via `uv tool install`, PyMuPDF may emit `DeprecationWarning: builtin type SwigPy* has no __module__ attribute` on macOS. This comes from upstream SWIG builds (see [pymupdf/PyMuPDF#3931](https://github.com/pymupdf/PyMuPDF/issues/3931)) and is safe to ignore; Debian/Windows builds or our local server (which suppresses the warning) do not show it.

