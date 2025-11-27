# Advanced Options

This guide covers features that go beyond the basic read / search / chunk workflows exposed by the PDF Toolbox CLI and MCP server.

## Table mode (`describe_pdf_sections`)

The `describe_pdf_sections` tool can operate in two modes:

- `mode="chunks"` (default) — returns deterministic text chunks with offsets and the embedding model metadata.
- `mode="tables"` — switches to the built-in table detector, returning structural information instead of free-form chunks.

### When to use table mode

- The PDF has clearly delineated tables (grid lines or consistent column spacing).
- You need bounding boxes, inferred headers, and per-cell text to feed downstream tooling (e.g., DataFrames, spreadsheet completion, or RAG pipelines that reason over cells).

### Response overview

Each table entry includes:

- `bbox` (x0, y0, x1, y1) in page coordinates.
- `headers` — inferred column headers when the detector recognizes them.
- `rows[].cells[]` — row-major cells; each cell exposes its column index, bounding box, and trimmed text.

### Limitations

- Tables without visual guides (e.g., text-aligned columns) or highly irregular layouts may not be detected.
- Multi-line or nested headers sometimes need manual cleanup.
- When detection quality drops (e.g., for irregular layouts), tweak the available detector strategies before resorting to OCR-based approaches.

### CLI example

```bash
pdf-reader describe-pdf-sections my-report.pdf --mode tables --max-chunks 5
```

### MCP example

```jsonc
{
  "tool": "describe_pdf_sections",
  "args": {
    "path": "reports/financials.pdf",
    "mode": "tables",
    "max_chunks": 10
  }
}
```
