# Contributing to PDF Toolbox

Thanks for taking the time to contribute! This project exposes local PDFs to MCP-compatible agents. Follow the guidelines below to keep the workflow smooth and reproducible.

## Getting Started

1. **Fork & Clone**
   ```bash
   git clone https://github.com/<your-user>/pdf-toolbox.git
   cd pdf-toolbox
   ```
2. **Install uv** (if you haven’t already): <https://github.com/astral-sh/uv>
3. **Sync dependencies**
   ```bash
   uv sync
   ```

## Development Workflow

1. Create a feature branch: `git checkout -b feat/<short-description>`
2. Make changes using `uv run <cmd>` to ensure the synced virtualenv is used.
3. Follow the [Conventional Commits](https://www.conventionalcommits.org/) spec (`feat:`, `fix:`, `docs:`, etc.) and keep each commit focused.

### Tests & Quality Gates

Before opening a PR:

```bash
uv run pytest            # unit tests
uv run ruff check .      # lint (E/F/I + import sorting)
uv run pyright           # strict type checking (py314 target)
```

Optional but helpful:

```bash
uv run ruff format       # enforce repository formatting
```

## Project Structure

```
src/pdf_toolbox/         # main package
tests/                   # pytest suite
learning/                # sample PDFs (ignored by VCS)
README.md                # usage guide
```

## Pull Request Guidelines

- Explain the motivation, behavior changes, and testing performed.
- Reference issues when applicable (`Fixes #123`).
- Keep screenshots/logs concise if relevant.
- Ensure the README/docs are updated whenever behavior changes.

## Code Style

- Python 3.13 features are welcome (match `pyproject.toml`).
- Use type hints everywhere; Pyright runs in strict mode.
- Avoid committing generated files or large binaries.

## Security & Privacy

- PDFs in `learning/` are local examples and should not include confidential data.
- Never log or commit sensitive paths from users—keep instrumentation minimal.

## Questions?

Open a discussion or issue on [GitHub](https://github.com/patriciomartinns/pdf-toolbox). Thanks again for contributing! 🎉

