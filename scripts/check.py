from __future__ import annotations

import subprocess  # nosec B404
from typing import Sequence

_CHECK_COMMANDS: Sequence[Sequence[str]] = (
    ("ruff", "check", "."),
    ("pyright",),
    ("bandit", "-r", "src"),
    ("pytest",),
)


def run_checks() -> None:
    for command in _CHECK_COMMANDS:
        print(f"→ {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)  # nosec B603
