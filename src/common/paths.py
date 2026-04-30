"""Path helpers for running tests/scripts from Windows, IDEs, or project root."""
from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when it is not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p
