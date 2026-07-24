"""Shared configuration for the data-collection scripts."""

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(
    os.environ.get("HUMAN_ACTION_ROOT", str(REPOSITORY_ROOT))
).expanduser()


def project_path(*parts: str) -> Path:
    """Return a path relative to the configured project root."""
    return PROJECT_ROOT.joinpath(*parts)


def require_env(name: str) -> str:
    """Return a required environment variable with a helpful error."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is required. "
            "Copy .env.example, set the value, and export it before running."
        )
    return value
