"""Canonical artifact directory: <ml package>/runs, independent of cwd."""

from __future__ import annotations

from pathlib import Path


def ml_root() -> Path:
    return Path(__file__).resolve().parents[1]


def runs_root() -> Path:
    return ml_root() / "runs"


def resolve_runs_path(p: str | Path) -> Path:
    """Map relative CLI paths like runs/ppo_mix3 onto ml/runs/."""
    path = Path(p)
    if path.is_absolute():
        return path
    root = runs_root()
    parts = path.parts
    if parts and parts[0] == "runs":
        return root.joinpath(*parts[1:]) if len(parts) > 1 else root
    return root / path


def resolve_artifact(p: str | Path) -> Path:
    """Like resolve_runs_path, but prefer a path that already exists (cwd or ml/runs)."""
    path = Path(p)
    if path.is_absolute():
        return path
    for candidate in (path, path.with_suffix(".zip")):
        if candidate.exists():
            return candidate
    resolved = resolve_runs_path(path)
    if resolved.exists() or resolved.with_suffix(".zip").exists():
        return resolved if resolved.exists() else resolved.with_suffix(".zip")
    alt = ml_root() / path
    if alt.exists() or alt.with_suffix(".zip").exists():
        return alt if alt.exists() else alt.with_suffix(".zip")
    return resolved
