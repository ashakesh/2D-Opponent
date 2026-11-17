"""Training receipts: bundle a saved model with its git/config provenance.

A receipt is a zip file containing:
 - the exact bytes stable_baselines3 writes via model.save()
 - receipt.json: git commit hashes, SIMULATOR_VERSION, and training CLI

Usage::

    receipt = ReceiptBuilder(path, parameters)
    model.save(receipt.receipt_buffer())
    receipt.finish_receipt()

The git helpers find how far back in history the ``game/`` or ``ml/`` subtrees
have been unchanged relative to HEAD, so a receipt can record the earliest
commit that still reflects the code that produced a model. Working tree / index
state (uncommitted changes) is intentionally ignored for hashes — only committed
trees are inspected. A dirty tree prints a warning.
"""

from __future__ import annotations

import io
import json
import posixpath
import sys
import zipfile
from pathlib import Path
from typing import Any

from git import Commit, InvalidGitRepositoryError, NoSuchPathError, Repo

RECEIPT_NAME = "receipt.json"


def get_repo() -> Repo:
    """Return the Repo containing this file, regardless of cwd.

    Raises RuntimeError if this file is not inside a git checkout (for example
    fightml installed into site-packages without the surrounding .git).
    """
    here = Path(__file__).resolve()
    try:
        return Repo(here, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        raise RuntimeError(
            f"'{here}' is not inside a git checkout, so git provenance "
            "(commit hashes, receipts) is unavailable. Train from a full clone."
        ) from exc


def _normalize_path(path: str) -> str:
    """Repo-root-relative path for GitPython tree indexing.

    ``./game``, ``game/``, and ``game`` are equivalent. ``.`` means the whole tree.
    """
    normalized = posixpath.normpath(path)
    return "" if normalized == "." else normalized


def _subtree_hexsha(commit: Commit, path: str) -> str | None:
    """Hash of the subtree at ``path`` in ``commit``, or None if missing."""
    if path == "":
        return commit.tree.hexsha
    try:
        return commit.tree[path].hexsha
    except KeyError:
        return None


def find_earliest_matching_commit(
    repo: Repo | None = None, path: str = "./", ref: str = "HEAD"
) -> Commit:
    """Earliest ancestor of ``ref`` whose ``path`` subtree matches ``ref``.

    Walks newest → oldest. Stops at the first commit where the subtree differs
    and returns the last matching commit (oldest in the unbroken run).
    """
    if repo is None:
        repo = get_repo()
    path = _normalize_path(path)

    start = repo.commit(ref)
    target_hash = _subtree_hexsha(start, path)
    if target_hash is None:
        raise ValueError(f"path '{path}' does not exist at {ref!r}")

    earliest = start
    for commit in repo.iter_commits(start):
        if _subtree_hexsha(commit, path) != target_hash:
            break
        earliest = commit
    return earliest


def json_safe(value: Any) -> Any:
    """JSON-serialize argparse values (Path → str; drop non-scalars)."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def parameters_from_args(args: Any) -> dict[str, Any]:
    """Curated JSON-safe dict from an argparse namespace (or mapping)."""
    raw = vars(args) if hasattr(args, "__dict__") and not isinstance(args, dict) else dict(args)
    return {str(k): json_safe(v) for k, v in raw.items()}


def _simulator_version() -> str | None:
    try:
        from fightsim.config import SIMULATOR_VERSION

        return str(SIMULATOR_VERSION)
    except Exception:
        return None


def read_receipt(path: str | Path) -> dict[str, Any] | None:
    """Load receipt.json from an SB3 zip, or None if missing / not a zip."""
    zip_path = Path(path)
    if zip_path.suffix != ".zip":
        candidate = zip_path.with_suffix(zip_path.suffix + ".zip")
        zip_path = candidate if candidate.exists() else zip_path
    if not zip_path.is_file():
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            if RECEIPT_NAME not in archive.namelist():
                return None
            return json.loads(archive.read(RECEIPT_NAME).decode("utf-8"))
    except (zipfile.BadZipFile, json.JSONDecodeError, OSError):
        return None


def format_receipt(receipt: dict[str, Any]) -> str:
    """Short stdout block for eval / train."""
    return json.dumps(receipt, indent=2, sort_keys=True)


class ReceiptBuilder:
    """SB3 model zip plus a receipt.json entry.

    Still loadable with model.load(...) / load_predict(...). SB3 only reads
    known zip names (data, *.pth, system_info.txt) and ignores the rest.
    """

    def __init__(self, path: str | Path, parameters: dict[str, Any] | None = None) -> None:
        path = Path(path)
        self._path = path if path.suffix == ".zip" else path.with_suffix(path.suffix + ".zip")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "wb+")
        self._finished = False

        repo = get_repo()
        self._warn_if_dirty(repo)
        self._receipt_json = json.dumps(
            self._build_metadata(repo, parameters or {}), indent=2
        )

    def _warn_if_dirty(self, repo: Repo) -> None:
        if repo.is_dirty(untracked_files=True):
            print(
                "WARNING: uncommitted changes in the repo — this receipt's "
                "git hashes will not reflect the code actually used to train. "
                "Commit first if you want the receipt to be accurate.",
                file=sys.stderr,
            )

    def _build_metadata(self, repo: Repo, parameters: dict[str, Any]) -> dict[str, Any]:
        head = repo.head.commit
        payload: dict[str, Any] = {
            "git": {
                "head": head.hexsha,
                "game_tree": _subtree_hexsha(head, "game"),
                "ml_tree": _subtree_hexsha(head, "ml"),
                "earliest_game_commit": find_earliest_matching_commit(
                    repo, "game", head
                ).hexsha,
                "earliest_ml_commit": find_earliest_matching_commit(
                    repo, "ml", head
                ).hexsha,
            },
            "train": json_safe(parameters),
        }
        sim = _simulator_version()
        if sim is not None:
            payload["simulator_version"] = sim
        return payload

    def receipt_buffer(self) -> io.BufferedIOBase:
        """Buffer to pass straight to model.save(...). Do not close it."""
        if self._finished:
            raise RuntimeError("receipt already finished; cannot reuse its buffer")
        return self._file

    def finish_receipt(self) -> Path:
        """Append receipt.json to the zip model.save() just wrote."""
        if self._finished:
            raise RuntimeError("finish_receipt() already called for this receipt")
        if self._file.tell() == 0:
            raise RuntimeError(
                "nothing was written to receipt_buffer(); call "
                "model.save(receipt.receipt_buffer()) before finish_receipt()"
            )

        with zipfile.ZipFile(self._file, "a") as archive:
            archive.writestr(RECEIPT_NAME, self._receipt_json)
        self._file.close()

        self._finished = True
        return self._path
