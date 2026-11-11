"""Training receipts: bundle a saved model with its git/config provenance.

A receipt is a zip file containing:
  - model.zip     the exact bytes stable_baselines3 writes via model.save()
  - receipt.json  git commit hashes + training parameters

Usage:
    receipt = ReceiptBuilder(path, parameters)
    model.save(receipt.recieptBuffer())
    receipt.finishReceipt()

The git provenance helpers below find how far back in history the
`game/` or `ml/` subtrees have been unchanged relative to HEAD, so a
receipt can record the earliest commit that still reflects the exact
code that produced a model. Working tree / index state (uncommitted
changes) is intentionally ignored - only committed trees are inspected.
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


def get_repo() -> Repo:
    """Return the Repo containing this file, regardless of cwd.

    Raises a RuntimeError if this file isn't inside a git checkout at all,
    which happens when fightml is installed into a venv (e.g.
    site-packages) and run without the surrounding .git directory - in
    that case git provenance (commit hashes, receipts) is unavailable.
    """
    # Anchor on this file's own location rather than os.getcwd(), so callers
    # get the right Repo no matter where within the checkout a script was
    # launched from.
    here = Path(__file__).resolve()
    try:
        return Repo(here, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        # Re-raise as a clear, actionable error instead of letting
        # GitPython's cryptic discovery error propagate.
        raise RuntimeError(
            f"'{here}' is not inside a git checkout, so git provenance "
            "(commit hashes, receipts) is unavailable here. This happens "
            "when fightml is installed/run outside of a full clone of the "
            "repo."
        ) from exc


def _normalize_path(path: str) -> str:
    """Normalize a repo-root-relative path spec to the plain form
    GitPython's tree indexing expects.

    Repo-root-relative paths are always used (never resolved against cwd),
    so "game" means the same thing no matter where a script is run from.
    Leading "./", trailing "/", and "." (whole tree) are all normalized,
    so "./game", "game/", and "game" are equivalent.
    """
    normalized = posixpath.normpath(path)
    return "" if normalized == "." else normalized


def _subtree_hexsha(commit: Commit, path: str) -> str | None:
    """Hash of the subtree at `path` in `commit`, or None if it doesn't exist."""
    if path == "":
        return commit.tree.hexsha
    try:
        return commit.tree[path].hexsha
    except KeyError:
        return None


def find_earliest_matching_commit(
    repo: Repo | None = None, path: str = "./", ref: str = "HEAD"
) -> Commit:
    """Earliest ancestor of `ref` whose `path` subtree matches `ref`'s.
      Set `path` to "./game" for the earliest same game commit.
      And similarly `path` to "./ml" for ml commit.
    """

    """
    Walks back through `ref`'s ancestry (newest to oldest) comparing the
    git tree hash of `path` at each commit. Stops at the first commit where
    the subtree differs, and returns the last commit before that - i.e. the
    oldest commit in the unbroken run of history where `path` has exactly
    the same files and content as it does at `ref`.

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


class ReceiptBuilder:
    """Builds a receipt: an SB3 model zip with a receipt.json entry added.

    The output is a single, ordinary SB3 save file - still loadable with
    model.load(...)/load_predict(...) exactly as before - with one extra
    "receipt.json" entry recording git provenance and training config.
    SB3's loader only ever reads specific known entries off the zip's
    namelist (data, *.pth, system_info.txt) and ignores the rest, so the
    extra entry is invisible to it.
    """

    def __init__(self, path: str | Path, parameters: dict[str, Any]) -> None:
        path = Path(path)
        # Mirror SB3's own save() convention of appending ".zip" when the
        # caller's path doesn't already end in it.
        self._path = path if path.suffix == ".zip" else path.with_suffix(path.suffix + ".zip")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Opened "wb+" (read+write) rather than "wb", so the same handle
        # that model.save() wrote the zip into can be reopened for append
        # afterwards - no temp files, no rewriting the model bytes.
        self._file = open(self._path, "wb+")
        self._finished = False

        repo = get_repo()
        self._warn_if_dirty(repo)
        # Built now rather than in finishReceipt(), so the recorded git
        # state reflects when the receipt was created, not whenever
        # training happens to finish.
        self._receipt_json = json.dumps(self._build_metadata(repo, parameters), indent=2)

    def _warn_if_dirty(self, repo: Repo) -> None:
        if repo.is_dirty(untracked_files=True):
            print(
                "WARNING: uncommitted changes in the repo - this receipt's "
                "git hashes will not reflect the code actually used to "
                "train. Run `git commit` (saves your currently staged "
                "changes as a new point in the repo's history) first if "
                "you want the receipt to be accurate.",
                file=sys.stderr,
            )

    def _build_metadata(
        self, repo: Repo, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        # TODO: write `parameters` into the receipt once there's a
        # reliable, reproducible source for it.
        head = repo.head.commit
        return {
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
        }

    def recieptBuffer(self) -> io.BufferedIOBase:
        """Buffer to pass straight to model.save(...).

        Returns the same open file handle every call; it is owned by this
        builder and must not be closed by the caller.
        """
        if self._finished:
            raise RuntimeError("receipt already finished; cannot reuse its buffer")
        return self._file

    def finishReceipt(self) -> Path:
        """Append receipt.json to the model zip model.save() just wrote."""
        if self._finished:
            raise RuntimeError("finishReceipt() already called for this receipt")
        if self._file.tell() == 0:
            raise RuntimeError(
                "nothing was written to recieptBuffer(); call "
                "model.save(receipt.recieptBuffer()) before finishReceipt()"
            )

        # mode "a" appends a new entry to the zip model.save() already
        # wrote into this same handle, instead of rebuilding the archive.
        with zipfile.ZipFile(self._file, "a") as archive:
            archive.writestr("receipt.json", self._receipt_json)
        self._file.close()

        self._finished = True
        return self._path
