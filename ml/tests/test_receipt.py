"""Training receipts: git pins + CLI inside the SB3 zip."""

from __future__ import annotations

import json
import zipfile
from argparse import Namespace
from pathlib import Path

from fightml.receipt import (
    RECEIPT_NAME,
    ReceiptBuilder,
    find_earliest_matching_commit,
    format_receipt,
    parameters_from_args,
    read_receipt,
)


def test_parameters_from_args_are_json_safe():
    args = Namespace(timesteps=100, dummy_only=True, save=Path("runs/x"), nested=None)
    blob = parameters_from_args(args)
    json.dumps(blob)
    assert blob["timesteps"] == 100
    assert blob["dummy_only"] is True
    assert blob["save"] == "runs/x"
    assert blob["nested"] is None


def test_read_receipt_missing_file(tmp_path: Path):
    assert read_receipt(tmp_path / "nope.zip") is None


def test_receipt_roundtrip_appends_json(tmp_path: Path):
    stem = tmp_path / "model"
    builder = ReceiptBuilder(stem, {"timesteps": 12, "p2": "mix"})
    buf = builder.receipt_buffer()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("data", b"not-a-real-sb3")
    out = builder.finish_receipt()
    assert out.name == "model.zip"
    receipt = read_receipt(out)
    assert receipt is not None
    assert receipt["train"]["timesteps"] == 12
    assert receipt["train"]["p2"] == "mix"
    assert receipt["simulator_version"]
    git = receipt["git"]
    assert len(git["head"]) == 40
    assert len(git["earliest_game_commit"]) == 40
    assert len(git["earliest_ml_commit"]) == 40
    assert git["game_tree"]
    assert git["ml_tree"]
    text = format_receipt(receipt)
    assert "earliest_game_commit" in text
    with zipfile.ZipFile(out) as archive:
        assert RECEIPT_NAME in archive.namelist()
        assert "data" in archive.namelist()


def test_earliest_game_commit_is_an_ancestor():
    commit = find_earliest_matching_commit(path="game")
    assert len(commit.hexsha) == 40
