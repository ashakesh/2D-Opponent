"""ml/runs is the only artifact directory, independent of process cwd."""

from pathlib import Path

from fightml.paths import ml_root, resolve_artifact, resolve_runs_path, runs_root


def test_relative_runs_maps_under_ml():
    assert ml_root().name == "ml"
    assert runs_root() == ml_root() / "runs"
    assert resolve_runs_path("runs/ppo_mix3") == runs_root() / "ppo_mix3"
    assert resolve_runs_path("runs/pool") == runs_root() / "pool"
    assert resolve_runs_path("ppo_zoner") == runs_root() / "ppo_zoner"
    abs_p = Path("/tmp/ckpt")
    assert resolve_runs_path(abs_p) == abs_p


def test_resolve_artifact_prefers_ml_runs_when_cwd_has_no_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_artifact("runs/pool")
    assert resolved == runs_root() / "pool"
