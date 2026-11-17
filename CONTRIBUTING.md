# Collaborating on 2D-Opponent

The repo is one tree with two **owned paths**. Training and eval need both `game/` and `ml/` in the **same checkout**. Do not keep long-lived `Game` and `ML` branches.

**Current published agent:** [`ml/artifacts/ppo_mix4.yaml`](ml/artifacts/ppo_mix4.yaml). Registry: [`ml/artifacts/README.md`](ml/artifacts/README.md).

## Daily workflow

`main` is the source of truth. Collaborators **do not push to `main`**.

1. `git pull origin main` and start a short-lived branch:
   - `game/<topic>` — only [`game/`](game/) (and sim docs).
   - `ml/<topic>` — only [`ml/`](ml/) (and train/eval docs).
   - `docs/<topic>` or `chore/<topic>` — README, lockfile, this file, CI.
2. Open a **Pull Request into `main`**. A teammate reviews (game PR: someone who cares about the meta; ML PR: someone who trains).
3. Merge (squash is fine). Delete the feature branch.

### Path rule

- A `game/` PR must not change `ml/fightml/` or `ml/artifacts/`.
- An `ml/` PR must not change `fightsim` physics, moves, dummies, or `SIMULATOR_VERSION`.
- If a change needs both (new dummy + env wiring): **game first**, merge, then ML. A rare `chore/` PR that touches both must say so in the description.

One clone, one branch, PR to `main`. Worktrees are optional for side-by-side old metas, not the default.

## Versioning the game (the meta)

[`SIMULATOR_VERSION`](game/fightsim/config.py) must stay in sync with [`game/pyproject.toml`](game/pyproject.toml). Replays store it. Log every bump in [`game/CHANGELOG.md`](game/CHANGELOG.md).

| Change | Bump |
|---|---|
| Balance, frames, hitboxes, health, dummy AI, new move | patch or minor (`0.4.1` → `0.5.0` if old agents' world is wrong) |
| Comments, pygame window, HUD that does not affect `step()` | no bump (or patch if you want a play tag) |
| Save/load or `GameState` layout | minor/major — old zips and replays are invalid |

After the version bump is on `main`:

```bash
git tag -a game-v0.5.0 -m "sim 0.5.0: <one-line meta change>"
git push origin game-v0.5.0
```

Play that meta: `git switch --detach game-v0.5.0` then `python -m fightsim.play`.

## Versioning agents

Two layers:

```
train   → ml/runs/foo.zip            # gitignored; receipt.json inside the zip
publish → ml/artifacts/<id>.zip      # copy the zip (keep the receipt)
        + ml/artifacts/<id>.yaml     # human card; paste git pins from the receipt
```

**Receipt** (automatic): `python -m fightml.train` writes `receipt.json` into the checkpoint zip (`head`, `game/` / `ml/` tree hashes, earliest matching commits, CLI flags, `SIMULATOR_VERSION`). SB3 still loads the zip. Dump it with `unzip -p path/to.zip receipt.json` or `python -m fightml.eval <zip> --headless` (prints the receipt if present). Do not train on a dirty tree if you need those hashes to be true.

**Card** (human): schema and index in [`ml/artifacts/README.md`](ml/artifacts/README.md). One yaml per published zip — not a single “latest” file. Copy `git.head`, `earliest_game_commit`, and `earliest_ml_commit` from the receipt. Add compat, opponents, notes.

Local pool dumps stay in `ml/runs/` (gitignored). Do not PR 30e6-step checkpoints.

Optional tag on the publish commit: `agent/<id>`.

## Playing and comparing

Same meta, different agents (both cards must list the same `compat.sim` / `compat.obs`):

```bash
python -m fightml.eval artifacts/ppo_mix4 --p1 zoner --headless --per-style
python -m fightml.eval artifacts/ppo_mix3 --p1 zoner --headless --per-style
```

Old meta + old agent: `git switch --detach` the card’s game tag or `git.head` / `earliest_game_commit` from the receipt, recreate the venv from that commit’s `requirements-lock.txt`, then eval the zip.

New meta + old agent: only if the card’s `compat.sim` includes the new `SIMULATOR_VERSION`. If you bumped the sim for combat, **retrain**; do not treat old evals as evidence.

## What to type

| Intent | Branch / command |
|---|---|
| Change hitboxes | `game/nerf-fireball` → PR → bump sim → tag `game-v0.5.0` |
| Change reward / train | `ml/timeout-ko-only` → PR; do not touch `game/` |
| Share a new bot | copy receipt-zip to `ml/artifacts/`, add yaml from `receipt.json`, PR `ml/publish-mix5` |
| Fight current bot | `main` + `python -m fightml.eval artifacts/ppo_mix4 --p1 human` |
| Fight that bot on that patch | checkout tag/hash from its yaml, then same eval |
