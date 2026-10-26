# 2D-Opponent

A short-term ML research project: an agent that learns adaptive fighting-game play.

The **game is not the product**. This repo is split so either half can be taken independently.

```
game/          standalone deterministic fighting simulator (`fightsim`)
ml/            research code (`fightml`); depends on a simulator backend, not on pygame
ml/artifacts/  pretrained generation zips (block → mix4)
```

**Dependency rule:** `game` never imports `ml`. Research wraps the simulator (or any other backend that matches the same API).

**Status:** Phase 1 (competent play vs scripted dummies) is in the repo. The current policy is **mix4**. Watch it, or fight it, without retraining. Next research step is self-play / PFSP, then raw-input learning. Details: [`ml/HANDOFF.md`](ml/HANDOFF.md).

## Simulator contract

```
reset(seed) → GameState
step(p1_input, p2_input) → GameState    # exactly one frame
get_state()
save_state() → bytes
load_state(snapshot)
render()                               # optional; not required to simulate
```

The sim speaks **raw controller inputs** only. Semantic actions (LIGHT, SPECIAL_1, …) are an adapter in `fightsim.input`.

## Install

Use **Python 3.10** (we trained on 3.10.11). Do not copy `venv/` from another machine.

Eval and train of the shipped zips need the pinned stack in [`requirements-lock.txt`](requirements-lock.txt) (SB3 2.9.0). A looser `pip install` can pull a newer SB3 that will not load mix4.

```bash
python3.10 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements-lock.txt
pip install -e "./game[play]" -e "./ml[dev]"
```

Play the game by itself (lockfile optional):

```bash
pip install -e "./game[play]"
python -m fightsim.play
```

Watch or fight a shipped agent (after the venv steps above):

```bash
cd ml
python -m fightml.eval artifacts/ppo_mix4 --p1 human
python -m fightml.eval artifacts/ppo_mix3 --p1 zoner   # jump-stall, for comparison
```

Do not add training code to `game/`. New train/eval checkpoints go in `ml/runs/` (gitignored) — not a `runs/` folder at the repo root. The shipped lineage lives in [`ml/artifacts/`](ml/artifacts/). Package docs: [`game/README.md`](game/README.md), [`ml/README.md`](ml/README.md).

## Tests

```bash
pip install -e "./game[dev]"
pytest game/tests
```
