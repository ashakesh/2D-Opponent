# 2D-Opponent

A short-term ML research project: an agent that learns adaptive fighting-game play.

The **game is not the product**. This repo is split so either half can be taken independently.

```
game/    standalone deterministic fighting simulator (`fightsim`)
ml/      research code (`fightml`); depends on a simulator backend, not on pygame
```

**Dependency rule:** `game` never imports `ml`. Research wraps the simulator (or any other backend that matches the same API).

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

Play the game by itself:

```bash
pip install -e "./game[play]"
python -m fightsim.play
```

Research (later milestones):

```bash
pip install -e "./game" -e "./ml"
```

ML research owners: start at [`ml/HANDOFF.md`](ml/HANDOFF.md). Do not add training code to `game/`. Train/eval checkpoints live in [`ml/runs/`](ml/runs/) — not a `runs/` folder at the repo root.

## Tests

```bash
pip install -e "./game[dev]"
pytest game/tests
```
