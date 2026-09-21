# fightml

ML research for fighting-game agents. **This package does not implement a game.**

**Catch-up for the research owner:** read [`HANDOFF.md`](HANDOFF.md) first (goals, API, milestones, approaches, what not to do).

It depends on a backend that matches [`fightml.protocol.FightingSimulator`](fightml/protocol.py):

```
reset / step / get_state / save_state / load_state / render
```

The default backend is `fightsim` (`../game`). Swap the backend by wrapping any other engine that implements the same protocol (including an Ikemen GO fallback later).

`fightml` must not be imported from `game/`. Rewards, Gymnasium wrappers, policies, and opponent pools live here — not in the simulator.

## Status

Phase 1 mix3: MaskablePPO vs 50/50 personality dummies and frozen `ml/runs/pool/` zips. Timeout needs a 2K health lead to score; cheap clock-outs are 0. Whiff penalty. Eval plays the zip as **P2**. See [HANDOFF.md](HANDOFF.md). Next: PFSP win-rate sampling, then raw-input with command flags off.

Checkpoints, pool zips, and TensorBoard logs live only in **`ml/runs/`**. CLI flags still look like `runs/ppo_mix3`; they always resolve there, even if you launch from the repo root.

## Install

```bash
pip install -e "../game" -e ".[dev]"
```

## Train

```bash
python -m fightml.train
python -m fightml.train --save runs/ppo_mix3 --n-envs 8 --timesteps 1000000
python -m fightml.train --dummy-only --p2 zoner --save runs/ppo_zoner --no-eval-after
```

Default P2 is a mix of personalities and, when `ml/runs/pool/` (or `ml/runs/ppo_mix2.zip`) exists, frozen copies (`--opponent-pool-frac 0.5`). `--timeout-lead 2000`. Training randomizes side and gap. Checkpoints go to `ml/runs/pool/`. `--eval-after` prints a per-style table (agent as P2).

TensorBoard logging is off unless you pass `--tb runs/tb`. Training does not import pygame.

## Eval (watch / human / dummies)

The agent is always **P2**. `--p1` selects a dummy or a human. Watch needs pygame (`pip install -e "../game[play]"`). Human P1 uses WASD + J/K/L (same as `fightsim.play`). Headless `--per-style` scores each personality as P1, both orientations.

```bash
python -m fightml.eval runs/ppo_mix3 --p1 zoner
python -m fightml.eval runs/ppo_mix3 --p1 defensive
python -m fightml.eval runs/ppo_mix3 --p1 human
python -m fightml.eval runs/ppo_mix3 --per-style --headless --episodes 10
```

`--frameskip` must match training (default 4). `--headless` cannot be combined with `--p1 human`.

## Tests

```bash
pytest tests
```
