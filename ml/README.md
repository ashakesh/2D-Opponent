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

Phase 1 mix4: MaskablePPO vs personality dummies only (`--dummy-only`, 30e6 steps). Timeout win is +0.25 (weaker than a KO); indecisive clock-outs are −0.25; a small time cost and a range-gated whiff penalty apply. Eval plays the zip as **P2**. See [HANDOFF.md](HANDOFF.md). Next: PFSP win-rate sampling, then raw-input with command flags off.

Checkpoints, pool zips, and TensorBoard logs live only in **`ml/runs/`**. CLI flags still look like `runs/ppo_mix4`; they always resolve there, even if you launch from the repo root.

## Install

```bash
pip install -e "../game" -e ".[dev]"
```

## Train

```bash
python -m fightml.train --dummy-only
python -m fightml.train --dummy-only --save runs/ppo_mix4 --pool runs/pool4 --n-envs 8 --timesteps 30000000
python -m fightml.train --dummy-only --p2 zoner --save runs/ppo_zoner --no-eval-after
```

Default P2 is a mix of personalities. For mix4, pass **`--dummy-only`** so `ml/runs/pool4/` checkpoints are written but not sampled as P2 this generation. `--timeout-lead 2000`, timeout win +0.25, indecisive timeout −0.25. Training randomizes side and gap. `--eval-after` prints a per-style table (agent as P2).

TensorBoard logging is off unless you pass `--tb runs/tb`. Training does not import pygame.

## Eval (watch / human / dummies)

The agent is always **P2**. `--p1` selects a dummy or a human. Watch needs pygame (`pip install -e "../game[play]"`). Human P1 uses WASD + J/K/L (same as `fightsim.play`). Headless `--per-style` scores each personality as P1, both orientations.

```bash
python -m fightml.eval runs/ppo_mix4 --p1 zoner
python -m fightml.eval runs/ppo_mix4 --p1 defensive
python -m fightml.eval runs/ppo_mix4 --p1 human
python -m fightml.eval runs/ppo_mix4 --per-style --headless --episodes 10
```

`--frameskip` must match training (default 4). `--headless` cannot be combined with `--p1 human`.

## Tests

```bash
pytest tests
```
