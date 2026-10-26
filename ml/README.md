# fightml

ML research for fighting-game agents. **This package does not implement a game.**

Goals, sim contract, and next steps: [`HANDOFF.md`](HANDOFF.md).

It depends on a backend that matches [`fightml.protocol.FightingSimulator`](fightml/protocol.py):

```
reset / step / get_state / save_state / load_state / render
```

The default backend is `fightsim` (`../game`). Swap the backend by wrapping any other engine that implements the same protocol (including an Ikemen GO fallback later).

`fightml` must not be imported from `game/`. Rewards, Gymnasium wrappers, policies, and opponent pools live here — not in the simulator.

## Status

Phase 1 mix4: MaskablePPO vs personality dummies only (`--dummy-only`, 30e6 steps). Timeout win is +0.25 (weaker than a KO); indecisive clock-outs are −0.25; a small time cost and a range-gated whiff penalty apply. Eval plays the zip as **P2**. See [HANDOFF.md](HANDOFF.md). Next: PFSP win-rate sampling, then raw-input with command flags off.

Shipped generation zips live in **`ml/artifacts/`** (block, mix, mix2, mix3, mix4). New local trains, pool dumps, and TensorBoard logs go in **`ml/runs/`** (gitignored). CLI flags like `runs/ppo_mix4` always resolve under `ml/runs/` regardless of cwd; `artifacts/ppo_mix4` resolves to the shipped zip.

## Install

From the **repo root**, recreate the mix4 environment (Python 3.10 + pinned SB3/torch). See the root [`README.md`](../README.md) and [`requirements-lock.txt`](../requirements-lock.txt).

```bash
python3.10 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements-lock.txt
pip install -e "./game[play]" -e "./ml[dev]"
```

From `ml/` only, if the venv is already active:

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
python -m fightml.eval artifacts/ppo_mix4 --p1 zoner
python -m fightml.eval artifacts/ppo_mix4 --p1 defensive
python -m fightml.eval artifacts/ppo_mix4 --p1 human
python -m fightml.eval artifacts/ppo_mix4 --p1 human --box no
python -m fightml.eval artifacts/ppo_mix4 --p1 zoner --box maybe --native
python -m fightml.eval artifacts/ppo_mix4 --per-style --headless --episodes 10
python -m fightml.eval artifacts/ppo_mix3 --p1 zoner   # jump-stall, for comparison
python -m fightml.eval artifacts/ppo_mix2 --p1 mash
```

`--frameskip` must match training (default 4). `--headless` cannot be combined with `--p1 human`. Watch uses the same window as `fightsim.play`: `--box yes|no|maybe` (default `yes`; `no` is the play HUD, no skip=4 overlay), `--native` (locked 800×400), `--fullscreen` (desktop resolution, largest 0.25-step scale centered with letterbox; F11 toggles unless `--native`). Windowed watch starts at 800×400 and snaps on resize. Headless ignores window flags.

## Tests

```bash
pytest tests
```
