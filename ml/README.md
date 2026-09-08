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

Phase 1 baseline: Gymnasium env + PPO vs a dummy (`env.py`, `train.py`), plus `eval.py` to watch or fight a checkpoint. See [HANDOFF.md](HANDOFF.md).

## Install

```bash
pip install -e "../game" -e ".[dev]"
```

## Train

```bash
python -m fightml.train --p2 block --timesteps 200000 --save runs/ppo_block
python -m fightml.train --p2 mash --seed 1 --save runs/ppo_mash
python -m fightml.train --p2 mix --n-envs 8
```

`--p2` is `idle`, `block` (default), `mash`, `crouch_block`, or `mix` (random dummy each episode). Checkpoints go under `runs/` (gitignored). Training does not import pygame.

TensorBoard logging is off unless you pass `--tb runs/tb` (requires `pip install tensorboard`).

## Eval (watch / human / dummies)

The agent is always **P1**. Scripted dummies or a human are **P2**. Watch mode needs pygame (`pip install -e "../game[play]"`). Training still does not import pygame.

```bash
# Watch the agent vs a dummy (auto-reset after each round; Esc quit, R reset)
python -m fightml.eval runs/ppo_block --p2 block
python -m fightml.eval runs/ppo_block --p2 mash
python -m fightml.eval runs/ppo_block --p2 mix

# Human P2 vs the agent (arrows + numpad 1/2/3, same as fightsim.play P2)
python -m fightml.eval runs/ppo_block --p2 human

# Headless score vs a dummy (no window)
python -m fightml.eval runs/ppo_block --p2 block --headless --episodes 20
```

`--frameskip` must match training (default 4). `--headless` cannot be combined with `--p2 human`.

## Tests

```bash
pytest tests
```
