# fightml

ML research for fighting-game agents. **This package does not implement a game.**

It depends on a backend that matches [`fightml.protocol.FightingSimulator`](fightml/protocol.py):

```
reset / step / get_state / save_state / load_state / render
```

The default backend will be `fightsim` (`../game`). Swap the backend by wrapping any other engine that implements the same protocol (including an Ikemen GO fallback later).

`fightml` must not be imported from `game/`. Rewards, Gymnasium/PettingZoo wrappers, policies, and opponent pools live here — not in the simulator.

## Status

Milestone 1 is the simulator only. Training (`env.py`, `train.py`) is the next milestone.

## Install

```bash
pip install -e "../game" -e "."
```
