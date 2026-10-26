# ML research notes

Training and experiment code live in `ml/` (`fightml`). The game lives in `game/` (`fightsim`) and is maintained separately. Do not turn the simulator into a training library, and do not import `fightml` from `game/`.

This document is the source of truth for goals, the sim contract, reward pitfalls, and what to do next.

---

## Goal

A short-term solo research project: an agent that learns **good, adaptive, potentially creative** fighting-game play.

The game is **not** the product. Visual polish, menus, and engine completeness do not matter. Research value is:

```
simulation → reproducible data → self-play → adaptive behavior → technique / combo discovery
```

A **custom deterministic simulator** was chosen over Ikemen GO so that iteration is fast and state/control is exact. Ikemen remains a fallback only if this sim cannot express a research-blocking mechanic. The Python API stays the same either way.

Long-term roadmap (in order):

1. **Competent play** — full `GameState`, complete move set, semantic actions.
2. **Adaptive play** — self-play / opponent populations, not one frozen dummy.
3. **Raw-input learning** — frame-level buttons; the agent must discover motions, cancels, combos.
4. **Generalization** — unfamiliar opponents (later).
5. **Pixels** — learn from rendered frames (last, if time).

Do **not** encode “creativity” in the reward. First show that the agent can learn strong adaptive behavior.

---

## Repo split

```
2D-Opponent/
  game/     fightsim — standalone combat simulator
  ml/       fightml — research; wraps any backend matching the protocol
```

**Hard rule:** `game` never imports `ml`. `ml` depends on `game` as a library (or on another engine that implements the same protocol).

Either folder can be copied out. Applying this research to another game means writing a new adapter that matches [`fightml/protocol.py`](fightml/protocol.py).

```
                    ┌──────────────────┐
raw inputs ───────► │    Simulator     │
                    │  physics/combat  │
                    └────────┬─────────┘
                             │ GameState
                    /                  \
                   ▼                    ▼
            ML observation           Renderer
            (fightml wrapper)      (optional pygame)
```

---

## Simulator contract

[`FightingSimulator`](fightml/protocol.py):

```
reset(seed) → GameState
step(p1_input, p2_input) → GameState    # exactly ONE simulation frame
get_state()
save_state() → bytes
load_state(snapshot)
render()                               # optional; must not be required to step
```

Implementation: `fightsim.Simulator` in [`game/fightsim/api.py`](../game/fightsim/api.py).

**Always both players, always one frame.** Frameskip, action-repeat, and the opponent policy belong in the `fightml` wrapper, never in the sim.

`render()` no-ops if pygame is missing. Training must not import pygame.

### Actions

The sim speaks **raw controller inputs only**:

`left, right, up, down, a, b, c, d`

Facing-relative **block = hold back**, not a button.

| Button | Role |
|--------|------|
| A | Light (stand / crouch+A low / jump+A overhead) |
| B | Heavy |
| C | Throw |
| 236+A | Fireball (meter; 1800 + chip; long commitment) |
| 623+A | DP / uppercut (meter, some invuln, highest launch) |
| 214+A | Pulse (cheaper meter; shove; **erases fireballs**) |
| 66 / 44 | Forward / back dash (strict tap window) |

**Semantic actions** (`SemanticAction` in `fightsim.input`) map onto one frame of `RawInput`. `SPECIAL_1` / `SPECIAL_2` / `SPECIAL_3` set `command_special_1/2/3` so Phase 1 can special in a single `step()`. Those flags are **debug / semantic-ML only**. If a raw-input experiment sets them, Phase 3 is fake — the agent never has to discover 236/623/214.

Phase-1 Gym action `Phase1Action.SPECIAL_3` maps to pulse.

### State and replays

`GameState` is explicit and JSON-serializable (integer subpixels, PCG32 RNG **inside** the state, no hidden globals). Replays (`fightsim.replay`) record:

- simulator version, character config, seed
- initial snapshot
- `inputs[t]`, `state[t]`, `events[t]`

**Do not put `reward[t]` in the game tape.** Add rewards in the wrapper. Useful events already include hits/blocks/whiffs/throws/projectiles; `combo_ended` includes `length` so you can bin short / medium / long later without a “style” reward.

Bump note: current sim is **0.3.0**. Older replays are invalid.

### What exists in the game (not ML's job)

- Two identical fighters, one stage, dummy CPU in `fightsim.dummy`: `idle` / `block` / `mash` / `crouch_block` plus personalities `aggressive` / `defensive` / `rushdown` / `mobile` / `zoner` / `grappler`. Play `--p2 mix` picks a personality (reproducible from `--seed`; `R` re-rolls). HUD shows the resolved name.
- Headless determinism, save/load, replay tests under `game/tests/`
- Play: `python -m fightsim.play` (optional pygame)

**Not in the game, and not to be added there:** Gymnasium, PettingZoo, PyTorch, opponent pools, checkpoints, pixel observations.

---

## Install and how to run the game

Play / inspect the sim (no `ml` needed):

```bash
pip install -e "./game[play]"
python -m fightsim.play                 # P1 vs idle dummy
python -m fightsim.play --p2 block
python -m fightsim.play --p2 mash
python -m fightsim.play --p2 mix
python -m fightsim.play --p2 human
```

P1: WASD + J/K/L (light/heavy/throw). `I`/`O`/`P` fire command fireball/DP/pulse (do not use these in raw-input training). `R` reset, Esc quit.

Tests:

```bash
pip install -e "./game[dev]"
PYTHONPATH=game python3 -m pytest game/tests
```

ML package:

```bash
pip install -e "./game" -e "./ml"
```

`fightml` has `env.py` / `train.py` / `eval.py`. Named generation zips are in [`artifacts/`](artifacts/) (see **What exists now**). New local trains still write under **`ml/runs/`** (gitignored; relative `runs/...` flags resolve there regardless of cwd). Eval plays the zip as **P2** (`--p1` dummy/human).

---

## Roadmap

### 1 — Competent play (done)

Full `GameState` → policy → **semantic** action → `RawInput` adapter → `step()`.

Opponent: `fightsim.dummy` (`block`, then `mash`). Question: **can an agent learn to fight at all?**

Suggested stack: Gymnasium env wrapping one `Simulator`; PPO (Stable-Baselines3); MLP on a flattened state vector; discrete semantic action space.

### 2 — Self-play

Checkpoint mix (current + historical policies, e.g. past-self 50% / PFSP). Question: **adaptive, or a one-dummy exploit?**

### 3 — Raw inputs

Same sim; **no** command flags. Question: **does it discover 236/623, dashes, light→heavy gatling, special cancel?**

Expect this to be much harder. Use recurrence. Do not start here.

### 4 — Optional later

Recurrent / transformer policies, opponent modeling, hierarchical options, population-based training, intrinsic exploration, world models, unseen-character generalization, **pixel** observations last.

---

## Combat vocabulary the agent can use (0.3.0)

Keep this in mind when designing obs/rewards; do not flatten it into a “creativity” bonus.

- **Jab** — fastest, shortest. Interrupt / tick / confirm.
- **Crouch kick** — longest grounded normal, **low** (must crouch-block).
- **Heavy** — slower, more hitstun; on-hit **gatling from lights**; punish tool.
- **Jump-in** — **overhead** (stand-block only), long down-forward kick; hitting airborne or jump-in connect **blows back with knockdown**. Landing has a short recovery so blocked jump-ins are punishable.
- **Throw** — beats block; **launches backward** (mid-arc) then knockdown / bounce. Short recovery **on hit** so you can jab a wall bounce.
- **Fireball** — anti-camp clock; costs 250 meter; **1800** raw / chips on block (`2/5`); long startup (20f) and recovery (28f). Jump or pulse it; do not throw it in DP range.
- **DP** — anti-air / reversal, invuln on startup, **highest launch arc**, long recovery if whiffed, short on hit (juggle into fireball). 1500 damage — less than a raw fireball, much faster.
- **Pulse** — 214+A, 150 meter; almost no damage; shoves; **erases incoming fireballs**. Not a launch.
- **Meter** starts at one fireball/DP (250 / 1000). Defender gains more meter for a successful block than the attacker. Hits weighted: special/throw > heavy > light > pulse.
- **Cancels (combo routes, not a combo list):** on-hit special cancel from normals; on-hit **light → heavy** only. Launches + **one** floor/wall bounce give a juggle window. Events: `combo_ended` with `length`. Bin 2 / 3–4 / 5+ in the wrapper if you want.
- **Mix:** high / low / throw. Strike block does not chip; fireballs do. No air block.

---

## Promising ML approaches

Start simple. Escalate only after Phase 1 actually beats a dummy.

### Must-do baseline

- **Gymnasium** single-agent env (inject P2). PettingZoo can wait until self-play.
- **PPO + MLP** on a numeric vector built from `GameState` (positions, velocities, health, meter, fsm one-hots, stun timers, projectile slots, facing, flags). Do not start from pixels.
- **Discrete semantic actions** (`SemanticAction` / a slightly grouped subset). Map with `semantic_to_raw(action, facing)`.
- **Reward:** round `+1 / -1` plus a **small** damage delta (dealt − taken). That is enough. Auxiliary signals (corner, meter) only if Phase 1 plateaus.
- **Action repeat / frameskip in the wrapper** (e.g. 4 sim frames per policy step). Never change `step()` to skip.
- Train vs `block`, then `mash`, then a mix of dummies.

### Strong next steps

- **Action mask** for illegal semantics (throw in air, special with no meter, heavy while not cancelable). PPO will otherwise burn entropy on no-ops.
- **Vectorized envs:** many `Simulator` instances. `deepcopy` per frame is acceptable until profiling says otherwise. Do not rewrite the sim in C++/Numba for speed first.
- **Self-play + frozen checkpoints** (PFSP / “uniform over last N”). A static dummy yields brittle one-trick policies.
- **Identity / opponent embedding** once you have a pool (which dummy or which checkpoint).

### Raw-input phase (after self-play works)

- Same `Simulator`; disable command flags.
- **LSTM or short transformer** — 236/623 are sequences.
- Optional **sticky actions** so it holds directions.
- Optional **RND / count-based bonus** on rare motion completions so it discovers specials. Keep the bonus small vs win/damage.
- If flat raw fails: **hierarchical** (semantic options that unroll into raw macros), still using the same sim.

### Later / optional

- League or population-based training.
- Behavior cloning from `ReplayRecorder` tapes (dummy or human `play.py`) then RL fine-tune.
- Opponent modeling heads.

### Avoid early

- Pixel CNNs, world models, MCTS/AlphaZero, Ikemen GO.
- Huge reward soups and “play style / creativity / combo beauty” terms. Use `combo_ended.length` only as analysis (or a tiny bonus after win rate is real).
- A second simulator path for semantic moves.
- Importing pygame or the debug renderer on the training process.

```
GameState → fightml env → Policy → RawInput P1
                 ↑                      ↓
         dummy / frozen policy → RawInput P2
                                fightsim.step
```

---

## What exists now

Phase 1 (competent play vs scripted dummies) is in the tree:

- [`env.py`](fightml/env.py) / [`train.py`](fightml/train.py) / [`eval.py`](fightml/eval.py) — Gymnasium env, MaskablePPO, eval as P2
- [`protocol.py`](fightml/protocol.py) stays free of `fightsim` imports so a future backend can drop in
- [`artifacts/`](artifacts/) — named generation zips to watch without retraining:
  - `ppo_block.zip` — first dummy (block)
  - `ppo_mix.zip` — early personality mix
  - `ppo_mix2.zip` — mash-heavy mix
  - `ppo_mix3.zip` — jump-stall mix (timeout-at-0; the failure mix4 fixed)
  - `ppo_mix4.zip` — current: dummy-only, 30e6 steps, timeout win +0.25 / indecisive timeout −0.25, time cost, range-gated whiff. KOs zoner/defensive instead of 10k–10k stalling

```bash
python -m fightml.eval artifacts/ppo_mix4 --p1 human
python -m fightml.eval artifacts/ppo_mix3 --p1 zoner
```

**Next:** PFSP / sampling a frozen pool (adaptive play, not one-dummy exploit), then raw-input with `command_special_*` off.

---

## Constraints checklist

- [ ] `game` is not modified for training convenience (no torch, no gym in `fightsim`).
- [ ] `step()` stays 1 frame, two inputs.
- [ ] Phase 3 policies never set `command_special_*`.
- [ ] Replays stay reward-free.
- [ ] Pygame is not a training dependency.
- [ ] Do not rebuild Ikemen unless the custom sim is research-blocking.
