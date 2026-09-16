# fightsim

Tiny deterministic 2D fighting **simulator**. Not a fighting game: no menus, netcode, or animation system.

Simulation is independent of rendering. `step()` always advances **one** frame and takes both players' raw inputs.

## Install

```bash
pip install -e ".[play]"    # pygame debug window
pip install -e ".[dev]"     # pytest
```

Headless (no pygame):

```bash
pip install -e .
```

## Play

```bash
python -m fightsim.play                 # P1 keyboard vs idle dummy
python -m fightsim.play --p2 block
python -m fightsim.play --p2 mash
python -m fightsim.play --p2 aggressive
python -m fightsim.play --p2 mix         # random personality dummy (R re-rolls)
python -m fightsim.play --p2 human       # two keyboards
```

P1: `A`/`D` walk, `W` jump, `S` crouch, `J` light, `K` heavy, `L` throw.
Motions: `236+J` fireball, `623+J` DP, `214+J` pulse (numpad, facing-relative).
`I`/`O`/`P` fire fireball / DP / pulse via debug **command flags** (not for raw-input ML policies).

P2 (human): arrows + `Num1` light, `Num2` heavy, `Num3` throw, `Num4–6` command specials.

Dummy styles: `idle`, `block`, `mash`, `crouch_block`, `aggressive`, `defensive`, `rushdown`, `mobile`, `zoner`, `grappler`, `mix`.

## API

```python
from fightsim import Simulator, RawInput

sim = Simulator()
sim.reset(seed=0)
state = sim.step(RawInput(right=True), RawInput(left=True))
blob = sim.save_state()
sim.load_state(blob)
```

Specials normally require motion inputs. `RawInput.command_special_1` / `command_special_2` / `command_special_3` skip the motion for semantic ML and debug only.
