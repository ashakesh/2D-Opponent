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
python -m fightsim.play --p2 block       # P2 always holds back
python -m fightsim.play --p2 mash        # P2 taps light
python -m fightsim.play --p2 human       # two keyboards
```

P1: `A`/`D` walk, `W` jump, `S` crouch, `J` light, `K` heavy, `L` throw.
Motions: `236+J` fireball, `623+J` DP (numpad, facing-relative). `I`/`O` fire the same specials via the debug **command flag** (not available to raw-input ML policies).

P2 (human): arrows + `Num1` light, `Num2` heavy, `Num3` throw.

## API

```python
from fightsim import Simulator, RawInput

sim = Simulator()
sim.reset(seed=0)
state = sim.step(RawInput(right=True), RawInput(left=True))
blob = sim.save_state()
sim.load_state(blob)
```

Specials normally require motion inputs. `RawInput.command_special_1` / `command_special_2` skip the motion for semantic ML and debug only.
