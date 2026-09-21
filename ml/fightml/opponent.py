"""P2 opponent protocol: scripted dummy or a frozen SB3 zip."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fightsim.dummy import PERSONALITY_STYLES, dummy_input
from fightsim.input import RawInput
from fightsim.state import GameState
from fightml.obs import flatten_state

DEFAULT_FRAMESKIP = 4


class Opponent(Protocol):
    def reset(self, seed: int) -> None: ...

    def raw(self, state: GameState, pid: int) -> RawInput: ...


class DummyOpponent:
    """Scripted fightsim dummy. `style='mix'` picks a personality on reset."""

    def __init__(self, style: str = "mix") -> None:
        self.style = style
        self.resolved = style if style != "mix" else PERSONALITY_STYLES[0]

    def reset(self, seed: int) -> None:
        if self.style == "mix":
            self.resolved = PERSONALITY_STYLES[int(seed) % len(PERSONALITY_STYLES)]
        else:
            self.resolved = self.style

    def raw(self, state: GameState, pid: int) -> RawInput:
        return dummy_input(state, pid, self.resolved)


def _load_model(path: str):
    try:
        from sb3_contrib import MaskablePPO

        try:
            return MaskablePPO.load(path)
        except Exception:
            pass
    except ImportError:
        pass
    from stable_baselines3 import PPO

    return PPO.load(path)


class PolicyOpponent:
    """Frozen SB3 policy as P2. Uses view_pid so a P1-trained zip can act as P2."""

    def __init__(
        self,
        path: str,
        frameskip: int = DEFAULT_FRAMESKIP,
        deterministic: bool = True,
    ) -> None:
        self.path = path
        self._model = _load_model(path)
        self.frameskip = max(1, int(frameskip))
        self.deterministic = deterministic
        self._raw = RawInput()
        self._left = 0

    def reset(self, seed: int) -> None:
        self._raw = RawInput()
        self._left = 0

    def raw(self, state: GameState, pid: int) -> RawInput:
        from fightml.env import action_to_raw
        from fightml.mask import action_mask

        if self._left <= 0:
            obs = flatten_state(state, view_pid=pid)
            masks = action_mask(state, pid)
            try:
                action, _ = self._model.predict(
                    obs, deterministic=self.deterministic, action_masks=masks
                )
            except TypeError:
                action, _ = self._model.predict(obs, deterministic=self.deterministic)
            fighter = state.p1 if pid == 1 else state.p2
            self._raw = action_to_raw(int(action), fighter.facing)
            self._left = self.frameskip
        self._left -= 1
        return self._raw


def collect_pool_paths(pool_dir: str) -> list[str]:
    """Zips in pool_dir, plus sibling ppo_mix2.zip if it exists."""
    if not pool_dir:
        return []
    from fightml.paths import resolve_artifact

    found: list[str] = []
    root = Path(pool_dir)
    if not root.is_absolute():
        root = resolve_artifact(pool_dir)
    if root.is_dir():
        found.extend(str(p) for p in sorted(root.glob("*.zip")))
    sibling = root.parent / "ppo_mix2.zip"
    if sibling.is_file() and str(sibling) not in found:
        found.append(str(sibling))
    return found
