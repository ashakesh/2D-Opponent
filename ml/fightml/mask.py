"""Legal semantic-action mask. Used by MaskablePPO via FightEnv.action_masks()."""

from __future__ import annotations

import numpy as np

from fightsim.config import PULSE_COST, SPECIAL_COST
from fightsim.state import FSM, GameState
from fightml.env import N_ACTIONS, Phase1Action


def action_mask(state: GameState, pid: int = 1) -> np.ndarray:
    """True where a Phase-1 semantic action can do something other than no-op."""
    f = state.p1 if pid == 1 else state.p2
    air = f.y > 0
    mask = np.ones(N_ACTIONS, dtype=bool)

    if air:
        mask[int(Phase1Action.DOWN)] = False
        mask[int(Phase1Action.THROW)] = False
        mask[int(Phase1Action.HEAVY)] = False
        mask[int(Phase1Action.CROUCH_LIGHT)] = False
        mask[int(Phase1Action.CROUCH_BLOCK)] = False
        mask[int(Phase1Action.SPECIAL_1)] = False
        mask[int(Phase1Action.SPECIAL_2)] = False
        mask[int(Phase1Action.SPECIAL_3)] = False
        mask[int(Phase1Action.UP)] = False
    else:
        pass

    if f.meter < SPECIAL_COST:
        mask[int(Phase1Action.SPECIAL_1)] = False
        mask[int(Phase1Action.SPECIAL_2)] = False
    if f.meter < PULSE_COST:
        mask[int(Phase1Action.SPECIAL_3)] = False

    if f.fsm in (FSM.HITSTUN, FSM.BLOCKSTUN, FSM.KNOCKDOWN, FSM.DEAD) or f.health <= 0:
        for act in (
            Phase1Action.LIGHT,
            Phase1Action.HEAVY,
            Phase1Action.THROW,
            Phase1Action.SPECIAL_1,
            Phase1Action.SPECIAL_2,
            Phase1Action.SPECIAL_3,
            Phase1Action.CROUCH_LIGHT,
            Phase1Action.UP,
        ):
            mask[int(act)] = False

    return mask
