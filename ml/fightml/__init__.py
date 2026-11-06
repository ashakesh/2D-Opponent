"""
ML research package.

Rules for usage and modification:
1. None of implementation of the fighting game should live here.

"""

from fightml.env import FightEnv, Phase1Action, action_to_raw, make_fight_env
from fightml.obs import OBS_SIZE, flatten_state
from fightml.protocol import FightingSimulator

__all__ = [
    "FightEnv",
    "FightingSimulator",
    "OBS_SIZE",
    "Phase1Action",
    "action_to_raw",
    "flatten_state",
    "make_fight_env",
]
