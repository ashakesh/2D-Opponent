"""Flatten GameState into a numeric observation vector.

Pixels, input buffers, and RNG are intentionally omitted (Phase-1 MLP).
"""

from __future__ import annotations

import numpy as np

from fightsim.config import HEALTH_MAX, METER_MAX, ROUND_TIMER, STAGE_HEIGHT, STAGE_WIDTH
from fightsim.state import FSM, FighterState, GameState, Projectile, initial_state

FSM_ORDER: tuple[FSM, ...] = tuple(FSM)

# Typical combat scales (not sim constants) so features sit near [-1, 1].
_VEL_X = STAGE_WIDTH * 0.02
_VEL_Y = STAGE_HEIGHT * 0.15
_STUN = 32.0
_STATE_FRAME = 40.0
_INVULN = 16.0
_LANDING = 8.0
_BOUNCE = 2.0
_HITSTOP = 16.0
_PROJ_LIFE = 80.0
_PROJ_VX = STAGE_WIDTH * 0.02
_MAX_PROJECTILES = 2


def _fighter_features(f: FighterState) -> list[float]:
    fsm_oh = [1.0 if f.fsm is state else 0.0 for state in FSM_ORDER]
    return [
        f.x / STAGE_WIDTH,
        f.y / STAGE_HEIGHT,
        f.vx / _VEL_X,
        f.vy / _VEL_Y,
        float(f.facing),
        f.health / HEALTH_MAX,
        f.meter / METER_MAX,
        *fsm_oh,
        f.state_frame / _STATE_FRAME,
        f.hitstun / _STUN,
        f.blockstun / _STUN,
        f.invuln / _INVULN,
        f.landing_lock / _LANDING,
        f.bounce_count / _BOUNCE,
        1.0 if f.y > 0 else 0.0,
        1.0 if f.can_special_cancel else 0.0,
        1.0 if f.knockdown_pending else 0.0,
        1.0 if f.move_has_hit else 0.0,
    ]


def _projectile_features(proj: Projectile | None, p1: FighterState) -> list[float]:
    if proj is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    owner = 1.0 if proj.owner == 1 else -1.0
    return [
        owner,
        (proj.x - p1.x) / STAGE_WIDTH,
        (proj.y - p1.y) / STAGE_HEIGHT,
        proj.vx / _PROJ_VX,
        proj.lifetime / _PROJ_LIFE,
    ]


def flatten_state(state: GameState) -> np.ndarray:
    """P1-centric float32 vector for an MLP policy."""
    feats: list[float] = []
    feats.extend(_fighter_features(state.p1))
    feats.extend(_fighter_features(state.p2))
    feats.append((state.p2.x - state.p1.x) / STAGE_WIDTH)
    feats.append(state.timer / ROUND_TIMER)
    feats.append(state.hitstop / _HITSTOP)
    slots = list(state.projectiles[:_MAX_PROJECTILES])
    while len(slots) < _MAX_PROJECTILES:
        slots.append(None)
    for proj in slots:
        feats.extend(_projectile_features(proj, state.p1))
    return np.asarray(feats, dtype=np.float32)


OBS_SIZE = int(flatten_state(initial_state(0)).shape[0])
