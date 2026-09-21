"""Flatten GameState into a numeric observation vector.

Pixels, input buffers, and RNG are intentionally omitted (Phase-1 MLP).
view_pid=1 is the training layout. view_pid=2 swaps me/opp for a frozen P2 policy.
No coordinate mirror: random spawn already trains both screen sides.
"""

from __future__ import annotations

import numpy as np

from fightsim.config import HEALTH_MAX, METER_MAX, ROUND_TIMER, STAGE_HEIGHT, STAGE_WIDTH
from fightsim.state import FSM, FighterState, GameState, Projectile, initial_state

FSM_ORDER: tuple[FSM, ...] = tuple(FSM)

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


def _projectile_features(
    proj: Projectile | None, me: FighterState, view_pid: int
) -> list[float]:
    if proj is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    owner = 1.0 if proj.owner == view_pid else -1.0
    return [
        owner,
        (proj.x - me.x) / STAGE_WIDTH,
        (proj.y - me.y) / STAGE_HEIGHT,
        proj.vx / _PROJ_VX,
        proj.lifetime / _PROJ_LIFE,
    ]


def flatten_state(state: GameState, view_pid: int = 1) -> np.ndarray:
    """Viewer-centric float32 vector. view_pid 1 or 2; default matches Phase-1 P1."""
    if int(view_pid) == 2:
        me, opp = state.p2, state.p1
        pid = 2
    else:
        me, opp = state.p1, state.p2
        pid = 1
    feats: list[float] = []
    feats.extend(_fighter_features(me))
    feats.extend(_fighter_features(opp))
    feats.append((opp.x - me.x) / STAGE_WIDTH)
    feats.append(state.timer / ROUND_TIMER)
    feats.append(state.hitstop / _HITSTOP)
    slots = list(state.projectiles[:_MAX_PROJECTILES])
    while len(slots) < _MAX_PROJECTILES:
        slots.append(None)
    for proj in slots:
        feats.extend(_projectile_features(proj, me, pid))
    return np.asarray(feats, dtype=np.float32)


OBS_SIZE = int(flatten_state(initial_state(0)).shape[0])
