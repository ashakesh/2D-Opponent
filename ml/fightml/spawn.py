"""Reset-time spawn helpers. Applied in the ML wrapper, not in fightsim.initial_state."""

from __future__ import annotations

from fightsim.config import STAGE_WIDTH, START_P1_X, START_P2_X, SUBPIXEL, THROW_RANGE
from fightsim.state import GameState

GAP_MIN = THROW_RANGE
GAP_MAX = 220 * SUBPIXEL
EDGE_MARGIN = 80 * SUBPIXEL


def canonical_gap() -> int:
    return START_P2_X - START_P1_X


def place_fighters(
    state: GameState,
    p1_on_left: bool,
    gap: int | None = None,
    mid: int | None = None,
) -> None:
    """Grounded start; facings toward each other. Does not touch RNG or health."""
    if gap is None:
        gap = canonical_gap()
    gap = int(gap)
    span = STAGE_WIDTH - 2 * EDGE_MARGIN
    if gap > span:
        gap = span
    if mid is None:
        mid = STAGE_WIDTH // 2
    half = gap // 2
    left_x = mid - half
    right_x = mid + half
    min_x = EDGE_MARGIN
    max_x = STAGE_WIDTH - EDGE_MARGIN
    if left_x < min_x:
        right_x += min_x - left_x
        left_x = min_x
    if right_x > max_x:
        left_x -= right_x - max_x
        right_x = max_x
    if p1_on_left:
        state.p1.x = left_x
        state.p2.x = right_x
        state.p1.facing = 1
        state.p2.facing = -1
    else:
        state.p1.x = right_x
        state.p2.x = left_x
        state.p1.facing = -1
        state.p2.facing = 1
    state.p1.y = 0
    state.p2.y = 0
    state.p1.vx = 0
    state.p2.vx = 0
    state.p1.vy = 0
    state.p2.vy = 0


def randomize_spawn(state: GameState, rng) -> None:
    p1_on_left = bool(int(rng.integers(0, 2)))
    gap = int(rng.integers(GAP_MIN, GAP_MAX + 1))
    span = STAGE_WIDTH - 2 * EDGE_MARGIN
    if gap > span:
        gap = span
    half = gap // 2
    mid_lo = EDGE_MARGIN + half
    mid_hi = STAGE_WIDTH - EDGE_MARGIN - half
    if mid_hi < mid_lo:
        mid = STAGE_WIDTH // 2
    else:
        mid = int(rng.integers(mid_lo, mid_hi + 1))
    place_fighters(state, p1_on_left, gap=gap, mid=mid)
