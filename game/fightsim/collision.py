"""Axis-aligned box helpers. Coordinates: x right, y up, origin at feet-center."""

from __future__ import annotations

from fightsim.config import CHAR_BOX_H, CHAR_BOX_W, HURT_PAD_X, HURT_PAD_Y
from fightsim.state import FighterState, Projectile


Box = tuple[int, int, int, int]  # left, bottom, right, top


def aabb_overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def overlap_x(a: Box, b: Box) -> int:
    if not aabb_overlap(a, b):
        return 0
    return min(a[2], b[2]) - max(a[0], b[0])


def character_box(f: FighterState) -> Box:
    """Fixed sprite slot at the fighter origin. Size does not follow pose push."""
    hw = CHAR_BOX_W // 2
    return (f.x - hw, f.y, f.x + hw, f.y + CHAR_BOX_H)


def pushbox(f: FighterState) -> Box:
    hw = f.push_w // 2
    return (f.x - hw, f.y, f.x + hw, f.y + f.push_h)


def hurtbox(f: FighterState) -> Box:
    l, b, r, t = pushbox(f)
    return (l - HURT_PAD_X, b, r + HURT_PAD_X, t + HURT_PAD_Y)


def hurtboxes(f: FighterState) -> list[Box]:
    return [hurtbox(f)]


def any_overlap(a: list[Box], b: list[Box]) -> bool:
    return any(aabb_overlap(x, y) for x in a for y in b)


def hitbox_from_local(
    f: FighterState, offset_x: int, offset_y: int, w: int, h: int
) -> Box:
    cx = f.x + f.facing * offset_x
    cy = f.y + offset_y
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


def projectile_box(p: Projectile) -> Box:
    return (p.x - p.w // 2, p.y - p.h // 2, p.x + p.w // 2, p.y + p.h // 2)
