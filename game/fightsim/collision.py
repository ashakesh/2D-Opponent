"""Axis-aligned box helpers. Coordinates: x right, y up, origin at feet-center."""

from __future__ import annotations

from fightsim.config import HURT_PAD_X, HURT_PAD_Y
from fightsim.state import FighterState, Projectile


Box = tuple[int, int, int, int]  # left, bottom, right, top


def aabb_overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def overlap_x(a: Box, b: Box) -> int:
    if not aabb_overlap(a, b):
        return 0
    return min(a[2], b[2]) - max(a[0], b[0])


def pushbox(f: FighterState) -> Box:
    hw = f.push_w // 2
    return (f.x - hw, f.y, f.x + hw, f.y + f.push_h)


def hurtbox(f: FighterState) -> Box:
    l, b, r, t = pushbox(f)
    return (l - HURT_PAD_X, b, r + HURT_PAD_X, t + HURT_PAD_Y)


def hitbox_from_local(
    f: FighterState, offset_x: int, offset_y: int, w: int, h: int
) -> Box:
    cx = f.x + f.facing * offset_x
    cy = f.y + offset_y
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


def projectile_box(p: Projectile) -> Box:
    return (p.x - p.w // 2, p.y - p.h // 2, p.x + p.w // 2, p.y + p.h // 2)
