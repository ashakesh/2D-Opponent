"""PCG-XSH-RR 32-bit generator. Entire state lives in GameState."""

from __future__ import annotations

from dataclasses import dataclass

_MULT = 6364136223846793005
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK32 = 0xFFFFFFFF


@dataclass
class RngState:
    state: int
    inc: int


def seed_rng(seed: int) -> RngState:
    seed &= _MASK64
    rng = RngState(0, ((seed << 1) | 1) & _MASK64)
    rng = _step(rng)
    rng = RngState((rng.state + seed) & _MASK64, rng.inc)
    return _step(rng)


def _step(rng: RngState) -> RngState:
    return RngState((rng.state * _MULT + (rng.inc | 1)) & _MASK64, rng.inc)


def pcg32(rng: RngState) -> tuple[int, RngState]:
    old = rng.state
    new = _step(rng)
    xorshifted = (((old >> 18) ^ old) >> 27) & _MASK32
    rot = (old >> 59) & 31
    out = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & _MASK32
    return out, new


def rand_u32(rng: RngState) -> tuple[int, RngState]:
    return pcg32(rng)
