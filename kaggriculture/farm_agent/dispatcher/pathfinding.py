"""Grid movement helpers. The farm has no obstacles, so Manhattan routing is optimal."""

from typing import Tuple


def step_toward(fx: int, fy: int, tx: int, ty: int) -> str:
    """One step toward the target: close the x gap first, then y."""
    if fx < tx:
        return "EAST"
    if fx > tx:
        return "WEST"
    if fy < ty:
        return "SOUTH"
    if fy > ty:
        return "NORTH"
    return "PASS"


def manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
