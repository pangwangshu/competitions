"""Maintenance: harvest, care, collect and apply fertilizer, clear weeds."""

from typing import List, Tuple

from farm_agent.advisors.base import BaseAdvisor
from farm_agent.config import (
    CROPS,
    PRIO_APPLY_FERT_BERRY,
    PRIO_APPLY_FERT_DUE,
    PRIO_CARE,
    PRIO_COLLECT_FERT,
    PRIO_HARVEST,
    PRIO_WEED,
)
from farm_agent.messages import MarketRequest, TaskRequest
from farm_agent.world.state import WorldModel


class MaintenanceAdvisor(BaseAdvisor):

    def get_requests(self, world: WorldModel) -> Tuple[List[TaskRequest], List[MarketRequest]]:
        tasks: List[TaskRequest] = []
        tc = world.classified_tiles

        for pos in tc.harvest:
            tasks.append(TaskRequest(category="harvest", pos=pos, priority=PRIO_HARVEST,
                                     action=["HARVEST"]))
        for pos in tc.care:
            tasks.append(TaskRequest(category="care", pos=pos, priority=PRIO_CARE, action=["CARE"]))
        for pos in tc.fert:
            tasks.append(TaskRequest(category="fert", pos=pos, priority=PRIO_COLLECT_FERT,
                                     action=["COLLECT_FERTILIZER"]))
        for pos in tc.apply_fert:
            # Every planned application is due today. On an ongoing crop it pays
            # a doubled production that expires tonight.
            prio = PRIO_APPLY_FERT_BERRY if _is_ongoing(world, pos) else PRIO_APPLY_FERT_DUE
            tasks.append(TaskRequest(category="apply_fert", pos=pos, priority=prio,
                                     action=["FERTILIZE"], requires_inv={"FERTILIZER": 1}))
        for pos in tc.weed:
            tasks.append(TaskRequest(category="weed", pos=pos, priority=PRIO_WEED, action=["DIG"]))

        return tasks, []


def _is_ongoing(world, pos):
    """True for a live strawberry or tomato tile."""
    x, y = pos
    tile = world.tiles[y][x]
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False
    cd = CROPS.get(tile.get("crop"))
    return cd is not None and cd["ongoing"]
