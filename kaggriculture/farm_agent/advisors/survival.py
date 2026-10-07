"""Survival: water plants and feed animals before they are lost."""

from typing import List, Tuple

from farm_agent.advisors.base import BaseAdvisor
from farm_agent.config import CROPS, PRIO_FEED, PRIO_FEED_URGENT, PRIO_WATER, PRIO_WATER_URGENT, TURNS_PER_DAY
from farm_agent.dispatcher.pathfinding import manhattan_dist
from farm_agent.messages import MarketRequest, TaskRequest
from farm_agent.planners.late_wheat import is_late_wheat
from farm_agent.world.state import WorldModel, planned_skip

# A deliberately skipped watering escalates to urgent only once the nearest
# unit has fewer than this many turns of slack left today.
WATER_DEFER_SLACK = 6
PRIO_LATE_WHEAT_WATER = 890.0
PRIO_LATE_WHEAT_WATER_URGENT = 940.0


class SurvivalAdvisor(BaseAdvisor):
    """Two missed days kill a plant or lose an animal; the second miss is urgent."""

    def get_requests(self, world: WorldModel) -> Tuple[List[TaskRequest], List[MarketRequest]]:
        tasks: List[TaskRequest] = []
        tc = world.classified_tiles

        for pos in tc.feed:
            x, y = pos
            tile = world.tiles[y][x]
            unfed = tile.get("consecutive_unfed", 0) if isinstance(tile, dict) else 0
            prio = PRIO_FEED_URGENT if unfed >= 1 else PRIO_FEED
            tasks.append(TaskRequest(category="feed", pos=pos, priority=prio,
                                     action=["FEED"], requires_inv={"WHEAT": 1}))

        for pos in tc.water:
            if pos in tc.water_skip:
                continue  # watering today changes no output
            x, y = pos
            tile = world.tiles[y][x]
            # A new plant starts one dry day behind, so its first day is urgent.
            unwatered = tile.get("consecutive_unwatered", 0) if isinstance(tile, dict) else 0
            prio = PRIO_WATER_URGENT if unwatered >= 1 else PRIO_WATER
            # After a deliberate skip, the second watering keeps routine priority
            # until the nearest unit is running out of time.
            if unwatered >= 1 and planned_skip(CROPS[tile["crop"]], tile, world.day):
                turns_left = TURNS_PER_DAY - world.hour
                near = min((manhattan_dist(pos, u) for _, u in world.units), default=0)
                if near + WATER_DEFER_SLACK < turns_left:
                    prio = PRIO_WATER
            if is_late_wheat(tile):
                prio = PRIO_LATE_WHEAT_WATER_URGENT if unwatered >= 1 else PRIO_LATE_WHEAT_WATER
            tasks.append(TaskRequest(category="water", pos=pos, priority=prio, action=["WATER"]))

        return tasks, []
