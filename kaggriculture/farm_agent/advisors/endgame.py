"""Endgame: stop investing, harvest everything, liquidate the shed.

Only the bank balance scores; unsold goods, standing crops and animals are
worth nothing at the end. From ENDGAME_STOP_INVEST_DAY the allocator stops
discretionary buying; from ENDGAME_HARVEST_ALL_DAY every productive tile is
harvested; and the shed is sold down to the feed reserve, which tapers to zero
on the final day.
"""

from typing import List, Tuple

from farm_agent.advisors.base import BaseAdvisor
from farm_agent.config import (
    CROPS,
    ENDGAME_HARVEST_ALL_DAY,
    ENDGAME_STOP_INVEST_DAY,
    PRIO_ENDGAME_HARVEST,
    PRIO_MKT_ENDGAME_SELL,
    PRODUCTS,
)
from farm_agent.messages import MarketRequest, TaskRequest
from farm_agent.planners.late_wheat import planned_harvest_day
from farm_agent.world.pricing import feed_wheat_reserve
from farm_agent.world.state import WorldModel

PRIO_WATER_BEFORE_HARVEST = 985.0


def is_active(world: WorldModel) -> bool:
    return world.day >= ENDGAME_STOP_INVEST_DAY


class EndgameAdvisor(BaseAdvisor):

    def get_requests(self, world: WorldModel) -> Tuple[List[TaskRequest], List[MarketRequest]]:
        if not is_active(world):
            return [], []
        tasks: List[TaskRequest] = []
        market_reqs: List[MarketRequest] = []
        tc = world.classified_tiles

        if world.day >= ENDGAME_HARVEST_ALL_DAY:
            for y in range(world.board_size):
                for x in range(world.board_size):
                    tile = world.tiles[y][x]
                    if not isinstance(tile, dict):
                        continue
                    if (tile.get("kind") == "PLANT" and tile.get("yield_units", 0) > 0
                            and world.day - tile["planted_day"] >= CROPS[tile["crop"]]["first_yield_day"]):
                        cd = CROPS[tile["crop"]]
                        age = world.day - tile["planted_day"]
                        if not cd["ongoing"] and tile["yield_units"] < cd["max_yield"]:
                            # A one-time crop below its cap waits for its planned
                            # day, and is watered first while watering still adds yield.
                            if world.day < planned_harvest_day(tile):
                                continue
                            if (not tile.get("watered_today")
                                    and (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]):
                                tasks.append(TaskRequest(category="water", pos=(x, y),
                                                         priority=PRIO_WATER_BEFORE_HARVEST,
                                                         action=["WATER"]))
                                continue
                        tasks.append(TaskRequest(category="harvest", pos=(x, y),
                                                 priority=PRIO_ENDGAME_HARVEST, action=["HARVEST"]))
                    elif (tile.get("kind") in ("COOP", "PASTURE") and tile.get("animal")
                          and tile.get("yield_units", 0) > 0 and (x, y) not in tc.deferred):
                        tasks.append(TaskRequest(category="harvest", pos=(x, y),
                                                 priority=PRIO_ENDGAME_HARVEST, action=["HARVEST"]))

        # Liquidate the shed down to the feed reserve, ahead of routine sells.
        for item in PRODUCTS:
            qty = world.shed.get(item, 0)
            if qty <= 0:
                continue
            if item == "WHEAT":
                qty = max(0, qty - feed_wheat_reserve(tc.animal_count, len(tc.unoccupied_structure), world.day))
            if qty > 0:
                market_reqs.append(MarketRequest(order=["SELL", item, qty], priority=PRIO_MKT_ENDGAME_SELL,
                                                 category="endgame_sell"))
        return tasks, market_reqs
