"""Hiring and land purchases."""

from typing import List, Tuple

from farm_agent.advisors.base import BaseAdvisor
from farm_agent.config import (
    FARM_HAND_COST_MULT,
    HIRE_BACKLOG_PER_UNIT,
    HIRE_BATCH_MAX,
    HIRE_CASH_FLOOR,
    HIRE_DAILY_CAP,
    HIRE_LAST_HOUR,
    HIRE_SCHEDULE,
    LAND_EMPTY_THRESHOLD,
    LAND_PRICES,
    PRIO_MKT_HIRE,
    PRIO_MKT_LAND,
)
from farm_agent.messages import MarketRequest, TaskRequest
from farm_agent.world.pricing import fib
from farm_agent.world.state import WorldModel


class ExpansionAdvisor(BaseAdvisor):

    def get_requests(self, world: WorldModel) -> Tuple[List[TaskRequest], List[MarketRequest]]:
        market_orders: List[MarketRequest] = []
        tc = world.classified_tiles

        # Hiring: today's target is the schedule floor, raised by one whenever
        # the chore backlog exceeds HIRE_BACKLOG_PER_UNIT per unit (that
        # reactive part alone is capped at HIRE_DAILY_CAP).
        hires_today = world.my_farm.get("hires_today", 0)
        pending_urgent = len(tc.water) + len(tc.feed) + len(tc.care) + len(tc.fert)
        target = HIRE_SCHEDULE[min(world.day, len(HIRE_SCHEDULE) - 1)]
        if hires_today < HIRE_DAILY_CAP and pending_urgent > len(world.units) * HIRE_BACKLOG_PER_UNIT:
            target = max(target, hires_today + 1)

        # Fill the target in one batch: a hand hired at dawn costs the same as
        # one hired at noon and works a full day. Each order carries its own
        # marginal wage so the allocator can settle it on its cash ledger.
        if world.hour <= HIRE_LAST_HOUR and hires_today < target:
            budget = float(world.money)
            for k in range(min(target - hires_today, HIRE_BATCH_MAX)):
                hire_cost = FARM_HAND_COST_MULT * fib(hires_today + k)
                if budget - hire_cost < HIRE_CASH_FLOOR:
                    break
                budget -= hire_cost
                market_orders.append(MarketRequest(order=["HIRE"], priority=PRIO_MKT_HIRE,
                                                   category="hire", cost_estimate=hire_cost))

        # Land: request the next quadrant once empty tiles run out. There is
        # deliberately no cash check here. The allocator settles sells before
        # land, so a turn that starts short but sells enough can still afford
        # it; a start-of-turn check here delayed purchases by 6-7 days.
        unlocked_quadrants = world.my_farm.get("unlocked_quadrants", ["NW"])
        n_extra_land = len(unlocked_quadrants) - 1
        if n_extra_land < len(LAND_PRICES) and len(tc.empty) <= LAND_EMPTY_THRESHOLD:
            market_orders.append(MarketRequest(order=["BUY_LAND"], priority=PRIO_MKT_LAND,
                                               category="land", cost_estimate=LAND_PRICES[n_extra_land]))

        return [], market_orders
