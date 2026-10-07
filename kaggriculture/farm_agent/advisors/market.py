"""Throttled selling and the feed-wheat top-up."""

from typing import List, Tuple

from farm_agent.advisors.base import BaseAdvisor
from farm_agent.config import (
    MARKET_I0,
    MONEY_RESERVE,
    PRIO_MKT_FEED_TOPUP,
    PRIO_MKT_SELL,
    PRODUCTS,
    WHEAT_BUY_THRESHOLD_RATIO,
)
from farm_agent.messages import MarketRequest, TaskRequest
from farm_agent.world.pricing import feed_wheat_reserve, sell_qty
from farm_agent.world.state import WorldModel


class MarketAdvisor(BaseAdvisor):

    def get_requests(self, world: WorldModel) -> Tuple[List[TaskRequest], List[MarketRequest]]:
        market_orders: List[MarketRequest] = []
        tc = world.classified_tiles
        shed_total = sum(world.shed.values())

        # The shared feed reserve: every subsystem sells above it and buys up
        # to it, so no two of them trade the same wheat back and forth.
        wheat_reserve = feed_wheat_reserve(tc.animal_count, len(tc.unoccupied_structure), world.day)
        fert_use_reserve = len(tc.apply_fert)

        # Sell everything except wheat (WheatLiquidityAdvisor's job), keeping
        # the fertilizer today's applications need.
        for item in PRODUCTS:
            if item == "WHEAT":
                continue
            qty = world.shed.get(item, 0)
            if item == "FERTILIZER" and fert_use_reserve:
                qty = max(0, qty - fert_use_reserve)
            n = sell_qty(item, qty, world.market_inv.get(item, MARKET_I0), shed_total)
            if n > 0:
                market_orders.append(MarketRequest(order=["SELL", item, n], priority=PRIO_MKT_SELL,
                                                   category="sell"))

        # Feed top-up. Bought wheat can be picked up only next turn, so buy
        # early whenever an animal that will escape tonight is not covered.
        carried_wheat = sum(inv.get("WHEAT", 0) for inv in world.inventories)
        wheat_available = world.shed.get("WHEAT", 0) + carried_wheat
        urgent_unfed = any(
            isinstance(world.tiles[y][x], dict)
            and world.tiles[y][x].get("consecutive_unfed", 0) >= 1
            for x, y in tc.feed
        )
        buy_qty = 0
        if wheat_available < wheat_reserve * WHEAT_BUY_THRESHOLD_RATIO:
            buy_qty = wheat_reserve - wheat_available
        elif urgent_unfed and wheat_available < len(tc.feed) and wheat_reserve > 0:
            # A zero reserve means the season ends tonight: an escape then costs nothing.
            buy_qty = len(tc.feed) - wheat_available
        if buy_qty > 0 and world.money >= MONEY_RESERVE:
            market_orders.append(MarketRequest(
                order=["BUY_PRODUCT", "WHEAT", buy_qty], priority=PRIO_MKT_FEED_TOPUP,
                category="feed_topup", cost_estimate=buy_qty * world.prices.get("WHEAT", 0)))

        return [], market_orders
