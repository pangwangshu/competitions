"""Wheat as a cash-flow product, not only feed.

Winning opponents sold hundreds of units of wheat a season while early
versions of this agent sold a handful: the feed reserve locked the most liquid
staple out of revenue. Wheat is on five of eight shop menus, has the gentlest
price curve, and can be bought back when short, so surplus above the shared
feed reserve is sold in steady batches.
"""

from typing import List, Tuple

from farm_agent.advisors.base import BaseAdvisor
from farm_agent.config import (
    MARKET_I0,
    MARKET_PARAMS,
    PRIO_MKT_SELL,
    SHED_HIGH_WATERMARK,
    WHEAT_SELL_MIN_PRICE_RATIO,
    WHEAT_SURPLUS_BATCH_MAX,
)
from farm_agent.messages import MarketRequest, TaskRequest
from farm_agent.world.pricing import feed_wheat_reserve, market_price
from farm_agent.world.state import WorldModel


class WheatLiquidityAdvisor(BaseAdvisor):

    def get_requests(self, world: WorldModel) -> Tuple[List[TaskRequest], List[MarketRequest]]:
        tc = world.classified_tiles
        wheat_in_shed = world.shed.get("WHEAT", 0)
        if wheat_in_shed <= 0:
            return [], []

        # Sell only above the shared reserve plus a 2-unit deadband. The top-up
        # buys *to* the reserve, so the floor must sit strictly above it or the
        # two would trade the same units back and forth.
        reserve = feed_wheat_reserve(tc.animal_count, len(tc.unoccupied_structure), world.day)
        if reserve:
            reserve += 2
        surplus = wheat_in_shed - reserve
        if surplus <= 0:
            return [], []

        # Hold rather than dump into a badly glutted market.
        price = market_price("WHEAT", world.market_inv.get("WHEAT", MARKET_I0))
        if price < WHEAT_SELL_MIN_PRICE_RATIO * MARKET_PARAMS["WHEAT"]["base"]:
            return [], []

        # The batch cap is wheat's throttle; a nearly full shed overrides it.
        batch = min(surplus, WHEAT_SURPLUS_BATCH_MAX)
        if sum(world.shed.values()) >= SHED_HIGH_WATERMARK:
            batch = surplus
        if batch <= 0:
            return [], []
        return [], [MarketRequest(order=["SELL", "WHEAT", batch], priority=PRIO_MKT_SELL, category="sell")]
