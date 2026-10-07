"""Crop scoring and seed purchasing."""

from typing import List, Tuple

from farm_agent.advisors.base import BaseAdvisor
from farm_agent.config import CROPS, PRIO_MKT_SEED, SEED_BUFFER
from farm_agent.messages import MarketRequest
from farm_agent.world.forecast import crop_score
from farm_agent.world.pricing import wheat_self_sufficiency_tiles
from farm_agent.world.state import WorldModel


class CropAdvisor(BaseAdvisor):

    def evaluate_crops(self, world: WorldModel) -> List[Tuple[str, float]]:
        """(crop, profit per tile-day) for every crop that can still pay back, best first."""
        scored = []
        for crop in CROPS:
            score = crop_score(crop, world.day, world.farms, world.market_inv,
                               world.unlocked_shops, world.player_idx)
            if score > float("-inf"):
                scored.append((crop, score))
        scored.sort(key=lambda x: -x[1])
        return scored

    def seed_requests(self, world: WorldModel, active_crops: List[str], plant_budget: int) -> List[MarketRequest]:
        """One seed per turn per rotated crop below its buffer.

        Nothing is bought while the labour ledger forbids new plantings. Wheat's
        buffer follows the feed self-sufficiency target, since the engine voids
        every PLANT of a crop in a turn whose PLANTs exceed its seed stock.
        """
        if plant_budget <= 0:
            return []
        tc = world.classified_tiles
        wheat_buffer = max(SEED_BUFFER, wheat_self_sufficiency_tiles(tc.animal_count, len(tc.unoccupied_structure)))
        reqs: List[MarketRequest] = []
        for crop in active_crops:
            buffer = wheat_buffer if crop == "WHEAT" else SEED_BUFFER
            if world.seeds.get(crop, 0) < buffer:
                reqs.append(MarketRequest(order=["BUY_SEED", crop, 1], priority=PRIO_MKT_SEED,
                                          category="seed", cost_estimate=CROPS[crop]["seed"]))
        return reqs
