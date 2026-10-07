"""Advisor interface."""

from typing import List, Tuple

from farm_agent.messages import MarketRequest, TaskRequest
from farm_agent.world.state import WorldModel


class BaseAdvisor:
    """Advisors read the world and propose work; they never act or veto."""

    def get_requests(self, world: WorldModel) -> Tuple[List[TaskRequest], List[MarketRequest]]:
        return [], []
