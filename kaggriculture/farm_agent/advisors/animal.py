"""Animal scoring (for the build rotation) and livestock purchases."""

from typing import Any, Dict, List, Tuple

from farm_agent.advisors.base import BaseAdvisor
from farm_agent.messages import MarketRequest, TaskRequest
from farm_agent.world.livestock import buy_requests, planning_farms, score
from farm_agent.world.state import WorldModel


class AnimalAdvisor(BaseAdvisor):

    def evaluate_animals(self, world: WorldModel) -> Dict[str, Any]:
        """Scores of a marginal goose and of the better pasture species."""
        farms = planning_farms(world)
        cow = score(world, farms, "COW")
        sheep = score(world, farms, "SHEEP")
        goose = score(world, farms, "GOOSE")
        pasture_animal = "COW" if cow >= sheep else "SHEEP"
        return {
            "goose_score": goose,
            "pasture_animal": pasture_animal,
            "pasture_score": cow if pasture_animal == "COW" else sheep,
        }

    def get_requests(self, world: WorldModel) -> Tuple[List[TaskRequest], List[MarketRequest]]:
        return [], buy_requests(world)
