"""FarmAgent: one player's decision pipeline, called once per turn.

    observation -> WorldModel -> advisors -> allocator -> dispatcher -> action

All cross-turn state (dispatcher commitments, the liquidity planner, the
market belief) lives on the instance, so two agents in one process do not
interfere.
"""

from typing import Any, Dict

from farm_agent.allocator import ResourceAllocator
from farm_agent.config import MAX_MARKET_ORDERS, SEASON_DAYS, SHED_CAPACITY
from farm_agent.dispatcher.dispatcher import Dispatcher
from farm_agent.planners.liquidity import LiquidityPlanner
from farm_agent.planners.sell_order import reorder
from farm_agent.planners.shed_room import room_orders
from farm_agent.world.state import WorldModel

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


class FarmAgent:

    def __init__(self):
        self.allocator = ResourceAllocator()
        self.dispatcher = Dispatcher()
        self.liquidity = LiquidityPlanner()

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        farms = obs.get("farms", [])
        if not farms or obs.get("player", 0) >= len(farms):
            return dict(PASS_ACTION)

        world = WorldModel(obs)
        self.allocator.late_wheat.observe(obs)

        # Carriers sent home to bank cash early; may add their deposits to world.shed.
        pinned = self.liquidity.plan(world, self.allocator)
        tasks, market_orders = self.allocator.resolve(world)
        farmer_action, hand_actions = self.dispatcher.dispatch(world, tasks, pinned)
        unit_actions = [farmer_action] + hand_actions

        if world.day == SEASON_DAYS - 1:
            market_orders = self._sell_final_drops(world, unit_actions, market_orders)
        market_orders = room_orders(world, unit_actions, market_orders)
        market_orders = reorder(world, market_orders)

        action = {"farmer": farmer_action, "hands": hand_actions, "market": market_orders}
        self.allocator.late_wheat.record(action)
        return action

    @staticmethod
    def _sell_final_drops(world, unit_actions, market_orders):
        """On the last day, sell goods dropped this turn in this turn's market phase
        (unit actions settle before market orders)."""
        additions = {}
        room = max(0, SHED_CAPACITY - sum(world.shed.values()))
        for (idx, pos), action in zip(world.units, unit_actions):
            if action != ["DROP"] or not world.is_shed_adjacent(pos):
                continue
            for item, n in world.inventories[idx].items():
                take = min(max(0, n), room)
                room -= take
                if item in world.prices:
                    additions[item] = additions.get(item, 0) + take
        for item, n in additions.items():
            if not n:
                continue
            order = next((o for o in market_orders if o[:2] == ["SELL", item]), None)
            if order is not None:
                order[2] += n
            elif len(market_orders) < MAX_MARKET_ORDERS:
                market_orders.append(["SELL", item, n])
        return market_orders
