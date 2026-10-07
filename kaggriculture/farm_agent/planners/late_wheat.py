"""Late-season wheat on land and labour that would otherwise sit idle.

From ENDGAME_STOP_INVEST_DAY the allocator stops planting, but the hands it
keeps paying for go idle while wheat (on five of the eight shop menus) still
trades near its season high. On days 24-27 this planner plants wheat on empty
tiles while each extra plant is still worth LW_MIN_VALUE after its seed, priced
on the MarketBelief wheat forecast for the day the harvest can be sold, with
the price impact of plants already scheduled included.

Wheat is cheap to buy back, so its marginal value is only the spread; it pays
here precisely because the labour and land have no other use (+$2.9k/game on
a held-out panel; report/paper.md §8.2).
"""

from farm_agent.config import CROPS, PRIO_MKT_SEED, PRIO_PRODUCTION
from farm_agent.dispatcher.pathfinding import manhattan_dist
from farm_agent.messages import MarketRequest, TaskRequest
from farm_agent.world.market_belief import MarketBelief, flow_inventory
from farm_agent.world.pricing import OPTIMAL_HARVEST, market_price
from farm_agent.world.state import planned_skip

LW_FIRST_DAY = 24
LW_LAST_DAY = 27
LW_MAX_STANDING = 40
LW_MIN_VALUE = 30
LAST_HARVEST_DAY = 29


def planned_harvest_day(tile):
    """Day a one-time crop is harvested: its optimal age, or the last day."""
    return min(tile["planted_day"] + OPTIMAL_HARVEST[tile["crop"]][0], LAST_HARVEST_DAY)


def is_late_wheat(tile):
    return (isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT"
            and tile.get("planted_day", -1) >= LW_FIRST_DAY)


class LateWheatPlanner:
    def __init__(self):
        self.belief = MarketBelief()
        self.last_action = None
        self.guard = {}  # day -> True if any plant or animal was in danger at hour 0

    def observe(self, obs):
        """Per-turn update, before any decision."""
        self.belief.update(obs, self.last_action)

    def record(self, action):
        """After deciding: remember our action for next turn's trade accounting."""
        self.last_action = action

    def _guard_blocked(self, world):
        """No late wheat on a day that started with a thirsty plant or a hungry animal."""
        day = world.day
        if world.hour == 0 and day not in self.guard:
            blocked = False
            for x, y in world.classified_tiles.water:
                t = world.tiles[y][x]
                if t.get("consecutive_unwatered", 0) >= 1 and not planned_skip(CROPS[t["crop"]], t, day):
                    blocked = True
            for row in world.tiles:
                for t in row:
                    if isinstance(t, dict) and "animal" in t and t.get("consecutive_unfed", 0) >= 1:
                        blocked = True
            self.guard[day] = blocked
        return self.guard.get(day, False)

    def requests(self, world, claimed_positions):
        """(tasks, market requests) for today's late wheat; empty outside the window."""
        day = world.day
        if not (LW_FIRST_DAY <= day <= LW_LAST_DAY) or self._guard_blocked(world):
            return [], []
        tc = world.classified_tiles
        harvest_day = min(day + 4, LAST_HARVEST_DAY)
        units = harvest_day - day
        if units < 2:
            return [], []
        sale_day = min(harvest_day + 1, LAST_HARVEST_DAY)
        standing = 0
        planned_units = 0.0
        for row in world.tiles:
            for t in row:
                if is_late_wheat(t):
                    standing += 1
                    h = min(t["planted_day"] + 4, LAST_HARVEST_DAY)
                    if min(h + 1, LAST_HARVEST_DAY) <= sale_day:
                        planned_units += max(t.get("yield_units", 1), h - t["planted_day"])
        room = LW_MAX_STANDING - standing
        if room <= 0:
            return [], []
        inv = flow_inventory(self.belief, dict(world.obs), "WHEAT", max(1, sale_day - day))
        empties = [p for p in tc.empty if p not in claimed_positions]
        unit_positions = [pos for _, pos in world.units]
        empties.sort(key=lambda t: min(manhattan_dist(t, p) for p in unit_positions) if unit_positions else 0)
        n = 0
        for _ in range(min(room, len(empties))):
            value = units * market_price("WHEAT", inv + planned_units + units / 2.0) - CROPS["WHEAT"]["seed"]
            if value < LW_MIN_VALUE:
                break
            planned_units += units
            n += 1
        tasks = [TaskRequest(category="late_wheat", pos=pos, priority=PRIO_PRODUCTION,
                             action=["PLANT", "WHEAT"], payload={"crop": "WHEAT"}) for pos in empties[:n]]
        market = []
        need = n - world.seeds.get("WHEAT", 0)
        if need > 0:
            market.append(MarketRequest(order=["BUY_SEED", "WHEAT", need], priority=PRIO_MKT_SEED,
                                        category="late_seed", cost_estimate=CROPS["WHEAT"]["seed"] * need))
        return tasks, market
