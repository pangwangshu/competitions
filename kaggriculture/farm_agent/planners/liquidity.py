"""Early intraday cash: bank carried milk and melons the same day on days 7-11.

Harvests normally reach the market only after the overnight sweep into the
shed, so goods picked up on day d are sold on day d+1 and every purchase they
fund slips a day. In the cash-constrained opening that delay compounds. This
planner sends a carrier home early only when all of the following hold:

1. Funding gap. Re-running the allocator with the carried value added to cash
   approves at least MIN_GAP more spending than without it. Because it is the
   allocator's own arbitration, every cap, reserve and ledger still applies.
2. Time and price. The deposit lands by LAST_DEPOSIT_HOUR, and selling today
   instead of at tomorrow's first market loses at most MAX_PRICE_LOSS of the
   revenue to town demand still due today.
3. Labour. After the round trips, today's remaining unit-turns still cover
   TURNS_PER_TASK per pending chore.

Deposits use `PLACE <item> <n>`, which is clamped to shed room and keeps any
remainder in hand. The placed quantity is added to `world.shed` so this turn's
market plan can sell it: unit actions settle before market orders.
"""

from farm_agent.config import (
    ANIMALS,
    CROPS,
    FARM_HAND_COST_MULT,
    LAND_PRICES,
    MARKET_I0,
    SHED_CAPACITY,
    SHOPS,
    TURNS_PER_DAY,
)
from farm_agent.dispatcher.pathfinding import manhattan_dist, step_toward
from farm_agent.world.pricing import fib, market_price, sell_qty
from farm_agent.world.state import WorldModel

DAYS = (7, 8, 9, 10, 11)
ITEMS = ("MILK", "MELON")
LAST_DEPOSIT_HOUR = 18
MIN_VALUE = 300
MIN_GAP = 100
MAX_PRICE_LOSS = 0.10
TURNS_PER_TASK = 2
MAX_RETURNERS = 2


def town_demand(item, shops, start, stop):
    """Units the town removes from the market at steps start <= s < stop."""
    per_tick = sum(2 if len(SHOPS[s]) == 1 else 1 for s in shops if item in SHOPS[s])
    n = 0
    for s in range(start, stop):
        if s % 4 == 0:
            n += per_tick
        if s % TURNS_PER_DAY == 0:
            n += 1
    return n


def revenue(item, inv, n):
    """Proceeds of n units sold one at a time from inventory inv."""
    total = 0
    for _ in range(n):
        p = market_price(item, inv)
        total += p
        if p > 1:
            inv += 1
    return total


def cash_value(world, goods):
    """Sellable value of {item: n} under the market advisor's glut throttle."""
    shed_total = sum(world.shed.values())
    total = 0
    for item, n in goods.items():
        inv = world.market_inv.get(item, MARKET_I0)
        total += revenue(item, inv, sell_qty(item, n, inv, shed_total))
    return total


def merged(goods_list):
    out = {}
    for goods in goods_list:
        for item, n in goods.items():
            out[item] = out.get(item, 0) + n
    return out


def spend(world, tasks, orders):
    """Cash a resolved plan commits: its orders plus cash-gated structure builds."""
    total, hires = 0, world.my_farm.get("hires_today", 0)
    extra_land = len(world.my_farm.get("unlocked_quadrants", ["NW"])) - 1
    for order in orders:
        op = order[0]
        if op == "BUY_SEED":
            total += CROPS[order[1]]["seed"] * int(order[2])
        elif op == "BUY_ANIMAL":
            total += ANIMALS[order[1]]["cost"] * int(order[2])
        elif op == "BUY_PRODUCT":
            total += world.prices.get(order[1], 0) * int(order[2])
        elif op == "BUY_LAND" and extra_land < len(LAND_PRICES):
            total += LAND_PRICES[extra_land]
        elif op == "HIRE":
            total += FARM_HAND_COST_MULT * fib(hires)
            hires += 1
    for task in tasks:
        if task.action and task.action[0] in ("BUILD_COOP", "BUILD_PASTURE"):
            total += ANIMALS[task.payload["animal"]]["cost"]
    return total


def planned_spend(world, allocator, extra_cash):
    """What the allocator would commit this turn holding `extra_cash` more."""
    obs = dict(world.obs)
    farms = list(world.farms)
    farms[world.player_idx] = dict(world.my_farm, money=world.money + extra_cash)
    obs["farms"] = farms
    shadow = WorldModel(obs)
    tasks, orders = allocator.resolve(shadow)
    return spend(shadow, tasks, orders)


def labor_slack(world, diverted_turns):
    tc = world.classified_tiles
    pending = len(tc.water) + len(tc.feed) + len(tc.care) + len(tc.fert) + len(tc.harvest)
    turns_left = TURNS_PER_DAY - world.hour
    return len(world.units) * turns_left - diverted_turns - TURNS_PER_TASK * pending


def price_loss(world, goods, step, deposit_step):
    """Share of today's proceeds that waiting for tomorrow's first market would add."""
    now = later = 0
    next_market = (world.day + 1) * TURNS_PER_DAY
    shed_total = sum(world.shed.values())
    for item, n in goods.items():
        inv = world.market_inv.get(item, MARKET_I0)
        inv -= town_demand(item, world.unlocked_shops, step, deposit_step)
        k = sell_qty(item, n, inv, shed_total)
        now += revenue(item, inv, k)
        inv -= town_demand(item, world.unlocked_shops, deposit_step, next_market)
        later += revenue(item, inv, k)
    return (later - now) / max(1, now)


class LiquidityPlanner:
    def __init__(self):
        self.day = None
        self.committed = {}  # unit idx -> step the commitment was made

    def plan(self, world, allocator):
        """Pinned unit actions for this turn; may add this turn's deposits to world.shed."""
        if self.day != world.day:
            self.day, self.committed = world.day, {}
        if world.day not in DAYS:
            return {}
        step = world.day * TURNS_PER_DAY + world.hour
        units = dict(world.units)
        goods = {}
        for idx in units:
            inv = world.inventories[idx] if idx < len(world.inventories) else {}
            goods[idx] = {i: inv[i] for i in ITEMS if inv.get(i, 0) > 0}

        def dist(idx):
            pos = units[idx]
            return manhattan_dist(pos, world.nearest_shed_tile(pos[0], pos[1]))

        # Drop finished commitments; abandon all of them if chores would go undone.
        for idx in list(self.committed):
            if idx not in units or not goods[idx]:
                self.committed.pop(idx)
        diverted = sum(2 * dist(i) + 1 for i in self.committed)
        if self.committed and labor_slack(world, diverted) < 0:
            self.committed, diverted = {}, 0

        room = SHED_CAPACITY - sum(world.shed.values())
        base_value = cash_value(world, merged(goods[i] for i in self.committed))
        base_spend = None
        candidates = sorted((i for i in units if goods[i] and i not in self.committed),
                            key=lambda i: -cash_value(world, goods[i]) / (2 * dist(i) + 1))
        for idx in candidates:
            d = dist(idx)
            value = cash_value(world, goods[idx])
            if len(self.committed) >= MAX_RETURNERS:
                continue
            if world.hour + d > LAST_DEPOSIT_HOUR or value < MIN_VALUE or room <= 0:
                continue
            if round(price_loss(world, goods[idx], step, step + d), 4) > MAX_PRICE_LOSS:
                continue
            if labor_slack(world, diverted + 2 * d + 1) < 0:
                continue
            if base_spend is None:
                base_spend = planned_spend(world, allocator, base_value)
            gap = planned_spend(world, allocator, base_value + value) - base_spend
            if gap < MIN_GAP:
                continue
            self.committed[idx] = step
            diverted += 2 * d + 1
            base_value += value
            base_spend += gap

        actions, incoming = {}, {}
        for idx in sorted(self.committed):
            pos = units[idx]
            if world.is_shed_adjacent(pos):
                item = max(goods[idx], key=lambda i: (goods[idx][i] * world.prices.get(i, 0), i))
                n = goods[idx][item]
                take = min(n, max(0, room))
                room -= take
                actions[idx] = ["PLACE", item, n]
                if take:
                    incoming[item] = incoming.get(item, 0) + take
            else:
                tx, ty = world.nearest_shed_tile(pos[0], pos[1])
                actions[idx] = [step_toward(pos[0], pos[1], tx, ty)]
        if incoming:
            shed = dict(world.shed)
            for item, n in incoming.items():
                shed[item] = shed.get(item, 0) + n
            world.shed = shed
        return actions
