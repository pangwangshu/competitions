"""Zero-labour fixes for the overnight shed pour.

On the last turn of each day the engine settles unit actions, then market
orders, then empties every unit's inventory into the shed up to its capacity of
100, destroying whatever does not fit. Before these fixes the pour destroyed
about $5k of goods a game; afterwards about $0.7k (report/paper.md §8.2).

Room sell     On the last turn of a pour day, sell the shed's wheat, then its
              fertilizer, up to the units the pour would otherwise destroy.
              They are the only goods the market sells back, and a buy/sell
              round trip against an unchanged market nets zero, so emptying
              them costs nothing. Market orders settle before the pour.
Defer harvest On a day whose pour is projected to overflow, leave animal
              products on their tile instead of harvesting them, provided
              tonight's production cannot be capped by doing so.

Neither fix sends a unit anywhere.
"""

from collections import Counter

from farm_agent.config import ANIMALS, MAX_MARKET_ORDERS, SHED_CAPACITY, TURNS_PER_DAY

BUYABLE = ("WHEAT", "FERTILIZER")
# The season stops after hour 22 of day 29, so its last pour is day 28's.
LAST_POUR_DAY = 28
# A product left on day d is harvested on d+1 at the earliest and sold on d+2.
LAST_DEFER_DAY = 27


def _load(inv):
    return sum(n for n in inv.values() if n > 0)


def _counter(d):
    return Counter({k: v for k, v in (d or {}).items() if v > 0})


def after_units(world, unit_actions):
    """(shed, carried) after this turn's unit actions, applied in engine order."""
    shed = _counter(world.shed)
    carried = []
    for (idx, pos), act in zip(world.units, unit_actions):
        inv = _counter(world.inventories[idx]) if idx < len(world.inventories) else Counter()
        op = act[0] if act else "PASS"
        x, y = pos
        tile = world.tiles[y][x] if 0 <= y < len(world.tiles) and 0 <= x < len(world.tiles[y]) else None
        is_animal = isinstance(tile, dict) and bool(tile.get("animal"))
        adjacent = world.is_shed_adjacent(pos)
        if op == "HARVEST" and isinstance(tile, dict) and tile.get("yield_units", 0) > 0:
            item = ANIMALS[tile["animal"]]["product"] if is_animal else tile.get("crop")
            if item:
                inv[item] += tile["yield_units"]
        elif op == "COLLECT_FERTILIZER" and is_animal and tile.get("fertilizer_available"):
            inv["FERTILIZER"] += 1
        elif op == "FEED" and is_animal and not tile.get("fed_today") and inv["WHEAT"] > 0:
            inv["WHEAT"] -= 1
        elif op == "FERTILIZE" and isinstance(tile, dict) and tile.get("kind") == "PLANT" \
                and inv["FERTILIZER"] > 0:
            inv["FERTILIZER"] -= 1
        elif op == "PICKUP" and adjacent and len(act) >= 2:
            n = min(int(act[2]) if len(act) >= 3 else 1, shed[act[1]])
            if n > 0:
                shed[act[1]] -= n
                inv[act[1]] += n
        elif op == "PLACE" and len(act) >= 2:
            item = act[1]
            if item in ANIMALS and isinstance(tile, dict) \
                    and tile.get("kind") == ANIMALS[item]["structure"] and not tile.get("animal"):
                if inv[item] > 0:
                    inv[item] -= 1
            elif adjacent:
                n = min(int(act[2]) if len(act) >= 3 else 1, inv[item],
                        max(0, SHED_CAPACITY - _load(shed)))
                if n > 0:
                    inv[item] -= n
                    shed[item] += n
        elif op == "DROP" and adjacent:
            for item, n in list(inv.items()):
                take = min(n, max(0, SHED_CAPACITY - _load(shed)))
                shed[item] += take
            inv = Counter()
        carried.append(+inv)
    return +shed, carried


def room_orders(world, unit_actions, orders):
    """Last turn of a pour day: add SELLs of buyable shed stock to cover the overflow."""
    if world.hour != TURNS_PER_DAY - 1 or world.day > LAST_POUR_DAY:
        return orders
    shed, carried = after_units(world, unit_actions)
    for order in orders:  # the market settles before the pour
        op = order[0]
        if op == "SELL" and len(order) >= 3:
            shed[order[1]] -= min(int(order[2]), shed[order[1]])
        elif op in ("BUY_PRODUCT", "BUY_ANIMAL") and len(order) >= 3:
            shed[order[1]] += min(int(order[2]), max(0, SHED_CAPACITY - _load(shed)))
    over = _load(shed) + sum(_load(inv) for inv in carried) - SHED_CAPACITY
    if over <= 0:
        return orders
    orders = [list(o) for o in orders]
    for item in BUYABLE:
        n = min(over, shed[item])
        if n <= 0:
            continue
        existing = next((o for o in orders if o[0] == "SELL" and o[1] == item), None)
        if existing is not None:
            existing[2] = int(existing[2]) + n
        elif len(orders) < MAX_MARKET_ORDERS:
            orders.append(["SELL", item, n])
        else:
            continue
        over -= n
    return orders


def _tonight(tile, day):
    """Units tonight's refresh adds to an animal tile, before the max_held cap."""
    a = ANIMALS[tile["animal"]]
    k = day + 1 - tile.get("placed_day", day) - a["first_yield_day"]
    if k < 0 or k % a["interval"]:
        return 0
    return 1 + tile.get("pending_care_bonus", 0)  # upper bound: the bonus needs feeding


def deferrable(tile, day):
    """Leaving this tile unharvested tonight loses no production and risks no escape."""
    if tile.get("consecutive_unfed", 0) >= 1:
        return False  # an escape tonight would take the held goods with it
    n = _tonight(tile, day)
    return n == 0 or tile.get("yield_units", 0) + n <= ANIMALS[tile["animal"]]["max_held"]


def projected_overflow(world, tc):
    """Units tonight's pour would destroy if every open harvest and collection is served.

    Buyable stock is sold at hour 23, so nothing in the shed is counted as kept.
    """
    carried = Counter()
    for inv in world.inventories:
        carried.update(_counter(inv))
    intake = len(tc.fert)
    for x, y in tc.harvest:
        tile = world.tiles[y][x]
        intake += tile.get("yield_units", 0) if isinstance(tile, dict) else 0
    wheat_used = min(len(tc.feed), carried["WHEAT"])
    fert_used = min(len(tc.apply_fert), carried["FERTILIZER"])
    return _load(carried) + intake - wheat_used - fert_used - SHED_CAPACITY


def deferred_harvests(world, tc):
    """Animal harvest tiles to leave standing today, largest holdings first."""
    if world.day > LAST_DEFER_DAY:
        return set()
    over = projected_overflow(world, tc)
    if over <= 0:
        return set()
    cands = []
    for x, y in tc.harvest:
        tile = world.tiles[y][x]
        if isinstance(tile, dict) and tile.get("animal") and deferrable(tile, world.day):
            cands.append((-tile.get("yield_units", 0), x, y))
    out = set()
    for neg, x, y in sorted(cands):
        if over <= 0:
            break
        out.add((x, y))
        over += neg
    return out
