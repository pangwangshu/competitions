"""Reorder SELL slots so the sales most exposed to rival supply settle first.

The engine settles market orders in queue order. Only the order of existing
SELLs changes; quantities, purchases and their positions do not.
"""

from farm_agent.config import ANIMALS, CROPS, MARKET_I0
from farm_agent.world.pricing import market_price


def rival_supply(world, item):
    """One day of the rival's visible production of `item`, plus goods held on its tiles."""
    n = 0.0
    for seat, farm in enumerate(world.farms):
        if seat == world.player_idx:
            continue
        for row in farm.get("tiles", []):
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                animal = tile.get("animal")
                crop = tile.get("crop")
                if animal:
                    a = ANIMALS[animal]
                    if item == a["product"]:
                        n += min(a["max_held"], 1 + a["interval"]) / a["interval"]
                        n += tile.get("yield_units", 0)
                    elif item == "FERTILIZER":
                        n += 0.45
                elif crop == item:
                    c = CROPS[crop]
                    cycle = c["first_yield_day"] + (c["max_yield"] - 1) * c["interval"] if c["ongoing"] else c["max_yield_day"]
                    n += c["max_yield"] / max(1, cycle)
                    n += tile.get("yield_units", 0)
    return n


def urgency(world, order):
    """(price we lose if the rival sells first, value of the sale) for a SELL order."""
    item, n = order[1], min(order[2], world.shed.get(order[1], 0))
    inv = world.market_inv.get(item, MARKET_I0)
    rival = rival_supply(world, item)
    loss = sum(market_price(item, inv + k) - market_price(item, inv + k + rival) for k in range(n))
    return loss, n * world.prices.get(item, 0)


def reorder(world, orders):
    slots = [i for i, o in enumerate(orders) if o[0] == "SELL"]
    sales = sorted((orders[i] for i in slots), key=lambda o: urgency(world, o), reverse=True)
    result = list(orders)
    for i, order in zip(slots, sales):
        result[i] = order
    return result
