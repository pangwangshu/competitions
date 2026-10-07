"""Calendar valuation of a marginal animal, and the purchase plan built on it.

A marginal animal is valued by laying out its production days (care bonuses
included) on the calendar and pricing each lot against the projected market on
its sale day. Two price effects are counted:

- own displacement: our extra output lowers the price our existing herd gets;
- rival displacement, at half weight: it also lowers the price the opponent
  gets. The game is decided by the margin, not by our own cash, so a sale that
  depresses a contested market is worth more than the cash it brings in.
  Counting it shifted the herd from geese toward cows and sheep and was worth
  about +$2.9k/game on a held-out panel (report/paper.md §8.2).
"""

from farm_agent.config import (
    ANIMAL_BUILD_RESERVE,
    ANIMALS,
    MARKET_I0,
    MAX_ANIMAL_BUYS_PER_TURN,
    PAYBACK_MARGIN,
    PRIO_MKT_ANIMAL,
    SEASON_DAYS,
)
from farm_agent.messages import MarketRequest
from farm_agent.world.forecast import demand_path
from farm_agent.world.pricing import market_price, projected_price

CARE_RATE = 0.95        # share of days an animal is actually cared for
SALE_LAG = 1            # days from production to sale
PLACEMENT_LAG = 1       # a purchase is placed the next day at the earliest
RIVAL_PRICE_IMPACT_WEIGHT = 0.5


def events(animal, placed, day, pending=0, held=0):
    """{sale_day: units} an animal will still deliver this season.

    The care bonus banked so far is paid with the next production; production
    whose sale would fall after the season is dropped.
    """
    a = ANIMALS[animal]
    result = {}
    if held and day + SALE_LAG < SEASON_DAYS:
        result[day + SALE_LAG] = float(held)
    bank = float(pending)
    for current in range(day, SEASON_DAYS - 1):
        if current < placed:
            continue
        age = current + 1 - placed
        if age >= a["first_yield_day"] and (age - a["first_yield_day"]) % a["interval"] == 0:
            sale = current + 1 + SALE_LAG
            if sale < SEASON_DAYS:
                result[sale] = result.get(sale, 0.0) + min(a["max_held"], 1.0 + bank)
            bank = 0.0
        bank += CARE_RATE
    return result


def dated_animal_score(animal, day, farms, market_inventory, unlocked_shops, prices, own_idx):
    """Net value per remaining day of buying one more `animal` today; -inf if it cannot pay back."""
    a = ANIMALS[animal]
    marginal = events(animal, day + PLACEMENT_LAG, day)
    if not marginal:
        return float("-inf")
    product = a["product"]
    supply, own, rival = {}, {}, {}
    for seat, farm in enumerate(farms):
        for row in farm.get("tiles", []):
            for tile in row:
                if not isinstance(tile, dict) or tile.get("animal") != animal:
                    continue
                for d, n in events(animal, tile["placed_day"], day,
                                   tile.get("pending_care_bonus", 0), tile.get("yield_units", 0)).items():
                    supply[d] = supply.get(d, 0.0) + n
                    if seat == own_idx:
                        own[d] = own.get(d, 0.0) + n
                    else:
                        rival[d] = rival.get(d, 0.0) + n
    demand = demand_path(product, day, unlocked_shops)
    inventory = float(market_inventory.get(product, MARKET_I0))
    extra = revenue = displacement = rival_loss = 0.0
    for d in range(day, SEASON_DAYS):
        inventory -= demand[d]
        base_units = supply.get(d, 0.0)
        n = marginal.get(d, 0.0)
        # Quote mid-lot so each lot pays for its own price impact.
        quote_inv = inventory + base_units / 2
        price_before = market_price(product, quote_inv)
        price_after = market_price(product, quote_inv + extra + n / 2)
        revenue += n * price_after
        displacement += own.get(d, 0.0) * (price_before - price_after)
        rival_loss += rival.get(d, 0.0) * (price_before - price_after)
        inventory += base_units
        extra += n
    remaining = SEASON_DAYS - day
    fert_price = projected_price("FERTILIZER", farms, market_inventory,
                                 unlocked_shops, day, 1.0, own_idx)
    net = revenue + (fert_price - prices.get("WHEAT", 0)) * remaining
    net -= displacement
    net += RIVAL_PRICE_IMPACT_WEIGHT * rival_loss
    if net < a["cost"] * PAYBACK_MARGIN:
        return float("-inf")
    return (net - a["cost"]) / remaining


# ------------------------------------------------------------ purchase plan


def planning_farms(world):
    """Both farms, with animals we own but have not placed yet added to ours.

    The additions exist only for valuation; the observation is not modified.
    """
    farms = list(world.farms)
    own = dict(farms[world.player_idx])
    own["tiles"] = list(own["tiles"]) + [[]]
    farms[world.player_idx] = own
    for animal in ANIMALS:
        stock = world.shed.get(animal, 0) + sum(i.get(animal, 0) for i in world.inventories)
        for _ in range(stock):
            add_animal(farms, world, animal)
    return farms


def add_animal(farms, world, animal):
    farms[world.player_idx]["tiles"][-1].append(dict(
        kind=ANIMALS[animal]["structure"], animal=animal,
        placed_day=world.day + 1, yield_units=0, pending_care_bonus=0))


def score(world, farms, animal):
    return dated_animal_score(animal, world.day, farms, world.market_inv,
                              world.unlocked_shops, world.prices, world.player_idx)


def buy_requests(world):
    """BUY_ANIMAL orders for free structures, re-valuing after each purchase."""
    farms = planning_farms(world)
    free = {kind: sum(t[2] == kind for t in world.classified_tiles.unoccupied_structure)
            for kind in ("PASTURE", "COOP")}
    for a, data in ANIMALS.items():
        free[data["structure"]] -= world.shed.get(a, 0) + sum(i.get(a, 0) for i in world.inventories)
    budget, counts, requests = world.money, {}, []
    for _ in range(MAX_ANIMAL_BUYS_PER_TURN):
        eligible = [a for a, d in ANIMALS.items() if free[d["structure"]] > 0
                    and budget - d["cost"] >= ANIMAL_BUILD_RESERVE]
        if not eligible:
            break
        values = [(score(world, farms, a), a) for a in eligible]
        value, animal = max(values)
        if value == float("-inf"):
            break
        budget -= ANIMALS[animal]["cost"]
        free[ANIMALS[animal]["structure"]] -= 1
        counts[animal] = counts.get(animal, 0) + 1
        add_animal(farms, world, animal)
    for animal, n in counts.items():
        requests.append(MarketRequest(order=["BUY_ANIMAL", animal, n], priority=PRIO_MKT_ANIMAL,
                                      category="animal_stock", cost_estimate=n * ANIMALS[animal]["cost"]))
    return requests
