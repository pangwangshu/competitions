"""Dated valuation: what one more planting is worth, sold on the day it lands.

Everything here uses the current observation only: both farms' visible tiles,
the revealed shops and the published market inventory. Strawberries and melons
are priced event by event on their sale days (their curves punish a glut and
their payoffs arrive 10+ days later); the other crops use a single projected
price at the end of their short cycle.
"""

from farm_agent.config import (
    CROPS,
    MARKET_I0,
    MAX_SHOP_INSTANCES,
    REALIZED_YIELD,
    SEASON_DAYS,
    SHOP_SELLS_PER_DAY,
    SHOP_UNLOCK_INTERVAL_DAYS,
    SHOPS,
)
from farm_agent.world.pricing import (
    CROP_PROFILE,
    crop_supply_rate,
    daily_demand,
    expected_daily_demand,
    market_price,
)

# The last day anything can still be turned into money: the season stops during
# day 29, and the end-of-day sweep that banks a backpack never runs that night.
LAST_SELL_DAY = SEASON_DAYS - 1

# Median lag, measured on both seats, from a production refresh to the sale that
# banks it: the wait to harvest, the overnight sweep into the shed, then the
# morning order.
MARKET_LAG_DAYS = 3

# Share of strawberry production days that are both watered and fertilized and
# therefore yield 2 units instead of 1 (measured ~0.75 for both players).
BONUS_RATE = 0.75
REFRESH_UNITS = 1.0 + BONUS_RATE

# Share of an animal's daily fertilizer that reaches the market.
FERT_SELL_SHARE = 0.45

DATED_CROPS = ("STRAWBERRY", "MELON")


def refresh_ages(crop):
    """Ages at which an ongoing crop produces; empty for one-time crops."""
    cd = CROPS[crop]
    if not cd["ongoing"]:
        return ()
    return tuple(cd["first_yield_day"] + i * cd["interval"] for i in range(cd["max_yield"]))


def marginal_events(crop, day):
    """(sale_day, units) a crop planted today would still bank this season."""
    events = []
    ages = refresh_ages(crop)
    if ages:
        for age in ages:
            if day + age <= LAST_SELL_DAY:
                events.append((day + age + MARKET_LAG_DAYS, REFRESH_UNITS))
    else:
        p = CROP_PROFILE[crop]
        if day + p["cycle_days"] <= LAST_SELL_DAY:
            events.append((day + p["cycle_days"] + MARKET_LAG_DAYS, float(p["total_yield"])))
    return [(min(d, LAST_SELL_DAY), n) for d, n in events]


def standing_supply(crop, day, farms):
    """{sale_day: units} that both farms' standing plantings will bring to market.

    Both farms count in full: their berries move our price exactly as much as
    ours do. Future plantings by either side are not modelled.
    """
    supply = {}

    def add(sale_day, units):
        d = min(sale_day, LAST_SELL_DAY)
        if d >= day:
            supply[d] = supply.get(d, 0.0) + units

    ages = refresh_ages(crop)
    cycle = CROP_PROFILE[crop]["cycle_days"]
    total = CROP_PROFILE[crop]["total_yield"]
    for farm in farms:
        for row in farm.get("tiles", []):
            for tile in row:
                if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"
                        and tile.get("crop") == crop):
                    continue
                held = tile.get("yield_units", 0)
                if held > 0:
                    add(day + MARKET_LAG_DAYS, float(held))
                planted = tile.get("planted_day", day)
                if ages:
                    if tile.get("max_lifespan_step", -1) != -1:
                        continue  # every refresh already fired; only `held` is left
                    for age in ages:
                        if planted + age > day:
                            add(planted + age + MARKET_LAG_DAYS, REFRESH_UNITS)
                elif planted + cycle > day:
                    add(planted + cycle + MARKET_LAG_DAYS, max(0.0, total - held))
    return supply


def demand_path(crop, day, unlocked_shops):
    """{day: units} the town removes, counting unrevealed shops at the mean draw.

    One shop unlocks every SHOP_UNLOCK_INTERVAL_DAYS, drawn uniformly with
    replacement, so an unrevealed slot is worth the average menu.
    """
    base = daily_demand(crop, unlocked_shops)
    per_draw = sum(
        SHOP_SELLS_PER_DAY * (2 if len(menu) == 1 else 1)
        for menu in SHOPS.values() if crop in menu
    ) / len(SHOPS)
    slots = max(0, MAX_SHOP_INSTANCES - len(unlocked_shops))
    path, revealed = {}, 0
    for d in range(day, LAST_SELL_DAY + 1):
        path[d] = base + min(slots, revealed) * per_draw
        if d > 0 and d % SHOP_UNLOCK_INTERVAL_DAYS == 0:
            revealed += 1
    return path


def inventory_path(crop, day, farms, market_inventory, unlocked_shops):
    """Projected market inventory of `crop`, day by day to the end of the season."""
    supply = standing_supply(crop, day, farms)
    demand = demand_path(crop, day, unlocked_shops)
    inv = float(market_inventory.get(crop, MARKET_I0))
    path = {}
    for d in range(day, LAST_SELL_DAY + 1):
        inv += supply.get(d, 0.0) - demand.get(d, 0.0)
        path[d] = inv
    return path


def fertilizer_opportunity(day, farms, market_inventory):
    """What one fertilizer kept off the market would really fetch.

    No shop buys fertilizer, so every unit sold is permanent inventory and its
    price only falls. Withholding a unit forgoes the *last* unit of the season's
    stream, priced after both herds' remaining output has landed.
    """
    herd = sum(
        1
        for farm in farms
        for row in farm.get("tiles", [])
        for tile in row
        if isinstance(tile, dict) and tile.get("animal")
    )
    days_left = max(0, LAST_SELL_DAY - day)
    inv = market_inventory.get("FERTILIZER", MARKET_I0) + herd * days_left * FERT_SELL_SHARE
    return market_price("FERTILIZER", inv)


def dated_crop_score(crop, day, farms, market_inventory, unlocked_shops, own_idx):
    """Profit per occupied tile-day of one more planting, priced event by event."""
    events = marginal_events(crop, day)
    if not events:
        return float("-inf")
    path = inventory_path(crop, day, farms, market_inventory, unlocked_shops)
    revenue, extra = 0.0, 0.0
    for sale_day, units in events:
        d = min(max(sale_day, day), LAST_SELL_DAY)
        # Price the lot at its own midpoint so it pays for the impact it makes.
        revenue += units * market_price(crop, path.get(d, float(MARKET_I0)) + extra + units / 2.0)
        extra += units
    cost = float(CROPS[crop]["seed"])
    ages = refresh_ages(crop)
    if ages:
        # One fertilizer covers two production reads, charged at the rate the
        # bonus is actually achieved.
        cost += BONUS_RATE * ((len(events) + 1) // 2) * fertilizer_opportunity(
            day, farms, market_inventory
        )
    occupied = min(CROP_PROFILE[crop]["cycle_days"], LAST_SELL_DAY - day)
    return (revenue - cost) / max(1, occupied)


def crop_score(crop, day, farms, market_inventory, unlocked_shops, own_idx):
    """Profit per tile-day of planting `crop` today; -inf if it cannot pay back."""
    if crop in DATED_CROPS:
        return dated_crop_score(crop, day, farms, market_inventory, unlocked_shops, own_idx)
    p = CROP_PROFILE[crop]
    if day + p["cycle_days"] > SEASON_DAYS:
        return float("-inf")
    supply = crop_supply_rate(farms, crop, own_idx)
    demand = expected_daily_demand(crop, unlocked_shops, day)
    inv = market_inventory.get(crop, MARKET_I0)
    price = market_price(crop, inv + (supply - demand) * p["cycle_days"])
    revenue = REALIZED_YIELD.get(crop, p["total_yield"]) * price
    return (revenue - p["seed_cost"]) / p["cycle_days"]


def fert_coverage(crop, planted_day, day, fertilized_until_day, apply_day):
    """Production reads a fertilizer applied on `apply_day` would newly pay a bonus on.

    An application covers the reads on apply_day..apply_day+2 (the engine sets
    fertilized_until_day = apply_day + 2). Reads already covered, or whose
    output would land after LAST_SELL_DAY, do not count.
    """
    covered = 0
    for age in refresh_ages(crop):
        produce_day = planted_day + age
        read_day = produce_day - 1
        if read_day < apply_day or read_day > apply_day + 2:
            continue
        if produce_day <= day or produce_day > LAST_SELL_DAY:
            continue
        if fertilized_until_day >= read_day:
            continue
        covered += 1
    return covered
