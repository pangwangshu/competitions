"""Market model: the engine's price curves, town demand and supply projections."""

import math
from typing import Any, Dict, List, Optional, Tuple

from farm_agent.config import (
    ANIMALS,
    CROPS,
    FEED_RESERVE_PER_ANIMAL,
    GLUT_SENSITIVE,
    HINGE_GAIN,
    MARKET_I0,
    MARKET_PARAMS,
    MAX_SHOP_INSTANCES,
    OPPONENT_HERD_DISCOUNT,
    PRICE_FLOOR,
    SEASON_DAYS,
    SELL_MIN_PRICE_RATIO,
    SELL_MIN_PRICE_RATIO_OVERRIDES,
    SHED_HIGH_WATERMARK,
    SHOP_SELLS_PER_DAY,
    SHOP_UNLOCK_INTERVAL_DAYS,
    SHOPS,
    SUPPLY_HORIZON_DAYS,
)

# ------------------------------------------------------------ price curves


def shape(func: str, x: float, T: Optional[float] = None) -> float:
    """Shape function of the price curve on one side of the reference inventory."""
    x = max(0.0, float(x))
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item: str, inventory: float) -> int:
    """Sale price of one unit at the given market inventory (engine formula)."""
    p = MARKET_PARAMS[item]
    base, T = p["base"], p["T"]
    if inventory < MARKET_I0:
        f = p["below_func"]
        amp = p["below_target"] * base / shape(f, T, T)
        price = base + amp * shape(f, MARKET_I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / shape(f, T, T)
        price = base - amp * shape(f, inventory - MARKET_I0, T)
    return max(PRICE_FLOOR, round(price))


def sell_qty(item: str, qty: int, inv0: float, shed_total: int) -> int:
    """How many of `qty` units to sell now without pushing a glut-sensitive
    product below its price floor. A near-full shed overrides the throttle."""
    if qty <= 0:
        return 0
    if item not in GLUT_SENSITIVE or shed_total >= SHED_HIGH_WATERMARK:
        return qty
    ratio = SELL_MIN_PRICE_RATIO_OVERRIDES.get(item, SELL_MIN_PRICE_RATIO)
    threshold = max(PRICE_FLOOR + 1, ratio * MARKET_PARAMS[item]["base"])
    inv, n = inv0, 0
    for _ in range(qty):
        price = market_price(item, inv)
        if price < threshold:
            break
        n += 1
        if price > PRICE_FLOOR:  # a sale at the $1 floor adds no inventory
            inv += 1
    return n


def fib(n: int) -> int:
    """Wage of the (n+1)-th hire of the day: 1, 1, 2, 3, 5, 8, ..."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# ------------------------------------------------------------- feed wheat


def feed_wheat_reserve(animal_count: int, unoccupied_structures: int, day: int) -> int:
    """Wheat to keep in the shed for feeding, shared by every buyer and seller.

    One definition matters: when the buyer (feed top-up) and the sellers used
    different reserves, the buyer bought back what the sellers had just sold,
    every turn. The reserve tapers with the days left to feed, so it is zero on
    the final day and the endgame liquidates all of it.
    """
    fed_structures = min(animal_count + unoccupied_structures, animal_count + 3)
    if fed_structures <= 0:
        return 0
    days_left = max(0, SEASON_DAYS - 1 - day)
    return min(FEED_RESERVE_PER_ANIMAL, days_left) * fed_structures


def wheat_self_sufficiency_tiles(animal_count: int, unoccupied_structures: int) -> int:
    """Growing wheat tiles needed to feed the herd from our own harvest.

    Wheat's thin per-tile margin rarely wins the crop ranking, so feed wheat is
    reserved explicitly in the rotation rather than left to compete on score.
    """
    fed_structures = min(animal_count + unoccupied_structures, animal_count + 3)
    if fed_structures <= 0:
        return 0
    wheat_yield_per_tile_per_day = CROP_PROFILE["WHEAT"]["total_yield"] / CROP_PROFILE["WHEAT"]["cycle_days"]
    return math.ceil(fed_structures / wheat_yield_per_tile_per_day)


# ------------------------------------------------------------ crop profiles


def optimal_harvest_age(crop: str) -> Tuple[int, int]:
    """(age, yield) at which daily watering stops adding yield to a one-time crop."""
    cd = CROPS[crop]
    max_yield_day, max_yield, first = cd["max_yield_day"], cd["max_yield"], cd["first_yield_day"]
    window_start = (max_yield_day + 1) // 2
    yield_units = 1
    for age in range(1, max_yield_day + 1):
        if window_start <= age <= max_yield_day:
            yield_units = min(max_yield, yield_units + 1)
        if age >= first and yield_units >= max_yield:
            return age, yield_units
    return max_yield_day, yield_units


OPTIMAL_HARVEST = {c: optimal_harvest_age(c) for c in CROPS if not CROPS[c]["ongoing"]}


def crop_profile(crop: str) -> Dict[str, Any]:
    """Cycle length, total yield and seed cost of one planting."""
    cd = CROPS[crop]
    if cd["ongoing"]:
        cycle_days = cd["first_yield_day"] + (cd["max_yield"] - 1) * cd["interval"]
        total_yield = cd["max_yield"]
    else:
        cycle_days, total_yield = OPTIMAL_HARVEST[crop]
    return {"cycle_days": max(1, cycle_days), "total_yield": total_yield, "seed_cost": cd["seed"]}


CROP_PROFILE = {c: crop_profile(c) for c in CROPS}


# ------------------------------------------------------- demand and supply


def daily_demand(product: str, unlocked_shops: List[str]) -> int:
    """Units the town consumes per day from the shops revealed so far."""
    d = 0 if product == "FERTILIZER" else 1  # town centre
    for shop in unlocked_shops:
        menu = SHOPS.get(shop, [])
        if product in menu:
            d += SHOP_SELLS_PER_DAY * (2 if len(menu) == 1 else 1)
    return d


def expected_daily_demand(product: str, unlocked_shops: List[str], day: int) -> float:
    """Current demand plus a discounted share of shops due to unlock soon."""
    slots_left = max(0, MAX_SHOP_INSTANCES - len(unlocked_shops))
    unlock_days = sum(
        1 for d in range(day + 1, day + 1 + SUPPLY_HORIZON_DAYS)
        if d % SHOP_UNLOCK_INTERVAL_DAYS == 0
    )
    expected_new = min(slots_left, unlock_days)
    per_draw = sum(
        SHOP_SELLS_PER_DAY * (2 if len(m) == 1 else 1)
        for m in SHOPS.values() if product in m
    ) / len(SHOPS)
    return daily_demand(product, unlocked_shops) + expected_new * per_draw * 0.5


def farm_weight(idx: int, own_idx: int) -> float:
    return 1.0 if idx == own_idx else OPPONENT_HERD_DISCOUNT


def product_supply_rate(farms: List[Dict[str, Any]], product: str, own_idx: int) -> float:
    """Units per day of an animal product (or fertilizer) from both standing herds."""
    rate = 0.0
    for idx, farm in enumerate(farms):
        weight = farm_weight(idx, own_idx)
        for row in farm.get("tiles", []):
            for t in row:
                if (isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE")
                        and t.get("animal")):
                    a = ANIMALS[t["animal"]]
                    if product == "FERTILIZER":
                        rate += weight
                    elif a["product"] == product:
                        rate += weight * min(a["max_held"], 1 + a["interval"]) / a["interval"]
    return rate


def projected_price(
    product: str,
    farms: List[Dict[str, Any]],
    market_inventory: Dict[str, int],
    unlocked_shops: List[str],
    day: int,
    extra_rate: float,
    own_idx: int,
) -> int:
    """Price after SUPPLY_HORIZON_DAYS of net excess supply, including `extra_rate`."""
    supply = product_supply_rate(farms, product, own_idx) + extra_rate
    demand = expected_daily_demand(product, unlocked_shops, day)
    excess = max(0.0, supply - demand)
    inv = market_inventory.get(product, MARKET_I0)
    return market_price(product, inv + excess * SUPPLY_HORIZON_DAYS)


def crop_supply_rate(farms: List[Dict[str, Any]], crop: str, own_idx: int) -> float:
    """Units per day of a crop from both farms' standing plantings."""
    per_tile_rate = CROP_PROFILE[crop]["total_yield"] / CROP_PROFILE[crop]["cycle_days"]
    rate = 0.0
    for idx, farm in enumerate(farms):
        weight = farm_weight(idx, own_idx)
        for row in farm.get("tiles", []):
            for t in row:
                if isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == crop:
                    rate += weight * per_tile_rate
    return rate
