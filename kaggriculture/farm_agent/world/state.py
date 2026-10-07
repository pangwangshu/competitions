"""WorldModel: the parsed observation plus a per-turn classification of own tiles."""

from typing import Any, Dict, List, Optional, Tuple

from farm_agent.config import CROPS, FERT_CASH_FLOOR, FERT_LEAD_DAYS, FERT_VALUE_MARGIN
from farm_agent.messages import TileClassification
from farm_agent.planners.shed_room import deferred_harvests
from farm_agent.world import forecast
from farm_agent.world.pricing import OPTIMAL_HARVEST, market_price


def water_pays(cd, tile, day):
    """Does watering this plant on `day` change any output?

    Ongoing crop: tonight's refresh reads a production. One-time crop: its age
    is in the bonus window, where each watering adds yield.
    """
    age = day - tile.get("planted_day", day)
    if cd["ongoing"]:
        k = age + 1 - cd["first_yield_day"]
        return k >= 0 and k % cd["interval"] == 0 and k // cd["interval"] < cd["max_yield"]
    return (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]


def rule_skips(cd, tile, day):
    """A plant watered at the last refresh survives one dry day, so its watering
    is skipped on days when watering pays nothing. Freed labour went to harvests
    and fertilizing: same output, far fewer waterings."""
    return not water_pays(cd, tile, day)


def planned_skip(cd, tile, day):
    """True if a plant is one dry day behind only because yesterday's watering
    was deliberately skipped (as opposed to a genuine miss)."""
    return (tile.get("consecutive_unwatered", 0) == 1
            and tile.get("planted_day", day) <= day - 2
            and rule_skips(cd, tile, day - 1))


class WorldModel:
    """The current observation, with fast queries and a cached tile classification."""

    def __init__(self, obs: Dict[str, Any]):
        self.obs = obs
        self.player_idx = obs.get("player", 0)
        self.step = obs.get("step", 0)
        self.day = obs.get("day", 0)
        self.hour = obs.get("hour", 0)

        self.farms = obs.get("farms", [])
        if self.farms and self.player_idx < len(self.farms):
            self.my_farm = self.farms[self.player_idx]
        else:
            self.my_farm = {"money": 0, "tiles": [], "farmer": [0, 0], "hands": []}

        self.money: int = self.my_farm.get("money", 0)
        self.tiles: List[List[Any]] = self.my_farm.get("tiles", [])
        self.board_size: int = len(self.tiles)

        self.private: Dict[str, Any] = obs.get("private") or {}
        self.shed: Dict[str, int] = self.private.get("shed") or {}
        self.seeds: Dict[str, int] = self.private.get("seeds") or {}
        self.inventories: List[Dict[str, int]] = self.private.get("inventories") or [{}]

        self.market: Dict[str, Any] = obs.get("market") or {}
        self.prices: Dict[str, int] = self.market.get("prices") or {}
        self.market_inv: Dict[str, int] = self.market.get("inventory") or {}

        self.town: Dict[str, Any] = obs.get("town") or {}
        self.unlocked_shops: List[str] = self.town.get("unlocked_shops") or []

        # Farmer is unit 0; hands are 1, 2, ...
        self.units: List[Tuple[int, Tuple[int, int]]] = [
            (0, tuple(self.my_farm.get("farmer", [0, 0])))
        ] + [
            (i + 1, tuple(pos))
            for i, pos in enumerate(self.my_farm.get("hands", []))
        ]

        self._classified_tiles: Optional[TileClassification] = None

    @property
    def classified_tiles(self) -> TileClassification:
        if self._classified_tiles is None:
            self._classified_tiles = self._classify()
        return self._classified_tiles

    def _classify(self) -> TileClassification:
        tc = TileClassification()
        for y in range(self.board_size):
            for x in range(self.board_size):
                tile = self.tiles[y][x]
                if tile is None:
                    tc.empty.append((x, y))
                    continue
                if tile == "LOCKED" or not isinstance(tile, dict):
                    continue

                kind = tile.get("kind")
                if kind == "WEED":
                    tc.weed.append((x, y))
                elif kind == "PLANT":
                    self._classify_plant(tc, x, y, tile)
                elif kind in ("COOP", "PASTURE"):
                    if tile.get("animal") is None:
                        tc.unoccupied_structure.append((x, y, kind))
                    else:
                        tc.animal_count += 1
                        if tile.get("yield_units", 0) > 0:
                            tc.harvest.append((x, y))
                        if not tile.get("fed_today"):
                            tc.feed.append((x, y))
                        if not tile.get("cared_today"):
                            tc.care.append((x, y))
                        if tile.get("fertilizer_available"):
                            tc.fert.append((x, y))
        # A fertilizer task no unit can deliver before the day ends would only
        # hold stock off the market and take a dispatcher slot.
        for pos in list(tc.apply_fert):
            if not self._fert_reachable(pos):
                tc.apply_fert.remove(pos)
        # On a night the shed would overflow, animal products wait on the tile.
        tc.deferred = deferred_harvests(self, tc)
        if tc.deferred:
            tc.harvest = [p for p in tc.harvest if p not in tc.deferred]
        return tc

    def _classify_plant(self, tc: TileClassification, x: int, y: int, tile: Dict[str, Any]) -> None:
        crop = tile.get("crop")
        if not crop or crop not in CROPS:
            return
        tc.plant_count += 1
        cd = CROPS[crop]
        age = self.day - tile.get("planted_day", self.day)
        yield_units = tile.get("yield_units", 0)
        can_harvest = False
        if yield_units > 0 and age >= cd["first_yield_day"]:
            # A fertilized one-time crop can hit max yield before its usual
            # optimal age; harvest as soon as it is capped.
            can_harvest = (
                True
                if cd["ongoing"]
                else age >= OPTIMAL_HARVEST[crop][0] or yield_units >= cd["max_yield"]
            )
        spent = cd["ongoing"] and tile.get("max_lifespan_step", -1) != -1

        if can_harvest:
            tc.harvest.append((x, y))
        elif spent:
            pass  # an ongoing crop past its last production, decaying to a weed
        elif not tile.get("watered_today"):
            tc.water.append((x, y))
            if tile.get("consecutive_unwatered", 0) == 0 and rule_skips(cd, tile, self.day):
                tc.water_skip.add((x, y))

        if can_harvest or spent or tile.get("fertilized_until_day", -1) >= self.day:
            return
        if cd["ongoing"]:
            due = self._fert_service_day(tile, crop)
            if due is not None and due <= self.day:
                tc.apply_fert.append((x, y))
        elif self.money >= FERT_CASH_FLOOR:
            # One-time crop: each watering in the bonus window yields +2 instead
            # of +1 while fertilized. A fertilizer applied is one not sold, so
            # require the extra units to out-value it.
            window_start = (cd["max_yield_day"] + 1) // 2
            if window_start <= age <= cd["max_yield_day"] and yield_units < cd["max_yield"]:
                extra = min(
                    3,  # a fertilizer lasts 3 days
                    cd["max_yield_day"] - age + 1,
                    cd["max_yield"] - yield_units,
                )
                crop_price = self.prices.get(crop, 0)
                fert_price = self.prices.get("FERTILIZER", 0)
                if extra * crop_price > FERT_VALUE_MARGIN * fert_price:
                    tc.apply_fert.append((x, y))

    def _fert_service_day(self, tile, crop):
        """The day to bring a fertilizer to this ongoing crop, or None to skip it.

        Picks, within FERT_LEAD_DAYS, the application day that covers the most
        production reads, then checks that the tile will be wet when they are
        read, and that the covered berries at their projected sale price beat
        what the fertilizer itself would fetch.
        """
        planted = tile.get("planted_day", self.day)
        fu = tile.get("fertilized_until_day", -1)
        best_day, best_cov = None, 0
        for ahead in range(0, FERT_LEAD_DAYS + 1):
            cov = forecast.fert_coverage(crop, planted, self.day, fu, self.day + ahead)
            if cov > best_cov:
                best_day, best_cov = self.day + ahead, cov
        if best_day is None:
            return None
        # A dry read pays no bonus, and a plant two dry days from death is not
        # worth stocking.
        if tile.get("consecutive_unwatered", 0) >= 2:
            return None
        sale_day = min(best_day + 1 + forecast.MARKET_LAG_DAYS, forecast.LAST_SELL_DAY)
        path = forecast.inventory_path(crop, self.day, self.farms, self.market_inv,
                                       self.unlocked_shops)
        berry = market_price(crop, path.get(sale_day, forecast.MARKET_I0))
        if best_cov * berry <= forecast.fertilizer_opportunity(
                self.day, self.farms, self.market_inv):
            return None
        return best_day

    def _fert_reachable(self, pos):
        """True if some unit could still fetch a fertilizer and reach `pos` today."""
        if not self.units:
            return False
        carried = max((inv.get("FERTILIZER", 0) for inv in self.inventories), default=0)
        if carried <= 0 and self.shed.get("FERTILIZER", 0) <= 0:
            return False
        turns_left = max(0, self.obs.get("turns_per_day", 24) - self.hour)
        for idx, unit in self.units:
            direct = abs(unit[0] - pos[0]) + abs(unit[1] - pos[1])
            if idx < len(self.inventories) and self.inventories[idx].get("FERTILIZER", 0) > 0:
                if direct <= turns_left:
                    return True
                continue
            if self.shed.get("FERTILIZER", 0) <= 0:
                continue
            shed_tile = self.nearest_shed_tile(unit[0], unit[1])
            trip = (abs(unit[0] - shed_tile[0]) + abs(unit[1] - shed_tile[1])
                    + abs(shed_tile[0] - pos[0]) + abs(shed_tile[1] - pos[1]))
            if trip + 1 <= turns_left:
                return True
        return False

    def shed_access_tiles(self) -> List[Tuple[int, int]]:
        """The 4 centre tiles from which a unit can use the shed."""
        half = self.board_size // 2
        return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]

    def is_shed_adjacent(self, pos: Tuple[int, int]) -> bool:
        return tuple(pos) in self.shed_access_tiles()

    def nearest_shed_tile(self, fx: int, fy: int) -> Tuple[int, int]:
        return min(
            self.shed_access_tiles(),
            key=lambda t: abs(t[0] - fx) + abs(t[1] - fy),
        )
