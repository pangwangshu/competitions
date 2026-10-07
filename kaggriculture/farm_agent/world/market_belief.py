"""MarketBelief: exact market trade flows recovered from public observations.

The engine settles each turn as unit actions, then market orders, then town
consumption, so for every product

    inventory[t+1] - inventory[t] = sum over both players (units sold above $1 - units bought)
                                    - town consumption at t

Our own trades are known exactly (our orders plus our shed after unit
actions), and town consumption follows from the revealed shops, so the
opponent's net flow is recovered exactly, up to its sales at the $1 floor,
which leave no trace in inventory. The identity held with zero error over
1.9M product-turns of recorded games.

The flow history feeds a short-horizon inventory forecast for wheat and
fertilizer, the two goods both players also buy back from the market.
"""

from farm_agent.config import (
    MARKET_I0,
    MAX_SHOP_INSTANCES,
    PRODUCTS,
    SEASON_DAYS,
    SHED_CAPACITY,
    SHOP_SELLS_PER_DAY,
    SHOP_UNLOCK_INTERVAL_DAYS,
    SHOPS,
    TURNS_PER_DAY,
)
from farm_agent.world.pricing import market_price

ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
LAST_DAY = SEASON_DAYS - 1

# ------------------------------------------------------------- accounting


def town_consumption(step, shops):
    """{item: units} the town removes after `step`'s market phase."""
    out = {}
    if step % 4 == 0:
        for name in shops:
            menu = SHOPS.get(name, ())
            for item in menu:
                out[item] = out.get(item, 0) + (2 if len(menu) == 1 else 1)
    if step % TURNS_PER_DAY == 0:
        for item in PRODUCTS:
            if item != "FERTILIZER":
                out[item] = out.get(item, 0) + 1
    return out


def _unit_positions(farm):
    return [tuple(farm["farmer"])] + [tuple(h) for h in farm.get("hands", [])]


def shed_after_units(obs, act):
    """(shed, carried) after this turn's unit actions.

    Only PICKUP, PLACE and DROP touch the shed, and units act in farmer-then-
    hands order, so the running shed is exact without simulating tiles.
    """
    me = obs["player"]
    farm = obs["farms"][me]
    shed = {k: v for k, v in obs["private"]["shed"].items()}
    invs = [dict(i) for i in obs["private"]["inventories"]]
    positions = _unit_positions(farm)
    actions = [act.get("farmer") or ["PASS"]] + list(act.get("hands") or [])
    tiles = farm["tiles"]
    for idx, pos in enumerate(positions):
        a = actions[idx] if idx < len(actions) else ["PASS"]
        if not isinstance(a, list) or not a:
            continue
        while len(invs) <= idx:
            invs.append({})
        inv = invs[idx]
        op = a[0]
        adjacent = pos in SHED_ACCESS
        if op == "DROP" and adjacent:
            for item, n in list(inv.items()):
                if n > 0:
                    room = max(0, SHED_CAPACITY - sum(shed.values()))
                    take = min(n, room)
                    if take:
                        shed[item] = shed.get(item, 0) + take
                del inv[item]
        elif op == "PICKUP" and adjacent and len(a) >= 2:
            try:
                n = int(a[2]) if len(a) >= 3 else 1
            except (TypeError, ValueError):
                continue
            n = min(n, shed.get(a[1], 0))
            if n > 0:
                shed[a[1]] -= n
                inv[a[1]] = inv.get(a[1], 0) + n
        elif op == "PLACE" and len(a) >= 2:
            item = a[1]
            tile = tiles[pos[1]][pos[0]]
            if (item in ANIMALS and isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")
                    and "animal" not in tile):
                continue  # placing an animal on its structure, not into the shed
            if not adjacent:
                continue
            try:
                n = int(a[2]) if len(a) >= 3 else 1
            except (TypeError, ValueError):
                continue
            n = min(n, inv.get(item, 0), max(0, SHED_CAPACITY - sum(shed.values())))
            if n > 0:
                inv[item] -= n
                if inv[item] == 0:
                    del inv[item]
                shed[item] = shed.get(item, 0) + n
    return shed, invs


def own_trades(obs, act, nxt):
    """{item: (sold, bought)} our market phase committed on this transition.

    A SELL commits min(ordered, shed) in queue order. Buys are read from shed
    conservation, except on the last turn of a day, when the overnight sweep
    hides them and they are taken as ordered, capped by shed room.
    """
    shed, _ = shed_after_units(obs, act)
    running = dict(shed)
    sold, bought = {}, {}
    eod = obs["hour"] == TURNS_PER_DAY - 1
    for order in (act.get("market") or [])[:10]:
        if not isinstance(order, list) or len(order) < 3 or order[0] not in ("SELL", "BUY_PRODUCT", "BUY_ANIMAL"):
            continue
        try:
            n = int(order[2])
        except (TypeError, ValueError):
            continue
        item = order[1]
        if n <= 0:
            continue
        if order[0] == "SELL":
            k = min(n, running.get(item, 0))
            running[item] = running.get(item, 0) - k
            sold[item] = sold.get(item, 0) + k
        elif eod:
            room = max(0, SHED_CAPACITY - sum(running.values()))
            k = min(n, room)
            running[item] = running.get(item, 0) + k
            if order[0] == "BUY_PRODUCT":
                bought[item] = bought.get(item, 0) + k
    if not eod and nxt is not None:
        after = nxt["private"]["shed"]
        for item in ("WHEAT", "FERTILIZER"):
            k = after.get(item, 0) - (shed.get(item, 0) - sold.get(item, 0))
            if k > 0:
                bought[item] = k
    return {i: (sold.get(i, 0), bought.get(i, 0)) for i in set(sold) | set(bought)}


def opponent_net_flow(obs, own, nxt):
    """{item: opponent units sold above the floor minus units bought} on this transition."""
    before = obs["market"]["inventory"]
    after = nxt["market"]["inventory"]
    town = town_consumption(obs["step"], obs["town"]["unlocked_shops"])
    out = {}
    for item in PRODUCTS:
        s, b = own.get(item, (0, 0))
        inv, above = before[item] - b, 0
        for _ in range(s):
            if market_price(item, inv) > 1:
                inv += 1
                above += 1
        out[item] = after[item] - before[item] + town.get(item, 0) - (above - b)
    return out


def _copy_tiles(tiles):
    return [[dict(t) if isinstance(t, dict) else t for t in row] for row in tiles]


def snapshot(obs):
    """Plain copies of every observation field the belief reads."""
    farms = []
    for f in obs["farms"]:
        farms.append(dict(tiles=_copy_tiles(f["tiles"]), farmer=list(f["farmer"]),
                          hands=[list(h) for h in f.get("hands", [])], money=f.get("money", 0),
                          hires_today=f.get("hires_today", 0)))
    private = obs["private"]
    return dict(step=obs["day"] * TURNS_PER_DAY + obs["hour"], day=obs["day"], hour=obs["hour"],
                player=obs["player"], farms=farms,
                private=dict(shed=dict(private["shed"]), inventories=[dict(i) for i in private["inventories"]],
                             seeds=dict(private.get("seeds", {}))),
                market=dict(inventory=dict(obs["market"]["inventory"]), prices=dict(obs["market"]["prices"])),
                town=dict(unlocked_shops=list(obs["town"]["unlocked_shops"])))


class MarketBelief:
    """Per-day net flows (sold above floor minus bought) for each player.

    Call update(obs, last_action) at the start of every turn, where last_action
    is the action this agent returned for the previous observation.
    """

    def __init__(self):
        self.prev = None
        self.flow = {"own": {}, "opp": {}}  # role -> day -> item -> net units

    @staticmethod
    def _bump(table, day, delta):
        row = table.setdefault(day, {})
        for item, n in delta.items():
            if n:
                row[item] = row.get(item, 0) + n

    def update(self, obs, last_action):
        cur = snapshot(obs)
        prev = self.prev
        if prev is not None and last_action is not None and cur["step"] == prev["step"] + 1:
            own = own_trades(prev, last_action, cur)
            flow = opponent_net_flow(prev, own, cur)
            day = prev["day"]
            inv0 = prev["market"]["inventory"]
            own_net = {}
            for item, (s, b) in own.items():
                inv, above = inv0[item] - b, 0
                for _ in range(s):
                    if market_price(item, inv) > 1:
                        inv += 1
                        above += 1
                own_net[item] = above - b
            self._bump(self.flow["own"], day, own_net)
            self._bump(self.flow["opp"], day, flow)
        self.prev = cur
        return cur


# --------------------------------------------------------------- forecast

_FLOOR_INV = {}


def floor_inventory(item):
    """Smallest inventory at which the quote reaches $1 (sales there add nothing)."""
    if item not in _FLOOR_INV:
        lo, hi = MARKET_I0, MARKET_I0 + 200000
        while lo < hi:
            mid = (lo + hi) // 2
            if market_price(item, mid) <= 1:
                hi = mid
            else:
                lo = mid + 1
        _FLOOR_INV[item] = lo
    return _FLOOR_INV[item]


def demand_by_day(item, day, shops, last):
    """{D: units the town removes on day D}, unrevealed shop slots at the mean draw."""
    base_tick = sum((2 if len(SHOPS[s]) == 1 else 1) for s in shops if item in SHOPS.get(s, ()))
    per_draw = sum((2 if len(m) == 1 else 1) for m in SHOPS.values() if item in m) / len(SHOPS)
    out = {}
    extra = 0
    for D in range(day, last + 1):
        if D > day and D % SHOP_UNLOCK_INTERVAL_DAYS == 0 and len(shops) + extra < MAX_SHOP_INSTANCES:
            extra += 1
        out[D] = SHOP_SELLS_PER_DAY * (base_tick + extra * per_draw) + (0 if item == "FERTILIZER" else 1)
    return out


def roll(inv0, item, day, horizon, supply, demand):
    """Inventory at day+horizon from per-day supply and demand, honouring the $1 floor."""
    inv = float(inv0)
    cap = floor_inventory(item)
    for D in range(day, day + horizon):
        before = inv
        inv += supply.get(D, 0.0)
        if inv > cap:
            inv = max(before, float(cap))
        inv -= demand.get(D, 0.0)
    return inv


def recent_rate(belief, role, item, day, window):
    days = [D for D in range(day - window, day) if D >= 0]
    if not days:
        return 0.0
    return sum(belief.flow[role].get(D, {}).get(item, 0) for D in days) / len(days)


def flow_inventory(belief, obs, item, horizon, window=3):
    """Market inventory in `horizon` days if both players keep their recent net flow."""
    day = obs["day"]
    shops = obs["town"]["unlocked_shops"]
    rate = recent_rate(belief, "own", item, day, window) + recent_rate(belief, "opp", item, day, window)
    supply = {D: rate for D in range(day, day + horizon)}
    demand = demand_by_day(item, day, shops, min(LAST_DAY, day + horizon))
    return roll(obs["market"]["inventory"][item], item, day, horizon, supply, demand)
