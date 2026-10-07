"""Game constants, strategy parameters and priority bands.

The single source of truth for every number the agent uses. Advisors and the
allocator read the same priority scale, so the order in which tasks win a tile
and the exchange rate the dispatcher applies between priority and walking
distance are defined once, here.

Rationale for the less obvious values is kept next to them. Measured effects
quoted below are paired offline comparisons; see report/paper.md for method.
"""

# --------------------------------------------------------------------- game

SEASON_DAYS = 30
TURNS_PER_DAY = 24
SHED_CAPACITY = 100

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}
STRUCTURE_ANIMALS = {"COOP": ("GOOSE",), "PASTURE": ("COW", "SHEEP")}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

SHOPS = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
SHOP_SELLS_PER_DAY = TURNS_PER_DAY // 4
SHOP_UNLOCK_INTERVAL_DAYS = 3
MAX_SHOP_INSTANCES = 8

# Price curve per product: `base` at the reference inventory MARKET_I0, rising
# below it and falling above it with a per-side shape. A port of the engine's
# own formula (tests/test_pricing.py checks it against the engine).
MARKET_I0 = 10000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0
MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

# The engine processes at most this many market orders per turn and silently
# drops the rest, so the allocator truncates its priority-sorted queue here.
MAX_MARKET_ORDERS = 10

# ------------------------------------------------------------ land and labour

# Prices of the second and third quadrant. The fourth ($4,000) is never bought:
# farm occupancy plateaus around 73 tiles, so its 25 tiles earned ~$354 a game
# (+$2.8k/game paired margin from dropping it; buying it *earlier* was worse).
LAND_PRICES = [1000, 2000]
LAND_EMPTY_THRESHOLD = 2  # buy the next quadrant once empty tiles fall to this

FARM_HAND_COST_MULT = 1   # engine default: the n-th hire of a day costs fib(n)

# Hiring. Hands vanish every evening and the n-th hire of a day costs fib(n),
# so hiring early in the day is free relative to hiring late. A reactive rule
# ("hire while backlog > 3x headcount") never bootstraps: on day 0 an empty
# farm has no backlog, so the season opens with one unit and locks in near
# seven. HIRE_SCHEDULE is a per-day hand-count floor under that rule (shape
# taken from the hiring curve of a public Kaggle notebook; see the paper §6.2).
HIRE_SCHEDULE = (5, 4, 4, 5, 4, 5, 8, 8, 10, 11, 11, 11, 9, 11, 10,
                 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11)
HIRE_BACKLOG_PER_UNIT = 3  # reactive rule: chores per unit before another hire
HIRE_DAILY_CAP = 8         # cap on the reactive rule only; the schedule may exceed it
HIRE_LAST_HOUR = 12        # a hand hired later cannot do a full day's work
HIRE_CASH_FLOOR = 30       # wages are $1, 1, 2, 3, ...: hire on pocket change
HIRE_BATCH_MAX = 6         # HIRE orders per turn; leaves queue room for seeds

# Labour ledger: daily chores one unit sustains, travel included. New plantings
# are approved only while (units + hands hirable now) x this covers the load.
UNIT_DAILY_TASK_CAPACITY = 12

MONEY_RESERVE = 300        # base of the payroll floor kept before discretionary buys

# --------------------------------------------------------------- production

ROTATION_SIZE = 3          # top-N scored options served round-robin over empty tiles
SEED_BUFFER = 2

MAX_ANIMAL_STRUCTURES = 16
MAX_ANIMAL_BUYS_PER_TURN = 3
ANIMAL_BUILD_RESERVE = 600  # cash kept after building or stocking a structure
PAYBACK_MARGIN = 1.5        # an animal must return 1.5x its price over the rest of the season

# Feed. Animals eat one wheat a day and escape after two missed days.
FEED_RESERVE_PER_ANIMAL = 2
WHEAT_BUY_THRESHOLD_RATIO = 0.5  # top up when stock falls below half the reserve

# Fertilizing one-time crops: a fertilizer applied is one not sold, and cash
# early in the season is better spent on hires and animals.
FERT_VALUE_MARGIN = 1.0
FERT_CASH_FLOOR = 1000

# Days of lead time a strawberry fertilizer task is planned ahead, so that one
# application lands on the day it covers two production reads (ages 9 and 13)
# instead of one.
FERT_LEAD_DAYS = 2

# ------------------------------------------------------------------ market

# Opponent herds count at this weight when projecting supply: their standing
# animals are visible but their future upkeep is not.
OPPONENT_HERD_DISCOUNT = 0.7
SUPPLY_HORIZON_DAYS = 6

# Sell throttle for goods whose price collapses under a glut: stop selling once
# the next unit would clear below this share of base price, unless the shed is
# close to overflowing.
GLUT_SENSITIVE = {p for p, cfg in MARKET_PARAMS.items() if cfg["above_target"] > 1.0}
SELL_MIN_PRICE_RATIO = 0.3
SELL_MIN_PRICE_RATIO_OVERRIDES = {"MELON": 0.5}
SHED_HIGH_WATERMARK = 80

# Wheat is sold as a cash-flow product above the feed reserve, in batches.
WHEAT_SURPLUS_BATCH_MAX = 12
WHEAT_SELL_MIN_PRICE_RATIO = 0.6

# Realised yield per planting used when scoring a crop, where it differs from
# the model's optimal-age yield. Carrots were measured at about 2 units.
REALIZED_YIELD = {"CARROT": 2.0}

# ----------------------------------------------------------------- endgame

ENDGAME_STOP_INVEST_DAY = 24   # stop buying seeds, animals and land
ENDGAME_HARVEST_ALL_DAY = 26   # harvest anything with yield

# The episode ends after hour 22 of day 29 and the last end-of-day sweep never
# runs, so goods still carried are worth nothing. Units carrying sellable goods
# walk home in time to drop them for the final market phase.
TERMINAL_SELL_STEP = SEASON_DAYS * TURNS_PER_DAY - 2
TERMINAL_RETURN_SAFETY_TURNS = 1   # one turn of slack on the walk home
TERMINAL_RETURN_MIN_VALUE = 0.0

# ---------------------------------------------------------- priority bands

# Task priorities. Survival 900-1000, maintenance 200-920, production lowest.
PRIO_FEED_URGENT = 1000.0      # unfed yesterday: escapes tonight
PRIO_FEED = 950.0
PRIO_WATER_URGENT = 1000.0     # dry yesterday: dies tonight
PRIO_WATER = 900.0
PRIO_HARVEST = 850.0
PRIO_CARE = 800.0
PRIO_COLLECT_FERT = 750.0
# A fertilizer due today. Below harvest and watering: a bonus is only paid on a
# tile that was also watered.
PRIO_APPLY_FERT_DUE = 820.0
# A due strawberry/tomato fertilizer, which pays a doubled production that
# expires tonight. Above routine watering: the carrier arrives with the
# fertilizer and waters the same tile on its next turn, while a watering-first
# visit rarely carries fertilizer. Urgent waters (1000) still come first.
PRIO_APPLY_FERT_BERRY = 920.0
PRIO_WEED = 200.0
PRIO_PRODUCTION = 150.0
# A purchased animal waiting in the shed is idle capital.
PRIO_PLACE_ANIMAL = 960.0
PRIO_ENDGAME_HARVEST = 980.0

# Market order priorities. Queue position is settlement order, so SELLs run
# first and free the cash and shed space the buys behind them depend on.
PRIO_MKT_ENDGAME_SELL = 950.0
PRIO_MKT_SELL = 900.0
PRIO_MKT_FEED_TOPUP = 800.0
PRIO_MKT_ANIMAL = 700.0
PRIO_MKT_HIRE = 650.0
PRIO_MKT_LAND = 600.0
PRIO_MKT_SEED = 500.0
