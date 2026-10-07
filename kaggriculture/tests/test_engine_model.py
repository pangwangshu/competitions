"""The agent's game model agrees with the engine it plays against."""

import pytest

kag = pytest.importorskip("kaggle_environments.envs.kaggriculture.kaggriculture")

from farm_agent import config  # noqa: E402
from farm_agent.world import market_belief, pricing  # noqa: E402


def test_static_tables_match_engine():
    assert config.CROPS == kag.CROPS
    assert config.ANIMALS == kag.ANIMALS
    assert config.PRODUCTS == kag.PRODUCTS
    assert config.SHOPS == kag.SHOPS
    assert config.MAX_SHOP_INSTANCES == kag.MAX_SHOP_INSTANCES
    assert config.MARKET_I0 == kag.MARKET_I0
    assert config.HINGE_GAIN == kag.HINGE_GAIN
    for item, p in kag.MARKET_PARAMS.items():
        assert {k: v for k, v in p.items() if k != "I0"} == config.MARKET_PARAMS[item]
    # The agent deliberately never buys the last quadrant.
    assert config.LAND_PRICES == kag.LAND_PRICES[:len(config.LAND_PRICES)]


@pytest.mark.parametrize("item", config.PRODUCTS)
def test_price_function_matches_engine(item):
    for inventory in list(range(8000, 12001, 7)) + [0, 5000, 20000, 60000]:
        assert pricing.market_price(item, inventory) == kag.market_price(item, inventory)


def test_town_consumption_rule():
    shops = ["YARN_STORE", "BAKERY"]
    # Every 4th step each shop takes one of each menu item, two for single-item menus.
    assert market_belief.town_consumption(4, shops) == {"WOOL": 2, "EGG": 1, "WHEAT": 1}
    # The town centre takes one of every product except fertilizer once a day.
    at_midnight = market_belief.town_consumption(24, [])
    assert set(at_midnight) == set(config.PRODUCTS) - {"FERTILIZER"}
    assert market_belief.town_consumption(5, shops) == {}


def test_hire_cost_is_fibonacci():
    assert [pricing.fib(n) for n in range(8)] == [1, 1, 2, 3, 5, 8, 13, 21]
