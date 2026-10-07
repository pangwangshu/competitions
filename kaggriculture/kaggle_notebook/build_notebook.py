"""Build the Kaggle notebook from the repository source.

    python kaggle_notebook/build_notebook.py

Writes kaggle_notebook/kaggriculture-planning-agent.ipynb. The notebook is
self-contained: it recreates the agent package with %%writefile cells taken
verbatim from farm_agent/, so it stays in sync with the code it documents.
Upload it in the Kaggle UI (File > Import Notebook), or push it with
`kaggle kernels push -p kaggle_notebook`.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "kaggriculture-planning-agent.ipynb")
REPO = "https://github.com/pangwangshu/competitions/tree/main/kaggriculture"
PAPER = REPO.replace("/tree/", "/blob/") + "/report/paper.md"
FIG = "https://raw.githubusercontent.com/pangwangshu/competitions/main/kaggriculture/report/figures/"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


def writefile(path):
    with open(os.path.join(ROOT, path)) as f:
        code(f"%%writefile {path}\n" + f.read().rstrip("\n"))


# ----------------------------------------------------------------- intro

md(f"""
# Kaggriculture: a planning agent, and how I measured it

A rule-based planning agent for Kaggriculture, with the full agent code in this notebook. It finished mid-pack, not near the top: my final pair rated about 756 (rank ~5,200 of ~10,200, leaderboard median 767). I'm sharing it for the **system design** and the **evaluation lessons**. Most of my six weeks went into one question: *did that change actually help?*

**What's here**
1. The whole agent (about 3,200 lines, standard library only), written to disk cell by cell with a short explanation of each layer.
2. A game against the built-in `starter` bot, with the replay.
3. A paired experiment that reproduces the project's biggest single gain, and shows why sample size matters.
4. A `submission.tar.gz` in the notebook output.

**Full write-up:** [paper]({PAPER}): design, a hierarchical-RL detour, nine evaluation traps, every adopted and rejected change with confidence intervals, and an honest gap analysis. **Code and tests:** [GitHub]({REPO}).

### Results in brief
- **Execution beat learning.** A min-cost-matching dispatcher was worth **+$9,968/game** on held-out seeds (t = 15.1). Six PPO submissions built on the same pipeline never beat it.
- **The interface sets the ceiling.** A *perfect* clone of the strongest public bot, replayed through my strategy interface, made only 52% of that bot's money, below my plain rule agent.
- **Same seed ≠ same game.** The engine draws a weed check for every empty tile before each shop draw. Your land use therefore changes the shops *both* players get, making paired variance about 13× larger for early-game changes.
- **Real effects, flat rating.** Eleven later changes each added $0.8k–7.2k per game with clean CIs, but my median online loss was −$17.6k, so a $3k gain overturns only about 6% of losses.
""")

md("""
## Setup

Needs **Internet on** (Settings → Internet) to install the engine version the agent was verified against. The whole notebook runs in about five minutes on CPU.
""")
code("""
%pip install -q kaggle-environments==1.32.7
import kaggle_environments
print("kaggle-environments", kaggle_environments.__version__)
""")
code("""
import os
for d in ["farm_agent/world", "farm_agent/advisors", "farm_agent/dispatcher", "farm_agent/planners"]:
    os.makedirs(d, exist_ok=True)
""")

# -------------------------------------------------------------- the agent

md("""
## The agent

```
observation ─► world model ─► advisors ─► allocator ─► dispatcher ─► post-passes ─► action
               prices,          propose      the only      min-cost         shed room,
               forecasts,       tasks and    stage that    assignment of    sell order,
               trade flows      orders       says no       units to tasks   last-day sales
```

**Advisors propose, the allocator decides, the dispatcher moves.** Every silent-failure rule of the engine is enforced in exactly one place: the 10-order cap, the 100-item shed, and all-or-nothing `PLANT` when you oversubscribe seeds.

### Entry point and configuration
`main.py` keeps one `FarmAgent` per player index, so self-play in one process works. `config.py` holds every number, with the reasoning next to it.
""")
for p in ["main.py", "farm_agent/__init__.py", "farm_agent/config.py", "farm_agent/messages.py", "farm_agent/agent.py"]:
    writefile(p)

md("""
### World model
- **Price model:** an exact port of the engine's price curves; a test later in this notebook checks it against the engine.
- **Dated forecasts:** crop output laid on the calendar and priced on its sale day.
- **Livestock valuation:** includes the price drop a new animal inflicts on the *opponent*, because the game is decided by the margin, not by own cash.
- **MarketBelief:** recovers the opponent's exact net trades from public inventory changes.
""")
for p in ["farm_agent/world/__init__.py", "farm_agent/world/pricing.py", "farm_agent/world/forecast.py",
          "farm_agent/world/livestock.py", "farm_agent/world/market_belief.py", "farm_agent/world/state.py"]:
    writefile(p)

md("""
### Advisors
Small modules that each look after one concern and emit prioritised `TaskRequest`s and `MarketRequest`s. Priorities share one scale: survival 900–1000, maintenance 200–920, planting 150.
""")
for p in ["farm_agent/advisors/__init__.py", "farm_agent/advisors/base.py", "farm_agent/advisors/survival.py",
          "farm_agent/advisors/maintenance.py", "farm_agent/advisors/crop.py", "farm_agent/advisors/animal.py",
          "farm_agent/advisors/expansion.py", "farm_agent/advisors/market.py", "farm_agent/advisors/wheat.py",
          "farm_agent/advisors/endgame.py"]:
    writefile(p)

md("""
### Allocator
Keeps one task per tile and orders the market queue by priority, because queue position *is* settlement order: sells come first and free cash and shed space for the buys behind them. Running ledgers check cash, shed space, labour and tomorrow's payroll before any purchase.
""")
writefile("farm_agent/allocator.py")

md("""
### Dispatcher
Each turn solves one assignment problem between free units and open tasks, with cost `16 × distance − priority`, using a pure-Python Jonker–Volgenant solver. The weight was the important part: at 1, distance can never override a priority tier, and the matching earned nothing. The experiment below shows the difference.
""")
for p in ["farm_agent/dispatcher/__init__.py", "farm_agent/dispatcher/pathfinding.py",
          "farm_agent/dispatcher/assignment.py", "farm_agent/dispatcher/dispatcher.py"]:
    writefile(p)

md("""
### Planners
Behaviours that act around the main pipeline:
- **Early cash, days 7–11:** bank milk and melons the same day when doing so funds purchases.
- **Late-season wheat:** plant wheat on idle land and labour at the end of the season.
- **Shed room:** stop the overnight shed overflow, which had been destroying about $5k of goods a game, at zero labour cost.
- **Sell order:** sell first where rival supply threatens the price most.
""")
for p in ["farm_agent/planners/__init__.py", "farm_agent/planners/shed_room.py", "farm_agent/planners/sell_order.py",
          "farm_agent/planners/liquidity.py", "farm_agent/planners/late_wheat.py"]:
    writefile(p)

# -------------------------------------------------------------- checks

md("""
## Sanity checks
The agent's model of the market must match the engine it plays against, and the assignment solver must be optimal.
""")
code("""
import itertools, random
from kaggle_environments.envs.kaggriculture import kaggriculture as engine
from farm_agent import config
from farm_agent.world.pricing import market_price
from farm_agent.dispatcher.assignment import min_cost_assignment

assert config.CROPS == engine.CROPS and config.ANIMALS == engine.ANIMALS and config.SHOPS == engine.SHOPS
mismatches = sum(market_price(i, inv) != engine.market_price(i, inv)
                 for i in config.PRODUCTS for inv in range(0, 30001, 13))
print("price mismatches vs engine:", mismatches)

rng = random.Random(0)
for _ in range(200):
    n, m = rng.randint(1, 5), rng.randint(1, 5)
    cost = [[rng.randint(-99, 99) for _ in range(m)] for _ in range(n)]
    got = sum(cost[i][j] for i, j in enumerate(min_cost_assignment(cost)) if j >= 0)
    best = min(sum(cost[i][j] for i, j in (enumerate(p) if n <= m else ((r, c) for c, r in enumerate(p))))
               for p in itertools.permutations(range(max(n, m)), min(n, m)))
    assert got == best
print("assignment solver optimal on 200 random matrices")
""")

# -------------------------------------------------------------- play

md("""
## Play a game
`starter` is the engine's built-in baseline. Try `"random"`, or `"main.py"` for self-play.
""")
code("""
from kaggle_environments import make
import main

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1}, debug=True)
env.run([main.agent, "starter"])
for i, s in enumerate(env.steps[-1]):
    print(f"player {i}: ${s.reward:,.0f}  {s.status}")
""")
code("""
env.render(mode="ipython", width=1000, height=700)
""")

# -------------------------------------------------------------- experiment

md("""
## A paired experiment
The same agent plays the same 30 seeds from both seats twice: once with the dispatcher's distance weight at 1, where priorities always win, and once at 16, as shipped. Each game runs in its own process. Takes a few minutes.

On my full held-out panel (60 seeds × 4 opponents × 2 seats) the gain was **+$9,968/game**. Note the noise: one game's margin difference has a standard deviation near $27k, so with a handful of seeds this test is inconclusive either way. Change `range(30)` to `range(6)` and you will see that. Size the sample before you read the result.
""")
code("""
import json, statistics, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

GAME = '''
import json, sys
sys.path.insert(0, ".")
import main
import farm_agent.dispatcher.dispatcher as dispatcher
dispatcher.DISPATCH_DISTANCE_WEIGHT = {weight}
from kaggle_environments import make
env = make("kaggriculture", configuration={{"episodeSteps": 720, "seed": {seed}}})
env.run([main.agent, "starter"] if {seat} == 0 else ["starter", main.agent])
money = [s.reward for s in env.steps[-1]]
print(json.dumps(money[{seat}] - money[1 - {seat}]))
'''

def margin(weight, seed, seat):
    out = subprocess.run([sys.executable, "-c", GAME.format(weight=weight, seed=seed, seat=seat)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.split()[-1])

pairs = [(seed, seat) for seed in range(30) for seat in (0, 1)]
with ThreadPoolExecutor(4) as pool:
    w1 = list(pool.map(lambda p: margin(1.0, *p), pairs))
    w16 = list(pool.map(lambda p: margin(16.0, *p), pairs))
delta = [b - a for a, b in zip(w1, w16)]
print(f"paired games: {len(delta)}")
print(f"mean margin  w=1: ${statistics.mean(w1):,.0f}   w=16: ${statistics.mean(w16):,.0f}")
se = statistics.stdev(delta) / len(delta) ** 0.5
print(f"paired delta: ${statistics.mean(delta):+,.0f}  95% CI [${statistics.mean(delta) - 1.96 * se:+,.0f}, "
      f"${statistics.mean(delta) + 1.96 * se:+,.0f}], w=16 better in {sum(d > 0 for d in delta)}/{len(delta)}")
""")

# -------------------------------------------------------------- submission

md("""
## Build a submission
Kaggle expects `main.py` at the root of the archive. The file appears in this notebook's Output.
""")
code("""
!tar --exclude=__pycache__ -czf submission.tar.gz main.py farm_agent && tar -tzf submission.tar.gz | wc -l && ls -la submission.tar.gz
""")

# -------------------------------------------------------------- lessons

md(f"""
## What I learned

![Why early changes were unmeasurable]({FIG}fig4_measurement_resolution.png)

1. **Measure the ceiling of a search space before searching it.** Replaying a teacher's own labels through my interface showed a perfect clone would lose, which no amount of training could fix. Instrumenting the controller showed that a 16-million-point parameter search touched none of the 39 constants that actually gated decisions.
2. **Same seed ≠ same game.** Report how often paired games kept the same shop sequence, and pin the weed and shop randomness in your local harness if you can.
3. **Replayed opponents (tapes) flatter big changes**, because they never react. **Tie-break perturbations are not noise either**: they acted like fixed openings worth about ±$2k. Average candidate-minus-control over several tie-break salts.
4. **Compare net cash flow, not gross sales.** "They sell 3× our wheat" was mostly them reselling wheat they had bought.
5. **Size improvements against the loss distribution.** With losses this large, polishing $1–3k mechanisms could not move the rating.

![Online losses]({FIG}fig6_flippable_losses.png)

The strongest agents I could identify by replaying ladder seeds were whole-season plans mined from strong games: plant everything on day 1, run few hands early, get cows by day 4, build a wheat engine from day 12. My turn-by-turn allocator could not express that. If I did it again, I would add a season-plan layer on top of it.

Thanks to Kaggle and the hosts for a genuinely interesting environment, and to everyone who published notebooks: this ladder was unusually rich to learn from. The hire schedule in `config.py` borrows its shape from [Shape the Shop, Work the Pasture](https://www.kaggle.com/code/indarkarhana/shape-the-shop-work-the-pasture-top-10). Questions welcome.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
    c["source"] = c["source"].splitlines(keepends=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(cells)} cells")
