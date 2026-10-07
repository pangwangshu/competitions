# Mid-pack rule-based agent: what 33 submissions taught me about measuring changes in Kaggriculture

This is not a top solution: my final pair rated about 756, close to the leaderboard median. I'm sharing it because most of my six weeks went into one question: *did that change actually help?* The answers include several traps that I think anyone building agents for this environment will run into. Runnable notebook (full agent, a game replay, and a paired experiment): **[wangshu/kaggriculture-planning-agent-and-evaluation](https://www.kaggle.com/code/wangshu/kaggriculture-planning-agent-and-evaluation)**. Paper, code and tests: **[GitHub](https://github.com/pangwangshu/competitions/tree/main/kaggriculture)**.

## TL;DR

- A hand-written **advisor → allocator → dispatcher** pipeline beat every version of my PPO and behavioural-cloning agents.
- The largest single gain was **boring execution work**: replacing greedy task assignment with Hungarian matching (+$10k per game) and fixing a hiring rule that never bootstrapped.
- **Same seed ≠ same game.** Your land use changes which shops *both* players get, so offline A/B tests of early-game changes are about 13× noisier than they look.
- I adopted 11 changes worth +$0.8k–7k per game each, all with clean confidence intervals. None visibly moved my rating, because my median online loss was −$17.6k.
- The agents at the top replay whole-season plans. My reactive architecture could not express that.

## The agent

Each turn runs four stages:

1. **World model.** An exact port of the price curves, plus town-demand and berry-production forecasts. It also does exact trade-flow accounting: because you know your own orders, `Δmarket_inventory = Σ(sales above $1 − buys) − town consumption` holds exactly, so the opponent's net sales can be recovered from public data alone.
2. **Advisors.** Survival, maintenance, crop, animal, expansion, market and endgame modules. They only *propose* prioritised task and market requests; they never move units.
3. **Allocator.** The only place allowed to say no. It keeps ledgers for labour, seeds, shed space, cash and next-day payroll. Market orders are sorted so that queue position equals settlement order (sells first). The queue is cut at 10 orders.
4. **Dispatcher.** Min-cost matching of units to tasks, with cost = `16 × travel − priority`.

Every candidate change shipped behind a flag whose "off" setting reproduced the previous version byte-for-byte, and that was verified on complete games.

## Engine facts worth knowing

None of these are secret, but each one cost me something before I noticed it:

- **Turn order is units → market → town → end-of-day sweep.** A `SELL` on hour 23 settles *before* the overnight shed pour. Selling buyable stock (wheat or fertilizer) on hour 23 frees shed space at zero labour cost. Together with skipping animal harvests on nights the shed would overflow, this cut my overflow losses from about $5.1k to $0.7k per game.
- **The last end-of-day sweep never runs.** Anything a unit is carrying on day 29 is worth $0. Walking carriers home and selling was worth +$3.7k per game.
- **One bad `PLANT` voids the rest.** If a turn's `PLANT` actions for a crop exceed your seed stock, *all* of them are voided.
- **Seeds lose the order cap.** With 10 market orders per turn and seeds as my lowest priority, I was silently dropping about 150 seed orders per game.
- **Own wheat is worth only the spread.** Only wheat and fertilizer can be bought back, and a buy/sell round trip nets zero. So the marginal value of your own wheat is roughly $5/unit, not its sale price. Never spend contested labour on it; idle late-season labour is fine (+$3k per game for me).
- **Replays are reproducible.** Replays store `info.seed`, so any ladder game can be replayed exactly offline. Check with a self-play control first (mine reproduced 719/719 steps in both seats).

## What worked (paired Δ final margin vs. the previous version, 95% CI)

| Change | Δ $/game |
|---|---|
| Hungarian dispatch + hire schedule | +19,926 (t = 27) |
| Dated strawberry/melon forecast + fertilizer service plan | +7,157 [5,076, 9,138] |
| Walk carried goods home on the last day | +3,705 [3,425, 3,985] |
| Hour-23 shed-room sells + defer animal harvest on overflow nights | +3,366 [3,109, 3,627] |
| Late-season wheat on idle land and labour | +2,903 [2,654, 3,158] |
| Value animals by *margin*: count the price drop you cause the rival (×0.5) | +2,913 [1,839, 4,008] |
| Never buy the $4,000 fourth quadrant | +2,824 [2,384, 3,210] |
| Two-day fertilizer look-ahead + skip no-value waterings | +2,628 [1,962, 3,292] |

Notes on three of these:

- **Dispatcher.** The distance/priority exchange rate mattered more than the algorithm. Matching that never let distance override a priority tier earned nothing (+$758, t = 0.9). At weight 16, roughly three tiles of detour outweigh one priority tier. "Pure distance" looked best on tuning seeds and lost on held-out seeds.
- **Hiring.** My reactive hire rule never bootstrapped: on day 0 an empty farm has no backlog, so I opened with one unit and plateaued around seven. Hiring cost resets daily, so hiring early in the day is free relative to hiring late. A per-day hand-count floor fixed it. Its shape came from the hire curve in [Shape the Shop, Work the Pasture](https://www.kaggle.com/code/indarkarhana/shape-the-shop-work-the-pasture-top-10); thanks to its author.
- **The fourth quadrant.** I tested whether it simply came too late. Buying it *earlier* was significantly worse. My farm occupancy plateaued around 73 tiles, so the extra 25 tiles brought in about $354.

## What didn't work

- **PPO over a strategy interface** (rotation weights, sell tiers, hire target, land on/off, with the rule pipeline executing). Six submissions rated 668–714. Changing only the executor underneath a fixed checkpoint helped more than any retraining did.
- **Behavioural cloning** of strong public bots reached 0.99 held-out agreement, yet the clones lost every game to their teachers. The diagnostic that explained it: replay the teacher's *own* labels through your interface, i.e. a perfect clone. That scored **52% of the teacher, below my plain rule agent.** A rotation *ratio* cannot express "12 melons and 4 pastures on day 0, then run on a cash knife-edge". Measure the ceiling of your interface before training anything.
- **Parameter search** over a 16.2M-point schedule space found nothing. Instrumenting the controller showed why: of the 39 constants that actually gated decisions, the search space touched 0.
- **Copying the opponent's herd** (more cows to close a milk gap). Price elasticity was about −1, so I sold more milk for the same money.
- **Better price forecasts.** My forecasting model changed 0 of 1,728 planting decisions. I was volume-limited, and my unit prices were already 5–8% above my rivals'.

## Evaluation traps (the part I'd most like to pass on)

1. **Shared end-of-day RNG.** The engine checks every *empty tile* for a weed before drawing the next shop. If your change alters land use before about day 21, both players get a different shop sequence. On my 151-game panel, a late-season change kept the same shops in 151/151 pairs (paired sd $1.7k). An opening change kept them in 0/151 (sd $23.3k). Resolving a $3k opening effect would need about 232 games. **Fix:** hash weeds from (seed, day, tile) and pin the shop sequence in your local harness. That cut sd by 35–45% for me.
2. **Replayed opponents (tapes) flatter big deviations.** A recorded opponent never reacts. Whichever variant deviates more from the game the tape came from gets an artificial boost. One comparison flipped sign depending on which arm had produced the tapes.
3. **Tie-break "placebos" are not noise.** Perturbing only tie-breaks (ε < 1e-3) shifted results by a *consistent* ±$2k across 72 seeds, because the opening is nearly identical across seeds. **Fix:** run candidate and control under several tie-break salts and average the per-salt differences.
4. **Your harness may be stricter than the engine.** The engine casts `"1"` to `1`; my validator rejected it, and a perfectly good opponent looked broken.
5. **Two agents, one process.** kaggle-environments runs both agents in one process. Two bundles with a package of the *same name* share whichever was imported first. My "frozen baseline" was silently running my current code for days.
6. **Compare net, not gross.** "Their wheat sales are 3× ours" was mostly them reselling bought wheat. Decompose `sales − buys` and check that the parts add up to the cash difference.

## Why it didn't move the rating, and what I'd do differently

My production was nearly identical in wins and losses (about 165 plantings per game). The result was decided by the opponent and the shop draw. A +$3k improvement overturns about 6% of my losses; overturning half needs about +$17k. The strongest agents I identified by seed-replay were tape replayers: whole-season action sequences mined from strong games, behind a thin router. Their edge is structural: plant everything on day 1, run few hands early, get cows by day 4, build a wheat engine from day 12. That puts them $13k ahead in cash by day 12, and the lead compounds.

If I did it again, I would:

- build the seed-replay evaluation set in week one, not week three;
- size each idea against the actual loss distribution before building it;
- add a season-plan layer (an opening-schedule search over a cash-flow simulator) on top of the reactive allocator, instead of polishing $1–3k mechanisms.

Thanks to Kaggle and the hosts for a genuinely interesting environment, and to everyone who published notebooks; this ladder was unusually rich to learn from. Happy to answer questions.
