"""Play one full game locally.

    python scripts/play.py                         # vs the built-in "starter" bot
    python scripts/play.py --opponent random --seed 3 --seat 1
    python scripts/play.py --opponent path/to/other_bot/main.py --html replay.html
    python scripts/play.py --opponent self         # self-play

Built-in opponents: pass, random, starter. "random" is unseeded, so its games
are not reproducible even with --seed.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.join(os.path.dirname(HERE), "main.py")
BUILTIN = ("pass", "random", "starter")


def run_game(me, opponent, seed, seat=0, html=None):
    """Play one 720-turn episode; returns a dict with both players' final money."""
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    if opponent == "self":
        opponent = me
    elif opponent not in BUILTIN:
        opponent = os.path.abspath(opponent)
    agents = [me, opponent] if seat == 0 else [opponent, me]
    env.run(agents)
    final = env.steps[-1]
    money = [s.reward for s in final]
    if html:
        with open(html, "w") as f:
            f.write(env.render(mode="html", width=1200, height=800))
    town = final[0].observation.get("town") or {}
    return {
        "seed": seed,
        "seat": seat,
        "money": money,
        "margin": money[seat] - money[1 - seat],
        "status": [s.status for s in final],
        "shops": list(town.get("unlocked_shops", [])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default=AGENT, help="path to the agent's main.py")
    ap.add_argument("--opponent", default="starter", help="pass | random | starter | self | path/to/main.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seat", type=int, choices=(0, 1), default=0)
    ap.add_argument("--html", help="write a replay viewer to this file")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args()

    result = run_game(os.path.abspath(args.agent), args.opponent, args.seed, args.seat, args.html)
    if args.json:
        print(json.dumps(result))
        return
    me, opp = result["money"][args.seat], result["money"][1 - args.seat]
    verdict = "won" if me > opp else "lost" if me < opp else "tied"
    print(f"seed {args.seed}, seat {args.seat}: agent ${me:,.0f} vs {args.opponent} ${opp:,.0f} -> {verdict} "
          f"by ${abs(me - opp):,.0f}  (status {result['status']})")


if __name__ == "__main__":
    sys.exit(main())
