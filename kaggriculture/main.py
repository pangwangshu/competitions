"""Kaggle entry point: `agent(obs) -> action`.

The Kaggle runner exec()s this file without defining __file__ and from a
working directory that is not the agent's, so the bundled package is located
on sys.path explicitly.
"""

import os
import sys

try:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE_DIR = next(
        (d for d in [os.getcwd()] + list(sys.path)
         if d and os.path.isdir(os.path.join(d, "farm_agent"))),
        os.getcwd(),
    )
if not sys.path or sys.path[0] != _BASE_DIR:
    sys.path.insert(0, _BASE_DIR)

from farm_agent import FarmAgent  # noqa: E402

_AGENTS = {}


def agent(obs):
    """One FarmAgent per player index, recreated at the start of every episode."""
    player = obs.get("player", 0)
    if player not in _AGENTS or (obs.get("day", 0) == 0 and obs.get("hour", 0) == 0):
        _AGENTS[player] = FarmAgent()
    return _AGENTS[player].act(obs)
