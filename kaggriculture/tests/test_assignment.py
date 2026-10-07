"""The dispatcher's assignment solver returns a minimum-cost matching."""

import itertools
import random

from farm_agent.dispatcher.assignment import UNREACHABLE, min_cost_assignment


def brute_force(cost):
    n, m = len(cost), len(cost[0])
    best = None
    if n <= m:
        for cols in itertools.permutations(range(m), n):
            total = sum(cost[i][j] for i, j in enumerate(cols))
            best = total if best is None else min(best, total)
    else:
        for rows in itertools.permutations(range(n), m):
            total = sum(cost[i][j] for j, i in enumerate(rows))
            best = total if best is None else min(best, total)
    return best


def total(cost, assignment):
    return sum(cost[i][j] for i, j in enumerate(assignment) if j >= 0)


def test_matches_brute_force_on_random_rectangular_matrices():
    rng = random.Random(0)
    for _ in range(300):
        n, m = rng.randint(1, 6), rng.randint(1, 6)
        cost = [[rng.randint(-1000, 1000) for _ in range(m)] for _ in range(n)]
        assignment = min_cost_assignment(cost)
        assert len(assignment) == n
        used = [j for j in assignment if j >= 0]
        assert len(used) == len(set(used)) == min(n, m)
        assert total(cost, assignment) == brute_force(cost)


def test_unreachable_pairs_are_avoided_when_possible():
    cost = [[UNREACHABLE, 5.0], [3.0, UNREACHABLE]]
    assert min_cost_assignment(cost) == [1, 0]


def test_dispatch_cost_prefers_nearby_task_of_slightly_lower_priority():
    # Unit 0 stands next to a WATER (900); unit 1 is next to a FEED (950).
    # With cost = 16 * distance - priority each unit should serve its neighbour.
    w = 16.0
    cost = [[w * 1 - 900, w * 8 - 950],
            [w * 8 - 900, w * 1 - 950]]
    assert min_cost_assignment(cost) == [0, 1]


def test_empty_inputs():
    assert min_cost_assignment([]) == []
    assert min_cost_assignment([[], []]) == [-1, -1]
