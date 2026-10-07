"""Rectangular min-cost assignment (Hungarian / Jonker-Volgenant).

Pure Python on purpose: the agent runs in the Kaggle sandbox, where a missing
dependency degrades a submission silently. Matrices are small (about 16 units
by 40 tasks), so the O(n^2 m) algorithm is far below the per-turn budget.
tests/test_assignment.py checks optimality against brute force.
"""

from typing import List, Sequence

# Sentinel cost for a (unit, task) pair the unit cannot serve. Finite so the
# algorithm stays well-defined; pairs at or above it are discarded by the
# caller instead of being treated as real assignments.
UNREACHABLE = 1e9


def min_cost_assignment(cost: Sequence[Sequence[float]]) -> List[int]:
    """Returns, per row, the column it is assigned to (-1 if unassigned).

    Handles rectangular input in either orientation. Total cost of the
    returned assignment is minimal.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if m == 0:
        return [-1] * n
    if n > m:  # algorithm below needs rows <= cols; solve the transpose
        transposed = [[cost[i][j] for i in range(n)] for j in range(m)]
        col_of_row = [-1] * n
        for j, i in enumerate(_solve(transposed, m, n)):
            if i >= 0:
                col_of_row[i] = j
        return col_of_row
    return _solve([list(row) for row in cost], n, m)


def _solve(cost: List[List[float]], n: int, m: int) -> List[int]:
    """e-maxx JV shortest-augmenting-path Hungarian; requires n <= m."""
    inf = float("inf")
    u = [0.0] * (n + 1)      # row potentials
    v = [0.0] * (m + 1)      # column potentials
    p = [0] * (m + 1)        # p[j] = 1-indexed row matched to column j
    way = [0] * (m + 1)      # augmenting-path predecessor column

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            row = cost[i0 - 1]
            ui = u[i0]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - ui - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    result = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            result[p[j] - 1] = j - 1
    return result
