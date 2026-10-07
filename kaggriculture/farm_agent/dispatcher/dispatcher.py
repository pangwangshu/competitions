"""Dispatcher: turns the approved task list into one action per unit.

Reads no strategy state and makes no economic decisions. Each turn:

  Phase 0  pinned units: the last-day walk home, units sent home by the
           liquidity planner, and deadline feeding runs
  Phase 1  a unit standing on its task tile with the required items acts now
  Phase 2  units mid-trip keep their target, so new tasks nearby cause no thrashing
  Phase 3  global min-cost assignment of free units to unclaimed tasks
  Phase 4  greedy sweep for anything the assignment could not place
"""

from typing import Any, Dict, List, Optional, Tuple

from farm_agent.config import (
    PRODUCTS,
    TERMINAL_RETURN_MIN_VALUE,
    TERMINAL_RETURN_SAFETY_TURNS,
    TERMINAL_SELL_STEP,
    TURNS_PER_DAY,
)
from farm_agent.dispatcher.assignment import UNREACHABLE, min_cost_assignment
from farm_agent.dispatcher.pathfinding import manhattan_dist, step_toward
from farm_agent.messages import TaskRequest
from farm_agent.world.state import WorldModel

PICKUP_BATCH_MAX = 4

# Exchange rate between one tile of walking and the priority scale in the
# assignment cost `DISPATCH_DISTANCE_WEIGHT * travel - priority`. Adjacent
# priority tiers are 50 apart, so at 16 about three tiles of detour outweigh
# one tier: a nearby WATER beats a distant FEED, an adjacent one never does.
#
# Tuned, not assumed. A weight that never lets distance invert a priority tier
# (1.0) earned nothing (+$758/game, t = 0.9). A sweep over 1-64 plateaued at
# 16: +$9,968/game on held-out seeds (t = 15.1, n = 480), 93% head-to-head.
# Pure distance looked best on the tuning seeds and lost on held-out ones.
# Feeding, the irreversible failure mode, became more reliable, not less.
DISPATCH_DISTANCE_WEIGHT = 16.0

# Deadline feeding: an animal unfed since yesterday escapes tonight. When the
# nearest wheat carrier has at most this many turns to spare, it is pinned to
# the feed run rather than left to the assignment.
FEED_DEADLINE_SLACK = 2
FEED_DEADLINE_LAST_DAY = 27
NO_COMMITMENTS_FROM_DAY = 24


class Dispatcher:

    def __init__(self):
        self.day: Optional[int] = None
        # unit idx -> committed task tile; hands are re-hired daily, so reset each day
        self.commitments: Dict[int, Tuple[int, int]] = {}

    def sync_memory(self, day: int, step: int):
        if step == 0 or self.day != day:
            self.commitments.clear()
            self.day = day

    def dispatch(
        self, world: WorldModel, tasks: List[TaskRequest],
        pinned: Optional[Dict[int, List[Any]]] = None,
    ) -> Tuple[List[Any], List[List[Any]]]:
        """Returns (farmer action, hand actions)."""
        self.sync_memory(world.day, world.step)

        # Phase 0 -- pinned units leave the task pool entirely.
        returning = self._terminal_returns(world)
        for idx, act in (pinned or {}).items():
            returning.setdefault(idx, act)

        rescue_tiles = set()
        if world.day <= FEED_DEADLINE_LAST_DAY:
            remaining = TURNS_PER_DAY - world.hour
            endangered = []
            for task in tasks:
                if task.action != ["FEED"]:
                    continue
                tile = world.tiles[task.pos[1]][task.pos[0]]
                if tile.get("consecutive_unfed", 0) >= 1 and not tile.get("fed_today"):
                    endangered.append(task)
            for task in endangered:
                eligible = [(manhattan_dist(pos, task.pos), idx, pos)
                            for idx, pos in world.units if idx not in returning
                            and self._inv(world, idx).get("WHEAT", 0) > 0]
                if not eligible:
                    continue
                dist, idx, pos = min(eligible)
                if not (0 <= remaining - dist - 1 <= FEED_DEADLINE_SLACK):
                    continue
                returning[idx] = (["FEED"] if dist == 0 else
                                  [step_toward(pos[0], pos[1], task.pos[0], task.pos[1])])
                rescue_tiles.add(task.pos)
        tasks = [t for t in tasks if t.action and t.pos not in rescue_tiles]
        # A plant starts one dry day behind, so one planted on the last turn of
        # the day dies at tonight's refresh. Keep the seed for tomorrow.
        if world.hour >= TURNS_PER_DAY - 1:
            tasks = [t for t in tasks if t.action[0] != "PLANT"]
        task_by_pos: Dict[Tuple[int, int], TaskRequest] = {t.pos: t for t in tasks}
        # The engine voids ALL of a crop's PLANTs in a turn that exceeds its
        # seed stock, so never issue more than the stock.
        seed_budget: Dict[str, int] = dict(world.seeds)
        # Pending demand for each carried item, for batched PICKUPs.
        item_demand: Dict[str, int] = {}
        for t in tasks:
            for item, amt in t.requires_inv.items():
                item_demand[item] = item_demand.get(item, 0) + amt

        actions: Dict[int, List[Any]] = dict(returning)
        claimed: set = set()
        free_units: Dict[int, Tuple[int, int]] = {}

        # Phase 1 -- act on the tile you are standing on.
        for idx, pos in world.units:
            if idx in returning:
                self.commitments.pop(idx, None)
                continue
            task = task_by_pos.get(pos)
            if (
                task is not None
                and pos not in claimed
                and self._missing_item(self._inv(world, idx), task) is None
                and self._consume_seed(task.action, seed_budget)
            ):
                actions[idx] = task.action
                claimed.add(pos)
                self.commitments.pop(idx, None)
            else:
                free_units[idx] = pos

        if world.day >= NO_COMMITMENTS_FROM_DAY:
            self.commitments.clear()

        # Phase 2 -- keep cross-turn commitments.
        for idx in list(free_units):
            cpos = self.commitments.get(idx)
            if cpos is None:
                continue
            task = task_by_pos.get(cpos)
            if task is None or cpos in claimed:
                self.commitments.pop(idx, None)
                continue
            act = self._move_or_fetch(world, free_units[idx], self._inv(world, idx), task,
                                      item_demand, seed_budget)
            if act is None:  # the required item is no longer obtainable
                self.commitments.pop(idx, None)
                continue
            actions[idx] = act
            claimed.add(cpos)
            free_units.pop(idx)

        # Phase 3 -- one assignment over every free unit and unclaimed task.
        # A task-by-task greedy let an early task take the one unit standing
        # next to a later task of equal priority (~1,900 such crossings a game).
        pending = [t for t in tasks if t.pos not in claimed]
        if free_units and pending:
            unit_idxs = list(free_units)
            matrix = []
            for idx in unit_idxs:
                pos = free_units[idx]
                inv = self._inv(world, idx)
                row = []
                for task in pending:
                    dist = self._effective_dist(world, pos, inv, task)
                    row.append(UNREACHABLE if dist is None
                               else DISPATCH_DISTANCE_WEIGHT * dist - task.priority)
                matrix.append(row)
            for row_i, col in enumerate(min_cost_assignment(matrix)):
                if col < 0 or matrix[row_i][col] >= UNREACHABLE:
                    continue
                idx = unit_idxs[row_i]
                task = pending[col]
                act = self._move_or_fetch(world, free_units[idx], self._inv(world, idx), task,
                                          item_demand, seed_budget)
                if act is None:
                    continue
                actions[idx] = act
                claimed.add(task.pos)
                self.commitments[idx] = task.pos
                free_units.pop(idx)

        # Phase 4 -- greedy sweep, so coverage is never lower than plain greedy.
        for task in tasks:
            if not free_units:
                break
            if task.pos in claimed:
                continue
            best: Optional[Tuple[int, int]] = None  # (distance, unit idx)
            for idx, pos in free_units.items():
                cost = self._effective_dist(world, pos, self._inv(world, idx), task)
                if cost is None:
                    continue
                if best is None or cost < best[0]:
                    best = (cost, idx)
            if best is None:
                continue
            idx = best[1]
            act = self._move_or_fetch(world, free_units[idx], self._inv(world, idx), task,
                                      item_demand, seed_budget)
            if act is None:
                continue
            actions[idx] = act
            claimed.add(task.pos)
            self.commitments[idx] = task.pos
            free_units.pop(idx)

        unit_actions = [actions.get(idx, ["PASS"]) for idx, _ in world.units]
        return unit_actions[0], unit_actions[1:]

    # ------------------------------------------------------- terminal return

    def _terminal_returns(self, world: WorldModel) -> Dict[int, List[Any]]:
        """Units walking home on the last day so their load can still be sold.

        The season stops at hour 22 of the last day and the end-of-day sweep
        that empties backpacks into the shed never runs, so goods carried at
        the end are worth nothing (+$3.7k/game once fixed). A carrier keeps
        working until the walk home only just fits before the final market
        phase, and is left alone once it can no longer arrive in time. Distance
        and slack both fall by one per turn, so a diverted unit stays diverted
        without any state of its own.
        """
        # Step from day and hour: saved replays omit `step` for the second seat.
        step = world.day * TURNS_PER_DAY + world.hour
        slack = TERMINAL_SELL_STEP - step
        if slack < 0:
            return {}

        returning: Dict[int, List[Any]] = {}
        for idx, pos in world.units:
            inv = self._inv(world, idx)
            if not inv:
                continue
            value = sum(n * world.prices.get(item, 0) for item, n in inv.items()
                        if item in PRODUCTS and n > 0)
            if value <= TERMINAL_RETURN_MIN_VALUE:
                continue
            shed_tile = world.nearest_shed_tile(pos[0], pos[1])
            dist = manhattan_dist(pos, shed_tile)
            if dist > slack or dist < slack - TERMINAL_RETURN_SAFETY_TURNS:
                continue
            returning[idx] = (["DROP"] if dist == 0
                              else [step_toward(pos[0], pos[1], shed_tile[0], shed_tile[1])])
        return returning

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _inv(world: WorldModel, idx: int) -> Dict[str, int]:
        return world.inventories[idx] if idx < len(world.inventories) else {}

    @staticmethod
    def _missing_item(inv: Dict[str, int], task: TaskRequest) -> Optional[str]:
        for item, amt in task.requires_inv.items():
            if inv.get(item, 0) < amt:
                return item
        return None

    def _effective_dist(self, world: WorldModel, pos: Tuple[int, int],
                        inv: Dict[str, int], task: TaskRequest) -> Optional[int]:
        """Travel for this unit to serve the task, via the shed if it must fetch an item."""
        missing = self._missing_item(inv, task)
        if missing is None:
            return manhattan_dist(pos, task.pos)
        if world.shed.get(missing, 0) <= 0:
            return None
        shed_tile = world.nearest_shed_tile(pos[0], pos[1])
        return manhattan_dist(pos, shed_tile) + manhattan_dist(shed_tile, task.pos)

    def _move_or_fetch(self, world: WorldModel, pos: Tuple[int, int], inv: Dict[str, int],
                       task: TaskRequest, item_demand: Dict[str, int],
                       seed_budget: Dict[str, int]) -> Optional[List[Any]]:
        """Fetch a missing item from the shed, walk to the tile, or act on it."""
        missing = self._missing_item(inv, task)
        if missing is not None:
            if world.shed.get(missing, 0) <= 0:
                return None
            if world.is_shed_adjacent(pos):
                qty = max(1, min(PICKUP_BATCH_MAX, item_demand.get(missing, 1), world.shed[missing]))
                return ["PICKUP", missing, qty]
            tx, ty = world.nearest_shed_tile(pos[0], pos[1])
            return [step_toward(pos[0], pos[1], tx, ty)]
        if pos == task.pos:
            if not self._consume_seed(task.action, seed_budget):
                return None
            return task.action
        return [step_toward(pos[0], pos[1], task.pos[0], task.pos[1])]

    @staticmethod
    def _consume_seed(action: List[Any], seed_budget: Dict[str, int]) -> bool:
        """Reserve a seed for a PLANT; True for every other action."""
        if not action or action[0] != "PLANT":
            return True
        crop = action[1]
        if seed_budget.get(crop, 0) <= 0:
            return False
        seed_budget[crop] -= 1
        return True
