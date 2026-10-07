"""ResourceAllocator: the only stage allowed to say no.

Merges every advisor's proposals into (a) one approved task per tile, sorted
by priority, for the dispatcher, and (b) a market order queue checked against
running cash, shed-space and payroll ledgers and cut to the engine's limit.
"""

from typing import Any, Dict, List, Tuple

from farm_agent.advisors.animal import AnimalAdvisor
from farm_agent.advisors.crop import CropAdvisor
from farm_agent.advisors.endgame import EndgameAdvisor
from farm_agent.advisors.endgame import is_active as endgame_active
from farm_agent.advisors.expansion import ExpansionAdvisor
from farm_agent.advisors.maintenance import MaintenanceAdvisor
from farm_agent.advisors.market import MarketAdvisor
from farm_agent.advisors.survival import SurvivalAdvisor
from farm_agent.advisors.wheat import WheatLiquidityAdvisor
from farm_agent.config import (
    ANIMAL_BUILD_RESERVE,
    ANIMALS,
    CROPS,
    FARM_HAND_COST_MULT,
    HIRE_BACKLOG_PER_UNIT,
    MAX_ANIMAL_STRUCTURES,
    MAX_MARKET_ORDERS,
    MONEY_RESERVE,
    PRIO_PLACE_ANIMAL,
    PRIO_PRODUCTION,
    ROTATION_SIZE,
    SEASON_DAYS,
    SHED_CAPACITY,
    STRUCTURE_ANIMALS,
    UNIT_DAILY_TASK_CAPACITY,
)
from farm_agent.dispatcher.pathfinding import manhattan_dist
from farm_agent.messages import MarketRequest, TaskRequest
from farm_agent.planners.late_wheat import LateWheatPlanner
from farm_agent.world.pricing import fib, wheat_self_sufficiency_tiles
from farm_agent.world.state import WorldModel

# Hires considered when estimating labour capacity and tomorrow's payroll.
MAX_HIRABLE_LOOKAHEAD = 8

Option = Tuple[str, str, float]  # (kind, item, score); kind is CROP, COOP or PASTURE


class ResourceAllocator:

    def __init__(self):
        self.survival_advisor = SurvivalAdvisor()
        self.maintenance_advisor = MaintenanceAdvisor()
        self.crop_advisor = CropAdvisor()
        self.animal_advisor = AnimalAdvisor()
        self.expansion_advisor = ExpansionAdvisor()
        self.market_advisor = MarketAdvisor()
        self.wheat_advisor = WheatLiquidityAdvisor()
        self.endgame_advisor = EndgameAdvisor()
        self.late_wheat = LateWheatPlanner()

    def resolve(self, world: WorldModel) -> Tuple[List[TaskRequest], List[List[Any]]]:
        """(approved tasks, highest priority first; market orders in settlement order)."""
        alloc_options = self._strategy_rotation(world)

        tasks: List[TaskRequest] = []
        market_reqs: List[MarketRequest] = []
        for advisor in (
            self.survival_advisor,
            self.maintenance_advisor,
            self.endgame_advisor,
            self.wheat_advisor,
            self.market_advisor,
            self.animal_advisor,
            self.expansion_advisor,
        ):
            adv_tasks, adv_market = advisor.get_requests(world)
            tasks.extend(adv_tasks)
            market_reqs.extend(adv_market)

        in_endgame = endgame_active(world)
        plant_budget = 0 if in_endgame else self._plant_budget(world)
        tasks.extend(self._production_tasks(world, alloc_options, plant_budget))
        if in_endgame:
            lw_tasks, lw_market = self.late_wheat.requests(world, {t.pos for t in tasks})
            tasks.extend(lw_tasks)
            market_reqs.extend(lw_market)
        else:
            active_crops = [item for kind, item, _ in alloc_options if kind == "CROP"]
            market_reqs.extend(self.crop_advisor.seed_requests(world, active_crops, plant_budget))

        if world.day == SEASON_DAYS - 1:
            tasks = [t for t in tasks if not self._pointless_on_last_day(world, t)]
        return self._arbitrate_tasks(tasks), self._arbitrate_market(world, market_reqs)

    @staticmethod
    def _pointless_on_last_day(world: WorldModel, task: TaskRequest) -> bool:
        """No end-of-day refresh follows the final turn, so upkeep pays nothing."""
        if task.action[0] in ("FEED", "CARE", "DIG"):
            return True
        if task.action[0] not in ("WATER", "FERTILIZE"):
            return False
        tile = world.tiles[task.pos[1]][task.pos[0]]
        if not isinstance(tile, dict):
            return False
        cd = CROPS[tile["crop"]]
        return cd["ongoing"] or world.day - tile["planted_day"] < cd["first_yield_day"]

    # ------------------------------------------------------------ strategy

    def _strategy_rotation(self, world: WorldModel) -> List[Option]:
        """The top-scored production options, served round-robin over empty tiles."""
        tc = world.classified_tiles
        animal_evals = self.animal_advisor.evaluate_animals(world)
        crop_evals = self.crop_advisor.evaluate_crops(world)

        options: List[Option] = [("CROP", crop, score) for crop, score in crop_evals]
        if animal_evals["goose_score"] > float("-inf"):
            # The goose keeps its rank in the rotation, but its slot builds a
            # pasture whenever the pasture species scores higher: otherwise a
            # goose was bought merely because the free structure was a coop.
            if animal_evals["pasture_score"] >= animal_evals["goose_score"]:
                options.append(("PASTURE", animal_evals["pasture_animal"], animal_evals["goose_score"]))
            else:
                options.append(("COOP", "GOOSE", animal_evals["goose_score"]))
        if animal_evals["pasture_score"] > float("-inf"):
            options.append(("PASTURE", animal_evals["pasture_animal"], animal_evals["pasture_score"]))
        options.sort(key=lambda o: -o[2])
        alloc_options = options[:ROTATION_SIZE] if options else [("CROP", "WHEAT", 0.0)]

        # Reserve enough wheat slots to feed the herd from our own harvest.
        # Slots above one are capped by seed stock so PLANT volume never
        # outruns seeds; one slot is always kept, because seeds are only
        # restocked for crops already in the rotation.
        wheat_slots_present = sum(1 for k, i, _ in alloc_options if k == "CROP" and i == "WHEAT")
        wheat_slots_needed = wheat_self_sufficiency_tiles(tc.animal_count, len(tc.unoccupied_structure))
        wheat_target = (
            min(wheat_slots_needed, max(1, world.seeds.get("WHEAT", 0)))
            if wheat_slots_needed > 0
            else 0
        )
        if wheat_target > wheat_slots_present:
            wheat_score = next((s for c, s in crop_evals if c == "WHEAT"), 0.0)
            alloc_options = alloc_options + [
                ("CROP", "WHEAT", wheat_score)
                for _ in range(wheat_target - wheat_slots_present)
            ]
        return alloc_options

    # ---------------------------------------------------- production tasks

    @staticmethod
    def _plant_budget(world: WorldModel) -> int:
        """Labour ledger: new plantings approved while capacity covers the chores.

        Capacity counts current units plus the hands hirable right now, since
        hands vanish every evening and are re-hired on demand.
        """
        tc = world.classified_tiles
        hirable, cash = 0, world.money
        hires_today = world.my_farm.get("hires_today", 0)
        while hirable < MAX_HIRABLE_LOOKAHEAD:
            cost = FARM_HAND_COST_MULT * fib(hires_today + hirable)
            if cash - cost < MONEY_RESERVE:
                break
            cash -= cost
            hirable += 1
        capacity = (len(world.units) + hirable) * UNIT_DAILY_TASK_CAPACITY
        workload = tc.plant_count + tc.animal_count
        return max(0, capacity - workload)

    def _production_tasks(self, world: WorldModel, alloc_options: List[Option],
                          plant_budget: int) -> List[TaskRequest]:
        tasks: List[TaskRequest] = []
        tc = world.classified_tiles

        # 1. Place purchased animals, never sending two structures after one animal.
        available: Dict[str, int] = {}
        for species in ANIMALS:
            available[species] = world.shed.get(species, 0) + sum(
                inv.get(species, 0) for inv in world.inventories
            )
        for x, y, kind in tc.unoccupied_structure:
            for cand in STRUCTURE_ANIMALS[kind]:
                if available.get(cand, 0) > 0:
                    available[cand] -= 1
                    tasks.append(TaskRequest(category="place", pos=(x, y), priority=PRIO_PLACE_ANIMAL,
                                             action=["PLACE", cand], requires_inv={cand: 1}))
                    break

        # 2. Every fundable empty tile gets a PLANT or BUILD task, cycling the
        #    rotation, so units crossing the farm can plant on the way. The
        #    dispatcher enforces per-turn seed stock.
        if not alloc_options:
            return tasks
        structure_count = tc.animal_count + len(tc.unoccupied_structure)
        unit_positions = [pos for _, pos in world.units]
        empty_tiles = sorted(
            tc.empty,
            key=lambda t: min(manhattan_dist(t, p) for p in unit_positions) if unit_positions else 0,
        )
        rotation_idx = 0
        for pos in empty_tiles:
            chosen = None
            for offset in range(len(alloc_options)):
                kind, item, _score = alloc_options[(rotation_idx + offset) % len(alloc_options)]
                if kind == "CROP":
                    if plant_budget <= 0 or world.seeds.get(item, 0) <= 0:
                        continue
                    plant_budget -= 1
                    chosen = TaskRequest(category="empty", pos=pos, priority=PRIO_PRODUCTION,
                                         action=["PLANT", item], payload={"crop": item})
                else:
                    if structure_count >= MAX_ANIMAL_STRUCTURES:
                        continue
                    if world.money - ANIMALS[item]["cost"] < ANIMAL_BUILD_RESERVE:
                        continue
                    structure_count += 1
                    chosen = TaskRequest(category="empty", pos=pos, priority=PRIO_PRODUCTION,
                                         action=["BUILD_COOP"] if kind == "COOP" else ["BUILD_PASTURE"],
                                         payload={"structure": kind, "animal": item})
                rotation_idx += 1
                break
            if chosen is not None:
                tasks.append(chosen)
        return tasks

    # --------------------------------------------------------- arbitration

    @staticmethod
    def _arbitrate_tasks(tasks: List[TaskRequest]) -> List[TaskRequest]:
        """Highest priority first; one task per tile."""
        tasks.sort(key=lambda t: -t.priority)
        approved: List[TaskRequest] = []
        claimed_pos = set()
        for task in tasks:
            if task.pos in claimed_pos:
                continue
            claimed_pos.add(task.pos)
            approved.append(task)
        return approved

    def _payroll_floor(self, world: WorldModel) -> float:
        """Cash kept for tomorrow's hires before any discretionary purchase.

        Without it, a seed spree on a cash-poor morning left too little to hire
        the hands needed to water those same seeds.
        """
        tc = world.classified_tiles
        workload = len(tc.water) + len(tc.feed) + len(tc.care) + len(tc.fert)
        needed = max(0, -(-workload // HIRE_BACKLOG_PER_UNIT) - len(world.units))
        needed = min(needed, MAX_HIRABLE_LOOKAHEAD)
        wages = sum(FARM_HAND_COST_MULT * fib(h) for h in range(needed))
        return MONEY_RESERVE + wages

    def _arbitrate_market(self, world: WorldModel, reqs: List[MarketRequest]) -> List[List[Any]]:
        """Queue orders by priority under running cash and shed ledgers.

        Queue position is settlement order. Sells are credited at the current
        price before the buys behind them are checked; buys that would overflow
        the shed (the engine fails them silently) are trimmed; seeds and land
        must leave the payroll floor intact.
        """
        reqs.sort(key=lambda r: -r.priority)
        payroll_floor = self._payroll_floor(world)
        virtual_shed = sum(world.shed.values())
        cash = float(world.money)
        endgame = endgame_active(world)
        orders: List[List[Any]] = []

        for req in reqs:
            if len(orders) >= MAX_MARKET_ORDERS:
                break
            order = list(req.order)
            op = order[0]

            if op == "SELL":
                qty = int(order[2])
                virtual_shed = max(0, virtual_shed - qty)
                cash += qty * world.prices.get(order[1], 0)
                orders.append(order)

            elif op in ("BUY_PRODUCT", "BUY_ANIMAL"):
                if endgame and op == "BUY_ANIMAL":
                    continue  # cannot pay back; feed top-ups still flow
                item, qty = order[1], int(order[2])
                unit_cost = ANIMALS[item]["cost"] if op == "BUY_ANIMAL" else world.prices.get(item, 0)
                qty = min(qty, max(0, SHED_CAPACITY - virtual_shed))
                if unit_cost > 0:
                    qty = min(qty, int(cash // unit_cost))
                if qty <= 0:
                    continue
                virtual_shed += qty
                cash -= qty * unit_cost
                orders.append([op, item, qty])

            elif op == "BUY_SEED":
                if endgame and req.category != "late_seed":
                    continue
                cost = CROPS[order[1]]["seed"] * int(order[2])
                if cash - cost < payroll_floor:
                    continue
                cash -= cost
                orders.append(order)

            elif op == "HIRE":
                if endgame and world.hour > 6:
                    continue  # an afternoon hire in the endgame earns less than its wage
                if cash < req.cost_estimate:
                    continue
                cash -= req.cost_estimate
                orders.append(order)

            else:  # BUY_LAND
                if endgame:
                    continue
                if cash - req.cost_estimate < payroll_floor:
                    continue
                cash -= req.cost_estimate
                orders.append(order)

        return orders
