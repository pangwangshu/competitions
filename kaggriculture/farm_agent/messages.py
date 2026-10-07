"""Messages passed between pipeline stages."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TaskRequest:
    """A unit task proposed by an advisor: go to `pos` and perform `action`."""

    category: str                      # "feed", "water", "harvest", "place", "empty", ...
    pos: Tuple[int, int]               # target tile (x, y)
    priority: float                    # higher wins the tile; see config PRIO_*
    action: Optional[List[Any]] = None  # e.g. ["WATER"], ["PLANT", "MELON"]
    requires_inv: Dict[str, int] = field(default_factory=dict)  # items the unit must carry
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketRequest:
    """A market order proposed by an advisor."""

    order: List[Any]                   # e.g. ["SELL", "MELON", 5], ["HIRE"]
    priority: float                    # queue position = settlement order
    category: str = "general"
    cost_estimate: float = 0.0


@dataclass
class TileClassification:
    """Own-farm tiles grouped by the work they need this turn."""

    harvest: List[Tuple[int, int]] = field(default_factory=list)
    water: List[Tuple[int, int]] = field(default_factory=list)
    weed: List[Tuple[int, int]] = field(default_factory=list)
    empty: List[Tuple[int, int]] = field(default_factory=list)
    feed: List[Tuple[int, int]] = field(default_factory=list)
    care: List[Tuple[int, int]] = field(default_factory=list)
    fert: List[Tuple[int, int]] = field(default_factory=list)        # fertilizer to collect
    apply_fert: List[Tuple[int, int]] = field(default_factory=list)  # fertilizer to apply today
    # Members of `water` whose watering today changes no output. They stay in
    # `water` (hiring and payroll read its length) but get no task.
    water_skip: Set[Tuple[int, int]] = field(default_factory=set)
    # Animal harvests deliberately left on the tile tonight (shed would overflow).
    deferred: Set[Tuple[int, int]] = field(default_factory=set)
    unoccupied_structure: List[Tuple[int, int, str]] = field(default_factory=list)  # (x, y, kind)
    animal_count: int = 0
    plant_count: int = 0
