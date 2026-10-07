from dataclasses import dataclass, field

from hextiles.geometry import grid_width


@dataclass
class GridState:
    n: int
    m: int
    b: int
    pairs: list
    grid: list
    bonus: set
    w: int = field(init=False)

    def __post_init__(self):
        self.w = grid_width(self.n)


def read_input(stream):
    n = int(stream.readline())
    m = int(stream.readline())
    b = int(stream.readline())
    p = int(stream.readline())

    pairs = []
    for _ in range(p):
        a, c = map(int, stream.readline().split())
        pairs.append((a, c))

    w = grid_width(n)
    grid = [[int(stream.readline()) for _ in range(w)] for _ in range(w)]

    bonus = set()
    for _ in range(b):
        r, c = map(int, stream.readline().split())
        bonus.add((r, c))

    return GridState(n=n, m=m, b=b, pairs=pairs, grid=grid, bonus=bonus)
