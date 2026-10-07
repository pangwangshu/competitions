import sys

from hextiles.model import read_input
from hextiles.io_utils import write_output
from hextiles.solver import solve


def main():
    state = read_input(sys.stdin)
    moves = solve(state)
    write_output(moves, sys.stdout)


if __name__ == "__main__":
    main()
