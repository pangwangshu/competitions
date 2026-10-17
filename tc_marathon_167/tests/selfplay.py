#!/usr/bin/env python3
"""Deterministic tester replacement: drives a WordGame solver binary over the
official stdin/stdout protocol without Java, replicating the feedback loop of
the official WordGameTester.java (TopCoder's tester source):

  initial : N, T, X, D, count, then each dictionary word, then T hand chars,
            X preview chars, then N*N bonus chars (one per write).
  feedback: after every valid PLACE/DISCARD the tester writes an integer
            cumulative-millisecond line, then T hand chars, then X preview
            chars; a hand/preview slot that cannot be refilled is '.'.
  end     : once the deck is exhausted and the hand is all '.', the tester
            writes nothing more (the solver then emits "-1" on its own).

The dictionary is enable2k.txt minus its header line, filtered to length<=N,
duplicates preserved (WordList semantics).  The deck is N*N letters sampled
from dictionary letter frequencies with a seeded RNG; the elapsed-time line
uses a fixed integer schedule so a game is fully deterministic.

Used by tests/test_golden.py to prove that refactors do not change behaviour.
"""
import argparse
import random
import select
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DICTIONARY = ROOT / "tools" / "enable2k.txt"


def load_words(n):
    """WordList: every enable2k.txt line after the header with length<=n."""
    with open(DICTIONARY) as f:
        next(f)  # header comment line, skipped by the tester
        return [w.rstrip("\n") for w in f if len(w.rstrip("\n")) <= n]


def sample_deck(rng, words, count):
    """constructTiles(): letters sampled by their dictionary frequency."""
    freq = [0] * 26
    for w in words:
        for ch in w:
            freq[ord(ch) - ord("a")] += 1
    letters, acc = [], 0
    cumulative = []
    for k in range(26):
        acc += freq[k]
        cumulative.append(acc)
    total = acc
    for _ in range(count):
        a = rng.randrange(total)
        for k in range(26):
            if a < cumulative[k]:
                letters.append(chr(ord("A") + k))
                break
    return letters


def make_bonus(rng, n, density):
    """Bonus grid: per cell one of d/D/t/T with the given density, plus S."""
    share = [rng.uniform(0.2, 1.0) for _ in range(4)]
    s = sum(share)
    share = [x / s * density for x in share]
    grid = []
    for _ in range(n * n):
        a, cell = rng.random(), "."
        if a < share[0]:
            cell = "d"
        elif a < share[0] + share[1]:
            cell = "D"
        elif a < share[0] + share[1] + share[2]:
            cell = "t"
        elif a < share[0] + share[1] + share[2] + share[3]:
            cell = "T"
        grid.append(cell)
    grid[rng.randrange(n * n)] = "S"
    return grid


class SelfPlay:
    """One deterministic game against a solver subprocess."""

    def __init__(self, exe, n, t, x, penalty, density, seed, flags,
                 turn_ms=25, stderr_file=None):
        rng = random.Random(seed)
        self.n, self.t, self.x, self.penalty = n, t, x, penalty
        self.words = load_words(n)
        self.deck = sample_deck(rng, self.words, n * n)
        self.bonus = make_bonus(rng, n, density)
        self.hand = list(self.deck[:t])          # tester: curTiles = allTiles[0..T)
        self.cur_id = t
        self.board = ["."] * (n * n)
        self.elapsed = 0
        self.turn_ms = turn_ms
        self.stderr_file = stderr_file
        err = open(stderr_file, "wb") if stderr_file else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [str(exe)] + list(flags), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=err, cwd=str(ROOT))

    # -- protocol ---------------------------------------------------------
    def initial_input(self):
        out = [str(self.n), str(self.t), str(self.x), str(self.penalty),
               str(len(self.words))]
        out += self.words
        out += self.hand
        out += self.deck[self.cur_id:self.cur_id + self.x]
        out += self.bonus
        return "".join(token + "\n" for token in out)

    def _refill(self):
        for i in range(self.t):
            if self.hand[i] == ".":
                if self.cur_id < self.n * self.n:
                    self.hand[i] = self.deck[self.cur_id]
                    self.cur_id += 1
        preview = [self.deck[self.cur_id + i] if self.cur_id + i < self.n * self.n else "."
                   for i in range(self.x)]

        def has_tiles():
            if self.cur_id < self.n * self.n:
                return True
            return any(ch != "." for ch in self.hand)

        return has_tiles(), [str(self.elapsed)] + self.hand + preview

    def apply(self, line):
        """Apply one solver action; returns feedback lines or None on '-1'."""
        parts = line.split()
        if parts[0] == "-1":
            return None
        if parts[0] == "PLACE":
            r, c = int(parts[1]), int(parts[2])
            dr, dc = (1, 0) if parts[3] == "V" else (0, 1)
            for ch in parts[4]:
                idx = r * self.n + c
                if self.board[idx] == ".":
                    self.board[idx] = ch
                    self.hand[self.hand.index(ch)] = "."
                r, c = r + dr, c + dc
        elif parts[0] == "DISCARD":
            for ch in parts[1]:
                self.hand[self.hand.index(ch)] = "."
        else:
            raise RuntimeError("invalid solver action: " + line)
        self.elapsed += self.turn_ms
        return self._refill()

    # -- plumbing ----------------------------------------------------------
    def _readline(self, timeout):
        """readline() with a wall-clock timeout on the action stream."""
        import time
        end = time.monotonic() + timeout
        buf = b""
        while b"\n" not in buf:
            wait = end - time.monotonic()
            if wait <= 0:
                raise TimeoutError("solver did not answer within %.0fs" % timeout)
            ready, _, _ = select.select([self.proc.stdout], [], [], wait)
            if not ready:
                continue
            chunk = self.proc.stdout.read1(64)
            if not chunk:
                if buf:
                    break
                return None  # EOF
            buf += chunk
        line, _, rest = buf.partition(b"\n")
        if rest:
            raise RuntimeError("protocol overrun; solver wrote more than one line")
        return line.decode()

    def run(self, timeout=240):
        """Play the whole game; returns the list of solver action lines."""
        self.proc.stdin.write(self.initial_input().encode())
        self.proc.stdin.flush()
        trace = []
        while True:
            line = self._readline(timeout)
            if line is None:
                raise RuntimeError("solver exited early after %d actions" % len(trace))
            trace.append(line)
            feedback = self.apply(line)
            if feedback is None:
                break
            has_tiles, lines = feedback
            self.proc.stdin.write(("".join(tok + "\n" for tok in lines)).encode())
            self.proc.stdin.flush()
            if not has_tiles:
                closing = self._readline(timeout)
                if closing != "-1":
                    raise RuntimeError("expected final -1, got %r" % closing)
                trace.append(closing)
                break
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            self.proc.wait(timeout=30)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream and stream is not subprocess.DEVNULL:
                try:
                    stream.close()
                except OSError:
                    pass
        if self.stderr_file:
            err.close()
        if self.proc.returncode != 0:
            raise RuntimeError("solver exited with code %d" % self.proc.returncode)
        return trace


def run_game(exe, n, t, x, penalty, density, seed, flags, stderr_file=None):
    return SelfPlay(exe, n, t, x, penalty, density, seed, flags,
                    stderr_file=stderr_file).run()


# Golden regression matrix: (name, n, t, x, penalty, bonus density, seed, flags).
# Small boards keep every turn far inside its wall-clock slice, so decisions
# come from the deterministic cost model and the trace is machine-stable.
# The p2-03 games were captured from the pre-refactor single-file solver; the
# final_* games from the O-01C package that was the base of the final submission.
P203 = ["--preset", "p2-03"]
GAMES = [
    ("default_n10_a", 10, 7, 3, 2, 0.30, 11, P203),
    ("default_n10_x0", 10, 5, 0, 5, 0.10, 12, P203),
    ("default_n12", 12, 9, 5, 0, 0.50, 13, P203),
    ("default_n20", 20, 7, 2, 3, 0.30, 14, P203),
    ("greedy_n10", 10, 7, 3, 2, 0.30, 15, P203 + ["--no-lookahead"]),
    ("discard2_n10", 10, 7, 3, 5, 0.30, 16, P203 + ["--discard", "2", "--discard-stop"]),
    ("rollout_n10", 10, 7, 3, 2, 0.30, 17, P203 + ["--rollout"]),
    ("chancetree_n10", 10, 7, 3, 2, 0.30, 18, P203 + ["--chance-tree"]),
    ("adaptive2_n10", 10, 7, 3, 2, 0.30, 19, P203 + ["--adaptive", "2", "--lean-x", "2"]),
    ("fishroot_n10", 10, 7, 3, 2, 0.30, 20, P203 + ["--discard", "2", "--fish-root"]),
    ("laseed42_n10", 10, 6, 4, 1, 0.20, 21, P203 + ["--la-seed", "42"]),
    ("discard1_n12", 12, 5, 1, 4, 0.20, 22, P203 + ["--discard", "1"]),
    ("final_n10", 10, 7, 3, 2, 0.30, 23, []),
    ("final_n12_x0", 12, 9, 0, 5, 0.50, 24, []),
    ("final_n14", 14, 6, 5, 0, 0.40, 25, []),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=ROOT / "build" / "WordGame")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--t", type=int, default=7)
    ap.add_argument("--x", type=int, default=3)
    ap.add_argument("--penalty", type=int, default=2)
    ap.add_argument("--density", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--flags", nargs="*", default=[])
    ap.add_argument("--stderr-file", type=Path, default=None)
    args = ap.parse_args()
    for line in run_game(args.exe, args.n, args.t, args.x, args.penalty,
                         args.density, args.seed, args.flags, args.stderr_file):
        print(line)


if __name__ == "__main__":
    main()
