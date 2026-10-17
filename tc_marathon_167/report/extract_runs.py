#!/usr/bin/env python3
"""Turn tester runs into the small CSV files the write-up plots.

    python3 report/extract_runs.py runs report/data

expects, for every build name in ORDER, the output of
    scripts/run_tester.py --seeds 30001:30200 --args="<flags> --diagnostics" \
        --save runs/<build> > runs/<build>.csv
i.e. runs/<build>.csv (seed, score, runtime_ms, ok) and runs/<build>/<seed>.diag
(the solver's DIAG lines). Builds missing from runs/ are skipped in the lineage
table; O-01C and B0 are required for the per-game anatomy. report/data was
produced by the same extraction applied to runs made with the development
repository's harness (same tester, flags and seeds; Appendix B). The serial
re-runs behind serial_probes.csv are tabulated separately.
"""
import csv
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import hashlib

RUNS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ['B0', 'P1-01', 'P1-02', 'P2-03', 'INT-01', 'O-01A', 'O-01C',
         'O-01C-B5000', 'O-01C-B3000', 'O-01C-W2000', 'O-01C-W1500', 'O-01C-W1000', 'O-01C-W700',
         'refactor-O-01C']
WM = {'D': 2, 'T': 3}
LM = {'d': 2, 't': 3}


def load(version):
    """Rows keyed by seed (score, runtime, status, action hash) and DIAG streams."""
    rows, diags = {}, {}
    table = RUNS / (version + '.csv')
    if not table.exists():
        return rows, diags
    for r in csv.DictReader(open(table)):
        seed = int(r['seed'])
        diag_file = RUNS / version / ('%d.diag' % seed)
        diag = [json.loads(line[5:]) for line in open(diag_file)] if diag_file.exists() else []
        actions = '\n'.join(d['action'] for d in diag if d.get('type') == 'turn')
        rows[seed] = dict(score=r['score'], runtime_ms=r['runtime_ms'], status='ok' if r['ok'] == '1' else 'failed',
                          action_sha256=hashlib.sha256(actions.encode()).hexdigest())
        diags[seed] = diag
    return rows, diags


def replay(diag):
    """Per-game anatomy from one DIAG stream."""
    init = next(d for d in diag if d.get('type') == 'initial')
    end = next((d for d in diag if d.get('type') == 'end'), {})
    n, t = init['N'], init['T']
    bonus = init['bonus']
    cell = lambda r, c: bonus[r * n + c]
    board = [['.'] * n for _ in range(n)]
    plays, discards, discarded_tiles, burned = [], 0, 0, defaultdict(int)
    rare = defaultdict(int)  # destination of J Q X Z K tiles
    phase_score = [0.0] * 4
    phase_tiles = [0] * 4
    work = []
    for d in diag:
        if d.get('type') != 'turn':
            continue
        frac = (n * n - d['remaining'] - t) / (n * n) if d['remaining'] >= 0 else 1
        q = min(3, max(0, int(4 * (n * n - d['remaining']) / (n * n + 1e-9))))
        if 's_work' in d:
            work.append(((n * n - d['remaining']) / (n * n), d['s_work'] + d.get('dc_work', 0), d.get('s_cap', 0)))
        act = d['action'].split()
        if act[0] == 'DISCARD':
            discards += 1
            discarded_tiles += len(act[1])
            for ch in act[1]:
                if ch in 'JQXZK':
                    rare['discarded'] += 1
            phase_tiles[q] += len(act[1])
            continue
        if act[0] != 'PLACE':
            continue
        r, c, dr, w = int(act[1]), int(act[2]), act[3] == 'V', act[4]
        prod, new = 1, 0
        cells, placed_now = [], []
        for k, ch in enumerate(w):
            rr, cc = (r + k, c) if dr else (r, c + k)
            if board[rr][cc] == '.':
                board[rr][cc] = ch
                new += 1
                placed_now.append(ch)
                b = cell(rr, cc)
                prod *= WM.get(b, 1)
                cells.append(b)
        sc = d['score']
        cls = 'x1' if prod == 1 else 'x2-8' if prod <= 8 else 'x9+'
        for ch in placed_now:
            if ch in 'JQXZK':
                rare[cls] += 1
        for b in cells:
            if b in WM:
                burned[b, min(prod, 81)] += 1
        plays.append(dict(score=sc, tiles=new, product=prod, phase=q, turn=d['turn']))
        phase_score[q] += sc
        phase_tiles[q] += new
    # Best product of word multipliers over T consecutive cells of one line.
    best = 1
    for line in range(n):
        for horiz in (0, 1):
            seq = [WM.get(cell(line, k) if horiz else cell(k, line), 1) for k in range(n)]
            for s in range(n):
                p = 1
                for k in range(s, min(n, s + t)):
                    p *= seq[k]
                    best = max(best, p)
    total = sum(p['score'] for p in plays)
    top = sorted((p['score'] for p in plays), reverse=True)
    return dict(n=n, t=t, x=init['X'], d=init['D'], density=init['bonus_density'], bonus=bonus,
                plays=plays, discards=discards, discarded_tiles=discarded_tiles,
                placed_tiles=sum(p['tiles'] for p in plays), total=total,
                top1=top[0] if top else 0, top3=sum(top[:3]), top5=sum(top[:5]),
                max_product=max((p['product'] for p in plays), default=1), board_best=best,
                phase_score=phase_score, phase_tiles=phase_tiles, burned=burned, work=work, rare=rare,
                n9=sum(p['product'] >= 9 for p in plays), n27=sum(p['product'] >= 27 for p in plays),
                end=end, board=board)


def geo(c, r):
    return math.exp(statistics.mean(math.log((a + 1) / (b + 1)) for a, b in zip(c, r)))


def main():
    global ORDER
    data = {v: load(v) for v in ORDER}
    ORDER = [v for v in ORDER if data[v][0]]
    seeds = sorted(set.intersection(*(set(data[v][0]) for v in ORDER)))
    print('seeds complete for all versions:', len(seeds))
    score = lambda v, s: max(0.0, float(data[v][0][s]['score'] or 0))
    # 1. lineage summary
    with open(OUT / 'lineage.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['version', 'n', 'failed', 'mean', 'median', 'G_vs_B0', 'G_vs_O01C', 'win_vs_prev', 'runtime_p50_ms', 'runtime_p95_ms', 'runtime_max_ms', 'guard_turns', 'identical_to_O01C'])
        prev = None
        for v in ORDER:
            sc = [score(v, s) for s in seeds]
            rt = sorted(int(data[v][0][s]['runtime_ms'] or 0) for s in seeds)
            failed = sum(data[v][0][s]['status'] != 'ok' for s in seeds)
            guard = sum(next((d.get('sched_guard_turns', 0) for d in data[v][1].get(s, []) if d.get('type') == 'end'), 0) for s in seeds)
            same = sum(data[v][0][s]['action_sha256'] == data['O-01C'][0][s]['action_sha256'] for s in seeds)
            b0 = [score('B0', s) for s in seeds]
            oc = [score('O-01C', s) for s in seeds]
            win = '' if prev is None else sum(a > b for a, b in zip(sc, [score(prev, s) for s in seeds]))
            w.writerow([v, len(seeds), failed, round(statistics.mean(sc), 1), statistics.median(sc), round(geo(sc, b0), 4),
                        round(geo(sc, oc), 4), win, rt[len(rt) // 2], rt[int(0.95 * (len(rt) - 1))], rt[-1], guard, same])
            if v in ORDER[:7]:
                prev = v
    # 2. per-seed scores of every version
    with open(OUT / 'lineage_seeds.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed'] + ORDER)
        for s in seeds:
            w.writerow([s] + [score(v, s) for v in ORDER])
    # 3. per-seed anatomy of the final build and of B0
    for v, name in (('O-01C', 'final'), ('B0', 'b0')):
        games = {s: replay(data[v][1][s]) for s in seeds}
        with open(OUT / f'anatomy_{name}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['seed', 'N', 'T', 'X', 'D', 'bonus_density', 'score', 'plays', 'placed_tiles', 'discards', 'discarded_tiles',
                        'top1', 'top3', 'top5', 'max_product', 'board_best_product', 'q1', 'q2', 'q3', 'q4',
                        'tiles_q1', 'tiles_q2', 'tiles_q3', 'tiles_q4', 'plays_x9', 'plays_x27', 'end_reason'])
            for s, g in games.items():
                w.writerow([s, g['n'], g['t'], g['x'], g['d'], g['density'], score(v, s), len(g['plays']), g['placed_tiles'], g['discards'],
                            g['discarded_tiles'], g['top1'], g['top3'], g['top5'], g['max_product'], g['board_best']]
                           + [round(x, 1) for x in g['phase_score']] + g['phase_tiles'] + [g['n9'], g['n27'], g['end'].get('reason', '')])
        # per-play buckets
        edges = [0, 5, 10, 20, 50, 100, 300, 1e9]
        tiles, pts, cnt = [0] * 7, [0] * 7, [0] * 7
        prod_pts = defaultdict(float)
        prod_cnt = defaultdict(int)
        for g in games.values():
            for p in g['plays']:
                ppt = p['score'] / max(1, p['tiles'])
                k = next(i for i in range(7) if edges[i] <= ppt < edges[i + 1])
                tiles[k] += p['tiles']; pts[k] += p['score']; cnt[k] += 1
                pb = 1 if p['product'] == 1 else 2 if p['product'] <= 3 else 4 if p['product'] <= 8 else 9 if p['product'] <= 26 else 27
                prod_pts[pb] += p['score']; prod_cnt[pb] += 1
        with open(OUT / f'per_tile_{name}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['bucket_lo', 'bucket_hi', 'plays', 'tiles', 'points'])
            for i in range(7):
                w.writerow([edges[i], edges[i + 1] if edges[i + 1] < 1e8 else '', cnt[i], tiles[i], pts[i]])
        with open(OUT / f'product_{name}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['product_class', 'plays_per_game', 'points_per_game'])
            for pb, label in ((1, '1'), (2, '2-3'), (4, '4-8'), (9, '9-26'), (27, '>=27')):
                w.writerow([label, round(prod_cnt[pb] / len(games), 2), round(prod_pts[pb] / len(games), 1)])
        if name == 'final':
            # word-multiplier cells consumed, by the product of the play that consumed them
            burned = defaultdict(int)
            for g in games.values():
                for (b, prod), k in g['burned'].items():
                    burned[prod] += k
            with open(OUT / 'multiplier_cells_final.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['play_product', 'word_multiplier_cells'])
                for prod in sorted(burned):
                    w.writerow([prod, burned[prod]])
            rare = defaultdict(int)
            for g in games.values():
                for k, cnt in g['rare'].items():
                    rare[k] += cnt
            with open(OUT / 'rare_letters_final.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['destination', 'tiles'])
                for k in ('discarded', 'x1', 'x2-8', 'x9+'):
                    w.writerow([k, rare[k]])
            # effort spent by game progress
            with open(OUT / 'effort_final.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['seed', 'N', 'progress', 'work_ms', 'cap_ms'])
                for s, g in games.items():
                    for prog, wk, cap in g['work']:
                        w.writerow([s, g['n'], round(prog, 4), round(wk, 3), round(cap, 3)])
            # one board for figure 1: the median-scoring game with N between 13 and 15
            cand = sorted((s for s, g in games.items() if 13 <= g['n'] <= 15), key=lambda s: score(v, s))
            pick = cand[len(cand) // 2]
            g = games[pick]
            json.dump(dict(seed=pick, N=g['n'], T=g['t'], X=g['x'], D=g['d'], bonus=g['bonus'], score=score(v, pick),
                           board=[''.join(r) for r in g['board']], plays=g['plays'],
                           actions=[d['action'] for d in data[v][1][pick] if d.get('type') == 'turn']),
                      open(OUT / 'board_final.json', 'w'), indent=1)
            print('board seed', pick, 'N', g['n'], 'score', score(v, pick))


if __name__ == '__main__':
    main()
