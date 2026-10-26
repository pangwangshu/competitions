# Provenance of the numbers

The write-up draws on two sources. Numbers from runs made for the report (§8.2–8.5) come from the tables in [`data/`](data/), which this repository can regenerate (Appendix B of the [write-up](writeup.md)). Numbers from the match itself (§4–7, the standings, the fishing mechanism experiment) come from my development repository, `marathon_167`, which is private. Its paths are given below as plain text; the underlying files are available on request. Some raw `results/` and `dist/` artifacts there are local archives that are not in its Git history.

The development repository's last commits are `c501cb8` (final score), `f2517d1` (refactor) and `1039302` (final submission). A commit title does not record which ZIP was uploaded.

## Where each claim comes from

| Claim | Source in `marathon_167` | Scope |
|---|---|---|
| Final 40.56866, rank 41; provisional 36.46748 | `eval_results/final scores.pdf` (saved standings page) | Does not identify which budget variant was uploaded last |
| INT-01 acceptance protocol | `plan/results/data/INT-01-PREREGISTER.md`, `INT-01-gates.json` | All 32 recorded gates pass; I checked the record, not every gate re-run |
| Fishing mechanism and four-arm screen (§5.5) | `plan/results/O-01A.md`, `plan/results/data/O-01A-mechanism_summary.json`, `O-01A-screen_summary.json`, `results/O-01A/mechanism_continuation.jsonl` | 40 selected states from 37 games, 16 tapes each, uncapped continuation |
| Frozen O-01A validation parameters | `plan/results/data/O-01A-dev2-FREEZE.json` | Separate from the earlier mechanism versions |
| Tile-economy statistics, including 891 of 1,230 (§9) | `analysis/tile-economy-independent-review.md`, `verify_tile_economy.py`, `tile-economy-verification.json` | INT-01 seeds 2001–2200; descriptive |
| Local/online discrepancy (§7.4) | `plan/results/O-01B-audit.md` | Tested platform approximations only |
| O-01C base and budget variants | `plan/results/data/O-01C-package.py`, `O-01C-time-variants.py`, `dist/O-01C/WordGame.cpp` | Base parameters known; the variant uploaded last is not |
| Release scores in `data/final5000.csv` | `results/final5000/` (INT-01, O-01B, O-01C chunk CSVs) | 5,000 matching seeds per build; the set also informed the STOP fix |
| INT-01 at 4,500 / 6,000 ms (§6) | `results/final_day/cmp_w4500.json`, `cmp_w6000.json` | G = 0.995352 and 0.997375; not O-01C experiments |
| Local normalized-score proxy | `tools/score_audit.py` | Local panel MAX, not the hidden online MAX |

## What I re-checked after the match (17 October 2026)

- Built this repository's source in an empty CMake directory with AppleClang 21; all five C++ suites and the golden-trace regression pass.
- Compiled the frozen O-01C base source and compared it with the refactor on three self-play games (N = 10/12/14, X = 3/0/4): all 153 actions match.
- The dictionary and tester have identical SHA256 values in both repositories.
- All 15,000 release scores in `data/final5000.csv` match the original chunk CSVs, with no failed rows; O-01C / INT-01 G = 1.02887.
- Re-ran the tile-economy extraction (matches the archived JSON exactly) and re-aggregated the 40-state continuation tapes (both mechanism effect sizes reproduce to within rounding).
- Regenerated all 11 figures with the versions pinned in [`requirements.txt`](requirements.txt).

## Added on 26 October 2026 (§11)

- Dumped the bonus grids of seeds 30001–30200 from the official tester with `line_bound.py --dump`, computed each line bound and joined it with `data/anatomy_final.csv` into `data/line_bound.csv`. Seed 30159's grid was dumped twice and matched.
- The runner-up's numbers in §11 are quoted from their write-up and not reproduced.

## Known gaps

- **Refactor equivalence.** I did not recover the per-seed logs of the original 200-game refactor comparison, including its 11 serial repeats. The golden traces and the three-game check above support it but do not replace it.
- **Final budget variant.** Which of the six O-01C probes was uploaded last is unknown, and no local comparison can identify it uniquely. The no-argument binary is the O-01C base configuration.
- **Stage gains.** The per-stage percentages come from different historical parents and seed sets; they are not a controlled decomposition of the fresh 1.65×.
- **Release set reuse.** The 5,000-game set exposed the STOP failure, so its post-fix gain is descriptive. A new strategy claim needs a frozen candidate and untouched seeds.
