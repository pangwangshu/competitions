# Evidence and reproduction index

Audit date: 17 October 2026. This report lives in the presentation repository; the development repository is `marathon_167`, a sibling of `competitions` under `src/`. The relative links below require that layout and access to the development checkout. They are provenance pointers, not files bundled with this report. Some raw `results/` and `dist/` artifacts are local archives excluded from Git; cloning the development repository alone may not recover them.

The development checkout inspected during the audit was at `c501cb8` (`final score`), following `f2517d1` (`refactor`) and `1039302` (`final submission`). A commit title is not evidence of which ZIP was uploaded.

## Claims and their sources

| Claim or analysis | Development source | Evidence boundary |
|---|---|---|
| Final 40.56866, rank 41; provisional 36.46748 | [Saved standings PDF](../../../marathon_167/eval_results/final%20scores.pdf) | Saved page, not a new online query; does not identify the uploaded budget variant |
| INT-01 acceptance protocol | [Preregistration](../../../marathon_167/plan/results/data/INT-01-PREREGISTER.md), [32 gates](../../../marathon_167/plan/results/data/INT-01-gates.json) | All recorded gates pass; this audit inspected the record rather than re-running all historical gates |
| Fishing mechanism and four-arm screen | [O-01A report](../../../marathon_167/plan/results/O-01A.md), [summary](../../../marathon_167/plan/results/data/O-01A-mechanism_summary.json), [raw continuations](../../../marathon_167/results/O-01A/mechanism_continuation.jsonl) | 40 selected states from 37 games, 16 tapes each, uncapped mechanism continuation; +275.95 root-only versus +457.52 combined effect |
| Frozen O-01A validation parameters | [Freeze record](../../../marathon_167/plan/results/data/O-01A-dev2-FREEZE.json) | Separate from the earlier mechanism and development versions |
| Tile economy and causal limitations | [Independent review](../../../marathon_167/analysis/tile-economy-independent-review.md), [recomputation script](../../../marathon_167/analysis/verify_tile_economy.py), [archived result](../../../marathon_167/analysis/tile-economy-verification.json) | INT-01 seeds 2001–2200, RNG 167; adjacency and phase statistics are descriptive |
| Local/online discrepancy | [O-01B audit](../../../marathon_167/plan/results/O-01B-audit.md) | Tested platform approximations do not exclude all platform differences or unknown per-case MAX effects |
| O-01C frozen base and budget variants | [Base packager](../../../marathon_167/plan/results/data/O-01C-package.py), [variant packager](../../../marathon_167/plan/results/data/O-01C-time-variants.py), [base source](../../../marathon_167/dist/O-01C/WordGame.cpp) | Base parameters are known; the variant uploaded last is not identified |
| Release scores in `data/final5000.csv` | [INT-01 chunks](../../../marathon_167/results/final5000/run_INT-01/), [O-01B chunks](../../../marathon_167/results/final5000/run_O-01B-exact/), [O-01C X=0 chunks](../../../marathon_167/results/final5000/x0_SD/), [O-01C X>0 chunks](../../../marathon_167/results/final5000/x1p_SD/) | 5,000 matching seeds per build; the set informed the STOP fix and is not untouched confirmation for it |
| INT-01 larger-budget results | [4,500 ms comparison](../../../marathon_167/results/final_day/cmp_w4500.json), [6,000 ms comparison](../../../marathon_167/results/final_day/cmp_w6000.json) | G = 0.995352 and 0.997375; these are not O-01C budget experiments |
| Local normalized-score proxy | [score_audit.py](../../../marathon_167/tools/score_audit.py) | Local panel MAX only, not an estimate of the hidden online MAX |

## Independent checks completed

- Built the presentation source in an empty temporary CMake directory with AppleClang 21; all five C++ suites and the golden Python regression passed.
- Compiled the original frozen O-01C base source and compared it with the refactor on three deterministic self-play games (N=10/12/14, X=3/0/4): all 153 actions matched. This is a new spot check, not a re-run of the claimed 200-game equivalence experiment.
- Verified identical SHA256 values for the dictionary and tester in the two repositories.
- Matched all 15,000 reported release scores against the three builds' original chunk CSVs, with no discrepancies or failed status rows. Recomputed O-01C / INT-01 G = 1.02887248. These were data checks, not 15,000 new solver runs.
- Re-ran the 200-game INT-01 tile-economy extraction; the result matched the archived JSON exactly. Independently aggregated the 40-state continuation tapes and reproduced both mechanism effect sizes (differences below 0.001 from rounding).
- Regenerated all 11 figures using matplotlib 3.11.1 and numpy 2.5.2 from the existing `marathon_166/.venv` environment. `requirements.txt` records those versions; no dependency on that particular local environment is required.

## Remaining limits

The checked-in fresh 200-seed tables and action-equivalence counts are report artifacts. This audit did not locate and independently replay the complete original 200-game refactor comparison, including its 11 serial guard-case repeats. The new three-game comparison does not establish that larger claim by itself. Preserve or publish those per-seed action hashes and logs before treating the report as a fully self-contained reproducibility package.

The exact final online budget variant remains unknown. Keep the base configuration, each probe and the online result distinct. Recovering an upload record or the actual submitted source would close this gap; no local score comparison can identify it uniquely.

The stage uplift percentages use different historical parents and seed sets. They should not be read as a single controlled decomposition of the fresh 1.65× result. The 5,000-case post-fix interval is descriptive after adaptive use of that set. Future strategy claims need a newly frozen candidate and untouched confirmation cases.
