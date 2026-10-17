# Evidence and audit scope

This index accompanies the [write-up](writeup.md). Revision audit: 17 October 2026. Paths in the development repository refer to `marathon_166`; those files are not all redistributed in this public report. The report's existing 250-seed run is an archive, not a new experiment performed during this revision.

## Sources

| Claim | Source | What it establishes |
|---|---|---|
| Final 51.66780324147004, rank 55; last provisional 52.35924643291182, rank 53 | Development `result/final scores.166.pdf`, page 23, `wash` row | Exact correspondence with the logged last provisional reading; the PDF does not identify an uploaded source hash |
| Winner 99.7902204561286; tenth place 97.56359793156847; 75 entrants | Same PDF, pages 1–5 and the complete 32-page extraction | Final aggregate standings, not per-case scores or algorithms; all rows in [data/leaderboard.csv](data/leaderboard.csv) |
| Development timeline and shipped/rejected changes | Development `plan/optimization-roadmap.md`, `plan/smart-optimization.md`, `plan/HexTiles-Postmortem.md`; report `data/submissions.csv`, `data/changes.csv` | Historical comparisons against different parents and seed sets; do not multiply stage gains into a controlled decomposition |
| Solution anatomy and hidden-board comparison | [data/seeds.csv](data/seeds.csv), [data/match_by_distance.csv](data/match_by_distance.csv), `scripts/summarize_run.py` | 250 archived local cases; not the 5,000 official final cases |
| Seed-199 illustration | `data/board_seed199.in`, `data/board_seed199.out` | A retained input/output pair that can be replayed |
| Current implementation and defaults | `src/hextiles/solver.py`, `incremental.py`, `geometry.py`, `model.py`, `io_utils.py`, `src/main.py` | Actual pipeline, deadline-check locations and scoring semantics |
| Joint feasibility and official scoring | Development `tools/tester_src/HexTilesTester.java` | Generator defines pairs from a hidden orientation board, then scrambles it; feasibility does not imply high score |

PDF SHA256: `e5c135034e7861d1170aa2e6ac5cd9b69b4204e29bdfee822e85d156cbb399ab`. It is retained at `marathon_166/result/final scores.166.pdf`. Displayed submission timestamps in the CSV are copied as shown; the saved page does not specify their timezone. Empty provisional fields remain empty and `N/A` stays `N/A`.

## Checks repeated for this revision

- Extracted all 75 standings rows, retaining exact decimal strings. Visually inspected PDF pages 1, 5 and 23 for winner, tenth place and our row; final ranks include a three-way tie at 73.
- Re-aggregated all 250 rows: pooled construction match rate 69.6801%, final match rate 75.7502%; median dominant-path share 79.4662%, median path length / tile count 1.43664. Median final/construction score is 4.05851 across 248 positive construction scores; two zero denominators are excluded.
- Checked 156/250 dominant paths visit every bonus tile; hidden boards score zero in 89 cases and beat our solutions in none. Archived mean runtime is 9,523.668 ms, maximum 9,538 ms. These runtime records are not new timing measurements.
- Replayed seed 199: 34/39 pairs, 95 rotations, path-score sum 2,304, raw score 75,106. The dominant path has 202 chords, 9 bonuses and value 2,020 (87.67% of total path score).
- Ran `python3 -m pytest -q -p no:cacheprovider .../tc_marathon_166/tests`: **105 passed in 61.59 s**. These exercise evaluator and operator consistency; they are not an end-to-end grader timeout certificate.
- Compared abstract syntax trees of all seven Python files under `src/` against the development checkout: all match. No implementation change was made.
- Reviewed the solver's control flow: the search reserve is estimated; probing depends on planned list size; tie-breaking uses bonus-weighted path score; route deadlines are checked between widening rounds, with up to four searches inside a conflict-retrying call.

## Remaining limits

- The saved standings establish final results but do not give per-case failures, raw scores or leaders' source. No claim about their algorithm is verified here.
- Full raw inputs/outputs and tester logs for the original 250-seed run are not checked into this report. The historical 250/250 agreement was not re-run during this audit; aggregate arithmetic and the retained example were checked.
- `summarize_run.py` shares `trace_path` and `evaluate_grid` with the solver. Its comparison to the Java tester is useful, but describing the Python replay script as fully independent overstates the separation.
- Syntax-tree equality between two current checkouts cannot establish that either is byte-for-byte the uploaded submission. An uploaded-bundle hash or submission artifact is still needed for that stronger claim.
- The old postmortem's 21–31% match-rate diagnosis, inferred leader architecture and 100–1,000× compiled speed-up are not accepted as evidence. The revised report retains the measured negative results and states testable alternatives.
- Existing figures 1–5 retain their archived data and were not regenerated. Figure 2 is a provisional submission timeline, not a final-standings chart. The added redesign diagram is a proposal, not measured performance.
