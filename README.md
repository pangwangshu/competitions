# Competitions

Write-ups and code from machine-learning, simulation and optimization competitions.

| Project | Competition | Summary |
|---|---|---|
| [kaggriculture](kaggriculture/) | Kaggle simulation, Aug–Sep 2026 | A planning agent for a two-player farming simulation: layered rule pipeline with min-cost task assignment, a retired hierarchical-RL variant, and a paired-evaluation methodology for an environment with shared randomness. [Paper](kaggriculture/report/paper.md). |
| [tc_marathon_166](tc_marathon_166/) | TopCoder Marathon Match 166, Aug 2026 | A heuristic solver for a hex-tile path-connection puzzle: Dijkstra routing over a (tile, entry-edge) state graph, basin-hopping search driven by a proven incremental evaluator, contention-aware rerouting, and an analysis of why one long bonus-threaded path dominates the score. [Write-up](tc_marathon_166/report/writeup.md). |
| [tc_marathon_167](tc_marathon_167/) | TopCoder Marathon Match 167, Oct 2026 | A C++ solver for a single-player Scrabble-like game: a five-stage per-turn pipeline (trie generation, rack-aware multiplier-lane prefilter, two-ply successive halving, shared-tape rollouts, a unified place/discard/stop decision) budgeted in deterministic effort units, and an analysis of why the score is a few multiplier jackpots that a move-by-move search does not plan for. [Write-up](tc_marathon_167/report/writeup.md). |

## License

[MIT](LICENSE)
