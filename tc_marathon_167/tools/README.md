# External files (not redistributed)

Two files from the [Marathon Match 167 page](https://www.topcoder.com/challenges/0b11e0fd-3e2d-46dc-8f7b-27c9f49d63cb) go here:

| File | Source on the match page | Used by |
|---|---|---|
| `tester.jar` | "Visualizer Binary - tester.jar.zip" (unzip it) | `scripts/run_tester.py`, `make bench` |
| `enable2k.txt` | "Dictionary - enable2k.txt" | the tester (it reads the dictionary from its working directory), `tests/selfplay.py` and the golden tests |

The solver itself needs neither: the tester sends it the dictionary on standard input. Without `enable2k.txt` the golden-trace tests are skipped and the C++ unit tests still run.
