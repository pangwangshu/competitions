# TopCoder offline tester

The scripts in `scripts/` drive TopCoder's official offline tester for Marathon Match 166. It generates the test case for any seed, runs a solution over stdin/stdout, scores it, and can show a visualizer. The tester is TopCoder's code and is not redistributed in this repository.

To use it, download **Visualizer Binary (`tester.jar.zip`)** from the Downloads section of the [match page](https://www.topcoder.com/challenges/15f1c681-42b2-4f62-99c7-6f61fe5c57d9), unzip it, and place `tester.jar` in this folder. It needs Java 11 or newer. On macOS, `scripts/find_java.sh` also looks for a Homebrew OpenJDK, because `/usr/bin/java` is a stub when no JDK is installed.

Useful tester options (the full list is in TopCoder's [local tester guide](https://www.topcoder.com/thrive/articles/marathon-match-local-tester-parameters)):

| Option | Effect |
|---|---|
| `-seed 1,250` | A seed or a range of seeds. Seed 1 uses the smallest board, seed 2 the largest. |
| `-threads 4` | Run several seeds in parallel |
| `-novis` | No visualizer |
| `-printRuntime` | Report each case's runtime |
| `-saveSolInput DIR` / `-saveSolOutput DIR` | Save each case's input and the solution's output |
| `-showOriginal` | Give the solution the hidden, unscrambled board instead of the scrambled one |
