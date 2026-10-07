#!/usr/bin/env bash
# Batch-run a range/list of seeds headlessly and print per-seed + summary scores.
# Usage: scripts/run_seeds.sh [seeds] [threads]
#   seeds:   single value, "start,end" range, or comma list (tester.jar -seed syntax)
#   threads: number of parallel worker threads (default 4)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/find_java.sh

SEEDS="${1:-1,50}"
THREADS="${2:-4}"
JAVA_BIN="$(find_java)"

"$JAVA_BIN" -jar tools/tester.jar -exec "python3 src/main.py" \
  -seed "$SEEDS" -threads "$THREADS" -novis -printRuntime
