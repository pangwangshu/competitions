#!/usr/bin/env bash
# Run a single seed against the current Python solution, with visualization.
# Usage: scripts/run_seed.sh [seed] [extra tester.jar args...]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/find_java.sh

SEED="${1:-1}"
shift || true
JAVA_BIN="$(find_java)"

"$JAVA_BIN" -jar tools/tester.jar -exec "python3 src/main.py" -seed "$SEED" "$@"
