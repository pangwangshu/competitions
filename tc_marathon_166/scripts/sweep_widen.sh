#!/usr/bin/env bash
# Sweep Tier 2.5 widen configs against the real tester, one config at a time
# (sequential -- parallel runs would contend CPU and distort each run's
# wall-clock deadline behavior). Usage:
#   scripts/sweep_widen.sh "<seeds>" <threads> <outdir> [label:strategy:rounds:pairs ...]
# Default config list covers the take-2 hypotheses if none are given.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${1:-1,20}"
THREADS="${2:-4}"
OUTDIR="${3:-/tmp/t25-sweep}"
shift 3 || true

if [[ $# -gt 0 ]]; then
  CONFIGS=("$@")
else
  CONFIGS=(
    "off:none:0:5"
    "r1p5:random:1:5"
    "r2p5:random:2:5"
    "r3p5:random:3:5"
    "r2p10:random:2:10"
  )
fi

mkdir -p "$OUTDIR"
# Use per-(label, seed-range) output files so re-running a different range
# never clobbers a previous one (earlier take-2 sweep lost seeds 1-40 this
# way when the 41-70 run overwrote the same filenames).
RANGE_TAG="$(echo "$SEEDS" | tr ',.' '__')"
for cfg in "${CONFIGS[@]}"; do
  label="${cfg%%:*}"; rest="${cfg#*:}"
  strat="${rest%%:*}"; rest="${rest#*:}"
  rounds="${rest%%:*}"; pairs="${rest#*:}"
  echo "=== $label (strategy=$strat rounds=$rounds pairs=$pairs) ==="
  PYTHONHASHSEED=0 \
  HEXTILES_WIDEN_STRATEGY="$strat" \
  HEXTILES_WIDEN_ROUNDS="$rounds" \
  HEXTILES_WIDEN_PAIRS="$pairs" \
    scripts/run_seeds.sh "$SEEDS" "$THREADS" | tee "$OUTDIR/$label.$RANGE_TAG.txt" | tail -6
done

echo
echo "=== per-config totals ($SEEDS) ==="
for cfg in "${CONFIGS[@]}"; do
  label="${cfg%%:*}"
  total=$(grep -o 'Score = [0-9.]*' "$OUTDIR/$label.$RANGE_TAG.txt" | awk '{s += $3} END {printf "%.0f", s}')
  echo "$label: $total"
done
