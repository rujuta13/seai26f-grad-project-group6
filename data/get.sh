#!/usr/bin/env bash
# Download the three MOOT data sets, unmodified, into data/.
set -euo pipefail
cd "$(dirname "$0")"
base=https://raw.githubusercontent.com/timm/moot/master/optimize
for f in config/SS-W.csv behavior_data/all_players.csv sales_data/Marketing_Analytics.csv; do
  out=$(basename "$f")
  [ -s "$out" ] || curl -fsSL "$base/$f" -o "$out"
  echo "$out: $(($(wc -l < "$out") - 1)) rows"
done
