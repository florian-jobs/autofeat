#!/usr/bin/env bash
# Runs ablation.py against every dataset under data/benchmark/ that has a connections.csv
# (i.e. an actual benchmark corpus, not just an entry in datasets.csv), then summarizes all
# results/thesis/*.csv into one best-accuracy-per-dataset+algorithm table.
#
# Usage:
#   ./run_all_ablation.sh [algorithm]
#
# algorithm defaults to LR. See evaluation_algorithms.get_hyperparameters for supported values
# (LR, RF, GBM, XT, XGB, KNN).
set -uo pipefail

ALGORITHM="${1:-LR}"
failed=()

for dir in data/benchmark/*/; do
    dataset=$(basename "$dir")
    if [ -f "${dir}connections.csv" ]; then
        echo "=== $dataset (--algorithm $ALGORITHM) ==="
        if ! uv run python src/feature_discovery/experiments/ablation.py --dataset "$dataset" --algorithm "$ALGORITHM"; then
            echo "!!! $dataset failed, continuing with the rest" >&2
            failed+=("$dataset")
        fi
    fi
done

uv run python summarize_results.py

if [ "${#failed[@]}" -gt 0 ]; then
    echo "Failed datasets: ${failed[*]}" >&2
    exit 1
fi
