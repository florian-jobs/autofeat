#!/usr/bin/env bash
# Runs ablation.py against each of the paper's benchmark datasets, then summarizes all
# results/thesis/*.csv into one best-accuracy-per-dataset+algorithm table. A dataset with no
# corpus present locally (missing data/benchmark/<name>/connections.csv) just errors and the
# script continues on to the next one -- no -e, so one missing/failing dataset doesn't stop
# the rest from running.
#
# For each dataset, prefers data/benchmark/<name>/connections_discovered.csv -- the real
# Valentine/Coma-discovered join graph written by export_discovered_connections.py (run
# reingest_dataset.py --dataset <name> against a live Neo4j instance first, then
# export_discovered_connections.py --dataset <name>) -- over the synthetic ground-truth
# connections.csv tree. Falls back to connections.csv when no discovered file exists yet, so
# this script works before the discovery pipeline has been run for every dataset.
#
# Usage:
#   ./run_all_ablation.sh [algorithm] [top_k]
#
# algorithm defaults to LR. See evaluation_algorithms.get_hyperparameters for supported values
# (LR, RF, GBM, XT, XGB, KNN). top_k defaults to 15 (kappa in the paper).

ALGORITHM="${1:-LR}"
TOP_K="${2:-15}"

DATASETS=(credit steel jannis miniboone eyemove bioresponse school covertype)
# concrete_compressive_strength house_sales kin8nm

for dataset in "${DATASETS[@]}"; do
    discovered="data/benchmark/$dataset/connections_discovered.csv"
    if [ -f "$discovered" ]; then
        connections="$discovered"
    else
        connections="data/benchmark/$dataset/connections.csv"
    fi
    uv run python src/feature_discovery/experiments/ablation.py \
        --dataset "$dataset" --algorithm "$ALGORITHM" --top-k "$TOP_K" --connections "$connections"
done

uv run python summarize_results.py
