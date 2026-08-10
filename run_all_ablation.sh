#!/usr/bin/env bash
# Runs ablation.py against each of the paper's benchmark datasets, then summarizes all
# results/thesis/*.csv into one best-accuracy-per-dataset+algorithm table. A dataset with no
# corpus present locally (missing data/benchmark/<name>/connections.csv) just errors and the
# script continues on to the next one -- no -e, so one missing/failing dataset doesn't stop
# the rest from running.
#
# Usage:
#   ./run_all_ablation.sh [algorithm]
#
# algorithm defaults to LR. See evaluation_algorithms.get_hyperparameters for supported values
# (LR, RF, GBM, XT, XGB, KNN).

ALGORITHM="${1:-LR}"

uv run python src/feature_discovery/experiments/ablation.py --dataset credit --algorithm "$ALGORITHM"
uv run python src/feature_discovery/experiments/ablation.py --dataset steel --algorithm "$ALGORITHM"
uv run python src/feature_discovery/experiments/ablation.py --dataset jannis --algorithm "$ALGORITHM"
uv run python src/feature_discovery/experiments/ablation.py --dataset miniboone --algorithm "$ALGORITHM"
uv run python src/feature_discovery/experiments/ablation.py --dataset eyemove --algorithm "$ALGORITHM"
uv run python src/feature_discovery/experiments/ablation.py --dataset bioresponse --algorithm "$ALGORITHM"
uv run python src/feature_discovery/experiments/ablation.py --dataset school --algorithm "$ALGORITHM"
uv run python src/feature_discovery/experiments/ablation.py --dataset covertype --algorithm "$ALGORITHM"

uv run python summarize_results.py
