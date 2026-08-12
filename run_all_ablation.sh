#!/usr/bin/env bash
# Runs ablation.py against each of the paper's benchmark datasets, then summarizes all
# results/thesis/*.csv into one best-accuracy-per-dataset+algorithm table. A dataset with no
# corpus present locally (missing data/benchmark/<name>/connections.csv) just errors and the
# script continues on to the next one -- no -e, so one missing/failing dataset doesn't stop
# the rest from running.
#
# Usage:
#   ./run_all_ablation.sh [algorithm] [top_k]
#
# algorithm defaults to LR. See evaluation_algorithms.get_hyperparameters for supported values
# (LR, RF, GBM, XT, XGB, KNN). top_k defaults to 15 (kappa in the paper).

ALGORITHM="${1:-LR}"
TOP_K="${2:-15}"

uv run python src/feature_discovery/experiments/ablation.py --dataset credit --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset steel --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset jannis --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset miniboone --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset eyemove --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset bioresponse --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset school --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset concrete_compressive_strength --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset house_sales --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset kin8nm --algorithm "$ALGORITHM" --top-k "$TOP_K"

uv run python summarize_results.py
