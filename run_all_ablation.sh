#!/usr/bin/env bash
# Runs ablation.py on all benchmark datasets and then summarize_results.py.
# No -e so a missing dataset doesn't stop the other runs.
#
# Usage:
#   ./run_all_ablation.sh [algorithm] [top_k]
#
# algorithm: LR (default), RF, GBM, XT, XGB, KNN. top_k: default 15 (kappa in the paper).

ALGORITHM="${1:-LR}"
TOP_K="${2:-15}"

uv run python src/feature_discovery/experiments/ablation.py --dataset credit --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset steel --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset jannis --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset miniboone --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset eyemove --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset bioresponse --algorithm "$ALGORITHM" --top-k "$TOP_K"
uv run python src/feature_discovery/experiments/ablation.py --dataset school --algorithm "$ALGORITHM" --top-k "$TOP_K"
#uv run python src/feature_discovery/experiments/ablation.py --dataset concrete_compressive_strength --algorithm "$ALGORITHM" --top-k "$TOP_K"
#uv run python src/feature_discovery/experiments/ablation.py --dataset house_sales --algorithm "$ALGORITHM" --top-k "$TOP_K"
#uv run python src/feature_discovery/experiments/ablation.py --dataset kin8nm --algorithm "$ALGORITHM" --top-k "$TOP_K"

uv run python summarize_results.py
