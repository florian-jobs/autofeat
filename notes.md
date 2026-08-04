1. Datasets used in AutoFeat:
    For a comprehensive evaluation, we choose six binary classification datasets varying in domains (e.g., medicine,
    web data, pattern recognition), the ratio of rows to columns,
    and types of features (e.g., discrete, continuous, nominal, and
    ordinal). These datasets are sourced from widely-employed
    ML repositories: OpenML, Kaggle, and UC Irvine1.

2. Compare performance of reproduced version with original versions numbers:
    Regarding AutoFeat, one thing that would be useful to try out is comparing the performance of the "reproduced"
    version against the numbers reported in their paper on some of the datasets they use, just to assess how close they
    are. Then, trying to build their own dataset according to their indications, as it might be required for a comparison.

This repo already has the infrastructure to do exactly this — you don't need to build it from scratch, just wire it up to real numbers. Here's the guide:

## Part 1: Reproduce the paper's numbers on their own datasets

The repo already has the thesis-reproduction pipeline, separate from `baseline.py`:

- `data/benchmark/datasets.csv` already lists the paper's benchmark datasets: `credit`, `steel`, `jannis`, `miniboone`, `eyemove`, `bioresponse`, `school`, `covertype` — you have `credit` locally, the rest are missing.
- `init_datasets.py` reads that CSV into `Dataset` objects.
- `ablation.py`'s `autofeat(...)` function runs the full pipeline (`AutoFeat.streaming_feature_selection` → `evaluate_paths`) and returns `Result` objects with `accuracy`, `train_time`, `feature_selection_time`, etc. — the same metrics reported in the paper's tables.
- `Result.TFD = "AutoFeat"` is the approach label used when comparing against `ARDA` and the `JOIN_ALL_*` baselines already defined in `result_object.py`.

Steps:
1. **Get the missing datasets + their lake corpora and precomputed `join_paths.csv`.** These come from the paper's own data release (the Zenodo link already referenced in the README history) — not from Valentine/`discover_join_paths.py`, since the paper reports numbers against a *specific, fixed* join graph, and matching that exactly is what makes the comparison meaningful. Using your own auto-discovered joins would compare a different pipeline, not reproduce their result.
2. **Drop each dataset into `data/benchmark/<name>/`** following the same layout as `credit` (base table + lake tables + a connections/join-paths file `ablation.py`'s `load_join_paths` can read — note it expects `fk_table`/`fk_column`/`pk_table`/`pk_column` columns, renamed internally).
3. **Run `ablation.py` per dataset** (it already loops via `filter_datasets`/`ALL_DATASETS`) with the same hyperparameters the paper used (`value_ratio`, `top_k`, `algorithm` — check the PDF's experimental-setup section, `docs/assets/papers/ICDE_FeatureDiscovery.pdf`, for the exact values they used per dataset).
4. **Collect `Result.accuracy`** per dataset (written under `RESULTS_FOLDER`) and put them side by side with the paper's reported table. Any difference is your reproduction gap — worth noting the *algorithm* (`LR`, etc.), `value_ratio`, and `top_k` used for each, since those directly affect the number.
![img.png](img.png)
![img_1.png](img_1.png)
## Part 2: Build your own dataset per the paper's construction method

Only needed for datasets the paper describes generically (e.g. "we built X from Socrata data using Y join keys") rather than shipping directly. To do this:

1. **Read the paper's dataset-construction section carefully** (`docs/assets/papers/ICDE_FeatureDiscovery.pdf`) — note the source (e.g. an open-data portal), the base table/target selection rule, and the join-key criteria they used to build the join graph.
2. **Assemble the base table + lake corpus** to match that description, and build a `join_paths.csv`/connections file **in the same schema `ablation.py` expects** (`fk_table`, `fk_column`, `pk_table`, `pk_column`) — you can lean on `discover_join_paths.py` here since, unlike Part 1, there's no "ground truth" join graph to match — your own is the point.
3. **Add a row to `data/benchmark/datasets.csv`** (or a separate CSV, same schema) and add a `Dataset` entry so it flows through `init_datasets()`/`ablation.py` like the built-in ones.
4. **Run the same `ablation.py` flow** and report accuracy/timing — now it's a genuinely new comparison point, not a reproduction, so present it as such (different join graph, possibly different value distributions than what the paper used).

Want me to start on either — e.g. write a small script that runs `ablation.py` over all datasets in `datasets.csv` and prints a results table, or sketch the `datasets.csv`/connections-CSV schema for a new self-built dataset?