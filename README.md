# Bioformer

Initial scaffold for the bioreactor transformer POC described in
`BIOREACTOR_TRANSFORMER_POC.md`.

## What is included

- EFP dataset ingestion and batch-safe splitting helpers
- Canonical sequence building with padding and missing-value masks
- Summary-feature baselines for ridge, elastic net, and XGBoost
- Small encoder-only transformer for early-to-final potency prediction
- Regression metrics and holdout plotting utilities
- Training entrypoints for baseline and transformer runs
- Configs, notebooks, and placeholder data directories

## Quickstart

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the project:

```bash
pip install -e ".[dev]"
```

3. Place the erythromycin dataset CSV at:

```text
data/raw/efp/EFP_long.csv
```

4. Adjust the config if the source column names differ from the defaults.

5. Run the baselines:

```bash
bioformer-train-baseline --config configs/baseline.yaml
```

6. Run the transformer:

```bash
bioformer-train-transformer --config configs/transformer_small.yaml
```

## Google Colab

If you want to run on Colab with a GPU runtime:

1. Switch Colab to a GPU runtime.
2. Clone this repo into `/content/bioformer`.
3. Run:

```bash
bash scripts/setup_colab.sh
bioformer-train-baseline --config configs/baseline.yaml
bioformer-train-transformer --config configs/transformer_small.yaml
```

There is also a starter notebook at `notebooks/00_colab_setup.ipynb`.

## Expected raw columns

The EFP config is aligned to the downloaded Zenodo CSV and assumes:

- `batch_id`
- `hh`
- `cer`

The training scripts rebase `hh` to per-batch `elapsed_hours` before horizon
filtering so `48` means "first 48 hours of a batch" rather than an absolute
process clock value. All other numeric columns are treated as process features
unless you list them explicitly in the config.
