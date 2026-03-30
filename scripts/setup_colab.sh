#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/content/bioformer}"
DATA_URL="${2:-https://zenodo.org/records/14619074/files/EFP_long.csv?download=1}"

cd "$ROOT_DIR"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
mkdir -p data/raw/efp
curl -fL --retry 3 "$DATA_URL" -o data/raw/efp/EFP_long.csv

echo "Colab setup complete in $ROOT_DIR"

