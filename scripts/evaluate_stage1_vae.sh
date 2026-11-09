#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/evaluate_stage1_vae.sh /path/to/vae.ckpt [test_data_path]"
  exit 1
fi

VAE_CKPT="$1"
TEST_DATA="${2:-../data/BraTS20/TestData}"

cd "$(dirname "$0")/../stage1_vae"

python evaluate.py \
  --config configs/brats20_vae_recon.yaml \
  --ckpt "$VAE_CKPT" \
  --test_data_path "$TEST_DATA" \
  --out_dir ../eval_outputs/vae
