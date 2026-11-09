#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash scripts/evaluate_stage2.sh /path/to/ldm.ckpt /path/to/vae.ckpt /path/to/msg.ckpt [test_data_path]"
  exit 1
fi

LDM_CKPT="$1"
VAE_CKPT="$2"
MSG_CKPT="$3"
TEST_DATA="${4:-../data/BraTS20/TestData}"

cd "$(dirname "$0")/../stage2_ldm"

python evaluate.py \
  --config configs/latent-diffusion/brats_ldm_vae_msg.yaml \
  --ldm_ckpt "$LDM_CKPT" \
  --vae_ckpt "$VAE_CKPT" \
  --msg_ckpt "$MSG_CKPT" \
  --test_data_path "$TEST_DATA" \
  --ddim_steps 50 \
  --max_batches 0 \
  --save_figs \
  --out_dir ../eval_outputs/brats20
