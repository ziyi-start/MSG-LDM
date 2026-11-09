#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/train_stage2_ldm.sh /path/to/vae.ckpt /path/to/msg.ckpt"
  exit 1
fi

VAE_CKPT="$1"
MSG_CKPT="$2"

cd "$(dirname "$0")/../stage2_ldm"

python main.py -t \
  --base configs/latent-diffusion/brats_ldm_vae_msg.yaml \
  --vae_ckpt "$VAE_CKPT" \
  --msg_ckpt "$MSG_CKPT" \
  -n brats20_ldm \
  --logdir ../logs/stage2_ldm
