#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../stage1_vae"

python main.py -t \
  --base configs/brats20_vae_recon.yaml \
  -n brats20_vae \
  --logdir ../logs/stage1_vae
