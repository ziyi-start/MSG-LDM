#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../stage1_msg"

python main.py -t \
  --base config_brats20.yaml \
  -n brats20_msg \
  --logdir ../logs/stage1_msg
