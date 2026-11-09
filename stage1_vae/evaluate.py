"""Evaluate the KL-VAE on the held-out test split.

Reports PSNR, SSIM and L1 per modality.

Example
-------
    python evaluate.py \
        --config configs/brats20_vae_recon.yaml \
        --ckpt /path/to/vae.ckpt \
        --test_data_path ../data/BraTS20/TestData
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from main import instantiate_from_config
from taming.data.brats_recon import CustomTest

MODALITIES = ["t1", "t2", "t1ce", "flair"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/brats20_vae_recon.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_data_path", default="../data/BraTS20/TestData")
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default="../eval_outputs/vae")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if not os.path.isdir(args.test_data_path):
        raise FileNotFoundError(f"test split not found: {args.test_data_path}")

    cfg = OmegaConf.load(args.config)
    cfg.model.params.ckpt_path = args.ckpt
    model = instantiate_from_config(cfg.model)
    model = model.cuda().eval() if torch.cuda.is_available() else model.eval()

    dataset = CustomTest(data_path=args.test_data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)

    stats = {m: {"psnr": [], "ssim": [], "l1": []} for m in MODALITIES}

    for batch in loader:
        x = batch["image"].cuda() if torch.cuda.is_available() else batch["image"]
        mod = batch["modality"]
        xrec, _ = model(x)

        mse = F.mse_loss(xrec, x, reduction="none").mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(4.0 / (mse + 1e-12))
        ssim_val = ssim(xrec, x, data_range=2.0, size_average=False)
        l1 = (xrec - x).abs().mean(dim=(1, 2, 3))

        for i in range(x.shape[0]):
            name = MODALITIES[int(mod[i])]
            stats[name]["psnr"].append(float(psnr[i]))
            stats[name]["ssim"].append(float(ssim_val[i]))
            stats[name]["l1"].append(float(l1[i]))

    summary = {}
    for name in MODALITIES:
        summary[name] = {k: float(np.mean(v)) for k, v in stats[name].items()}
        print(f"{name}: PSNR={summary[name]['psnr']:.2f} "
              f"SSIM={summary[name]['ssim']:.4f} L1={summary[name]['l1']:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
