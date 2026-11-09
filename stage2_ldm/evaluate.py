"""
Offline evaluation for the MSG-conditioned LDM (missing-modality synthesis).

For every missing-modality pattern it generates the full 4-modality set with
DDIM + inpainting (available modalities are pinned to their VAE latents) and
reports, per pattern / per modality / per group:

    - PSNR / SSIM / MAE on the *missing* modalities (vs GT images)
    - mean +/- std over test subjects, after averaging the slices of each subject

It also dumps result figures (GT | prediction | error) per pattern.

Example
-------
    python evaluate.py \
        --config configs/latent-diffusion/brats_ldm_vae_msg.yaml \
        --ldm_ckpt /path/to/ldm.ckpt \
        --vae_ckpt /path/to/vae.ckpt \
        --msg_ckpt /path/to/msg.ckpt \
        --test_data_path ../data/BraTS20/TestData \
        --ddim_steps 50 --max_batches 0 --out_dir eval_outputs
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from ldm.util import instantiate_from_config

MODALITIES = ["t1", "t2", "t1ce", "flair"]

SINGLE = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
DOUBLE = [[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]]
TRIPLE = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
PATTERN_GROUPS = {"single": SINGLE, "double": DOUBLE, "triple": TRIPLE}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/latent-diffusion/brats_ldm_vae_msg.yaml")
    p.add_argument("--ldm_ckpt", required=True)
    p.add_argument("--msg_ckpt", default=None)
    p.add_argument("--vae_ckpt", default=None)
    p.add_argument("--test_data_path", default="../data/BraTS20/TestData",
                   help="held-out test split; never used during training")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--max_batches", type=int, default=0, help="batches per pattern (0 = all test subjects)")
    p.add_argument("--groups", nargs="+", default=["single", "double", "triple"])
    p.add_argument("--out_dir", default="eval_outputs")
    p.add_argument("--save_figs", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def pattern_name(pattern):
    miss = [MODALITIES[i] for i in range(4) if pattern[i] == 0]
    return "miss_" + "+".join(miss)


def load_model(cfg, args):
    vae_ckpt = args.vae_ckpt or cfg.model.params.first_stage_config.params.get("ckpt_path")
    msg_ckpt = args.msg_ckpt or cfg.model.params.structural_features_config.params.get("ckpt_path")
    if not vae_ckpt or not msg_ckpt:
        raise ValueError(
            "evaluation requires frozen Stage-I weights: "
            "pass --vae_ckpt /path/to/vae.ckpt --msg_ckpt /path/to/msg.ckpt"
        )
    cfg.model.params.first_stage_config.params.ckpt_path = vae_ckpt
    cfg.model.params.structural_features_config.params.ckpt_path = msg_ckpt

    model = instantiate_from_config(cfg.model)
    sd = torch.load(args.ldm_ckpt, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] LDM ckpt: {len(missing)} missing / {len(unexpected)} unexpected keys")
    model = model.cuda().eval()
    return model


def build_loader(cfg, args):
    # Evaluation always uses the held-out test split. The training config's
    # validation entry points to a subject-level subset of TrainingData, so it
    # must not be reused here.
    if not os.path.isdir(args.test_data_path):
        raise FileNotFoundError(f"test split not found: {args.test_data_path}")
    test_cfg = OmegaConf.create({
        "target": "ldm.taming.data.brats3.CustomTest",
        "params": {"data_path": args.test_data_path},
    })
    dataset = instantiate_from_config(test_cfg)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, drop_last=False)


def psnr_ssim_mae_per_sample(pred, gt):
    """Per-slice PSNR / SSIM / MAE for a batch of shape (B, 1, H, W)."""
    pred = pred.clamp(-1., 1.).float()
    gt = gt.clamp(-1., 1.).float()
    mse = ((pred - gt) ** 2).flatten(1).mean(1)
    psnr = 10. * torch.log10(4.0 / (mse + 1e-12))
    a = ((pred + 1.) / 2.)
    b = ((gt + 1.) / 2.)
    s = ssim(a, b, data_range=1.0, size_average=False)
    mae = (pred - gt).abs().flatten(1).mean(1)
    return psnr.cpu().numpy(), s.cpu().numpy(), mae.cpu().numpy()


def save_grid(grid, path):
    arr = (grid[0].permute(1, 2, 0) * 255).clamp(0, 255).byte().cpu().numpy()
    Image.fromarray(arr).save(path)


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _subject_means(per_subject):
    """Average the slices of each subject, per modality and metric."""
    out = {}
    for subject, mods in per_subject.items():
        out[subject] = {}
        for mod, metrics in mods.items():
            out[subject][mod] = {
                k: (float(np.mean(v)) if v else None) for k, v in metrics.items()
            }
    return out


def _summarize(per_subject):
    """Mean +/- std across subjects for one pattern."""
    subj_means = _subject_means(per_subject)
    out = {}
    for mod in MODALITIES:
        if not any(mod in d for d in subj_means.values()):
            continue
        out[mod] = {}
        for metric in ("psnr", "ssim", "mae"):
            values = [d[mod][metric] for d in subj_means.values()
                      if mod in d and d[mod][metric] is not None]
            mean, std = _mean_std(values)
            out[mod][metric + "_mean"] = mean
            out[mod][metric + "_std"] = std
        out[mod]["n_subjects"] = len([d for d in subj_means.values() if mod in d])
        out[mod]["n_slices"] = sum(
            len(per_subject[s][mod]["psnr"]) for s in per_subject if mod in per_subject[s]
        )
    return out


def _fmt(d, key, digits):
    mean, std = d.get(key + "_mean"), d.get(key + "_std")
    if mean is None:
        return "NA"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


@torch.no_grad()
def run_pattern(model, loader, pattern, args):
    # per_subject[subject][modality][metric] = list of per-slice values
    per_subject = defaultdict(
        lambda: defaultdict(lambda: {"psnr": [], "ssim": [], "mae": []})
    )
    n = 0
    fig = None

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        B = batch["t1"].shape[0]
        names = batch["name"]
        lst = torch.tensor([pattern] * B, device=model.device, dtype=torch.float32)

        z_tgt, cond, lst_, x_src_list = model.get_input(
            batch, model.first_stage_key, return_first_stage_outputs=True, lst=lst)
        C = z_tgt.shape[1] // 4
        mod_mask = lst_.repeat_interleave(C, dim=1)[:, :, None, None]

        with model.ema_scope():
            samples, _ = model.sample_log(
                cond=cond, batch_size=B, ddim=True, ddim_steps=args.ddim_steps,
                eta=0.0, mask=mod_mask, x0=z_tgt)

        x_gen_list = [model.decode_first_stage(samples[:, i * C:(i + 1) * C]) for i in range(4)]

        for i, name in enumerate(MODALITIES):
            if pattern[i] == 1:
                continue
            p, s, m = psnr_ssim_mae_per_sample(x_gen_list[i], x_src_list[i])
            for j, subject in enumerate(names):
                per_subject[subject][name]["psnr"].append(float(p[j]))
                per_subject[subject][name]["ssim"].append(float(s[j]))
                per_subject[subject][name]["mae"].append(float(m[j]))

        if bi == 0 and args.save_figs:
            log = {f"samples/{name}": x_gen_list[i].to(model.device)
                   for i, name in enumerate(MODALITIES)}
            fig = model._build_result_grid(x_src_list, log, lst_, MODALITIES)

        n += B

    return per_subject, fig, n


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = OmegaConf.load(args.config)

    model = load_model(cfg, args)
    loader = build_loader(cfg, args)

    results = {}
    # modality_pool[modality][subject][metric] = list of per-pattern subject means
    modality_pool = defaultdict(
        lambda: defaultdict(lambda: {"psnr": [], "ssim": [], "mae": []})
    )

    for group in args.groups:
        for pattern in PATTERN_GROUPS[group]:
            key = pattern_name(pattern)
            print(f"\n[eval] pattern {key}  (group={group})")
            per_subject, fig, n = run_pattern(model, loader, pattern, args)
            mod_sum = _summarize(per_subject)
            subj_means = _subject_means(per_subject)

            results[key] = {
                "group": group,
                "available": pattern,
                "n_slices": n,
                "n_subjects": len(per_subject),
                "modalities": mod_sum,
            }
            for subject, mods in subj_means.items():
                for mod, vals in mods.items():
                    for metric in ("psnr", "ssim", "mae"):
                        if vals[metric] is not None:
                            modality_pool[mod][subject][metric].append(vals[metric])

            msg = "  ".join(
                f"{m}: PSNR={_fmt(mod_sum[m], 'psnr', 2)} "
                f"SSIM={_fmt(mod_sum[m], 'ssim', 4)} "
                f"MAE={_fmt(mod_sum[m], 'mae', 4)}"
                for m in mod_sum
            )
            print("  " + msg)
            if fig is not None:
                save_grid(fig, os.path.join(args.out_dir, f"result_{key}.png"))

    # Aggregate over patterns: average each subject's per-pattern means, then
    # compute mean +/- std across subjects.
    mod_agg = {}
    for mod, subj_map in modality_pool.items():
        agg = {}
        for metric in ("psnr", "ssim", "mae"):
            values = []
            for subject in subj_map:
                vals = subj_map[subject][metric]
                if vals:
                    values.append(float(np.mean(vals)))
            mean, std = _mean_std(values)
            agg[metric + "_mean"] = mean
            agg[metric + "_std"] = std
        agg["n_subjects"] = len(subj_map)
        mod_agg[mod] = agg

    summary = {"per_pattern": results, "per_modality": mod_agg}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "group", "modality", "n_subjects", "n_slices",
                    "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
                    "mae_mean", "mae_std"])
        for key, r in results.items():
            for mod, d in r["modalities"].items():
                w.writerow([
                    key, r["group"], mod, d["n_subjects"], d["n_slices"],
                    f"{d['psnr_mean']:.4f}", f"{d['psnr_std']:.4f}",
                    f"{d['ssim_mean']:.4f}", f"{d['ssim_std']:.4f}",
                    f"{d['mae_mean']:.4f}", f"{d['mae_std']:.4f}",
                ])

    print("\n==== per-modality aggregate (mean +/- std over subjects) ====")
    for mod, d in mod_agg.items():
        print(f"  {mod}: PSNR={_fmt(d, 'psnr', 2)} "
              f"SSIM={_fmt(d, 'ssim', 4)} MAE={_fmt(d, 'mae', 4)}")
    print(f"\nSaved summary.json / summary.csv / result_*.png to {args.out_dir}")


if __name__ == "__main__":
    main()
