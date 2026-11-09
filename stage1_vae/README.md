# Stage 1: KL-VAE (Section 3.2)

This directory contains the KL-regularized autoencoder used in the first stage. It compresses one modality slice from `192 x 192` to a `4 x 24 x 24` latent and reconstructs it.

The implementation is derived from the taming-transformers code base. The model class is `VAEKL`; the legacy directory name `VQGAN` is not used in this release.

## Files

| File | Purpose |
|---|---|
| `main.py` | Training entry point, data module and image logger. |
| `taming/models/vae_kl.py` | `VAEKL` LightningModule (encoder, posterior, decoder, PSNR/SSIM validation). |
| `taming/modules/diffusionmodules/model.py` | Encoder and decoder with the resolution `192 -> 24`. |
| `taming/modules/losses/vae_perceptual.py` | L1 + LPIPS + KL loss. |
| `taming/modules/losses/lpips.py` | Single-channel LPIPS with VGG16. |
| `taming/data/brats_recon.py` | BraTS dataset, slice sampling and intensity normalization. |
| `configs/brats20_vae_recon.yaml` | Default BraTS2020 training configuration. |

## Training

```bash
python main.py -t \
  --base configs/brats20_vae_recon.yaml \
  -n brats20_vae \
  --logdir ../logs/stage1_vae
```

The config reads `../data/BraTS20/TrainingData` and splits it into train and validation subsets at the subject level (`val_ratio: 0.1`, `seed: 42`). Weights are written under the log directory. Set `data_path` and `gpus` in the config for your machine. `TestData` is not used in this stage.

The LPIPS weights are not stored in this repository; `taming.util.get_ckpt_path` downloads them on first run.

## Evaluation

```bash
python evaluate.py \
  --config configs/brats20_vae_recon.yaml \
  --ckpt /path/to/vae.ckpt \
  --test_data_path ../data/BraTS20/TestData
```

## Output

The trained VAE checkpoint is passed to Stage II with `--vae_ckpt`.
