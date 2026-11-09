# Stage 2: Structure-conditioned latent diffusion (Section 3.4)

This directory contains the latent diffusion model. The denoiser is conditioned on the Stage-I structural prior `z_c`, a target mask, and the available modality latents. During sampling, the observed latents are fixed and only the missing ones are generated.

## Files

| File | Purpose |
|---|---|
| `main.py` | Training entry point, data module and callbacks. |
| `evaluate.py` | DDIM inpainting over all missing-modality patterns; writes PSNR/SSIM/MAE summaries. |
| `ldm/models/diffusion/ddpm.py` | `LatentDiffusion`, `DiffusionWrapper`, training and sampling loops. |
| `ldm/models/diffusion/ddim.py` | `DDIMSampler` used by evaluation and inpainting. |
| `ldm/models/autoencoder.py` | Frozen Stage-I KL-VAE wrapper. |
| `ldm/models/msg.py` | Frozen Stage-I MSG wrapper. It imports `disentangle.py` from `../stage1_msg`. |
| `ldm/modules/diffusionmodules/openaimodel.py` | UNet backbone. |
| `ldm/taming/data/brats3.py` | BraTS dataset returning all four modalities. |
| `configs/latent-diffusion/brats_ldm_vae_msg.yaml` | Default BraTS2020 configuration. |

## Conditioning

The UNet input has `16 + 128 + 4 + 16` channels:

| Channels | Content |
|---:|---|
| 16 | noisy target latents (4 modalities x 4 latent channels) |
| 128 | structural prior `z_c` from Stage-I MSG |
| 4 | target mask (1 for missing modalities) |
| 16 | available-modality latents |

The available latents are pinned during DDIM sampling, so the acquired modalities are not regenerated.

## Training

Place the BraTS data under `../data/BraTS20/` and pass the two frozen Stage-I weights on the command line:

```bash
python main.py -t \
  --base configs/latent-diffusion/brats_ldm_vae_msg.yaml \
  --vae_ckpt /path/to/vae.ckpt \
  --msg_ckpt /path/to/msg.ckpt \
  -n brats20_ldm \
  --logdir ../logs/stage2_ldm
```

`TrainingData` is split into train and validation subsets at the subject level (`val_ratio: 0.1`, `seed: 42`). `TestData` is only used by `evaluate.py`.

The VAE and MSG weights are loaded from `--vae_ckpt` and `--msg_ckpt` and kept frozen. Only the denoising UNet is trained.

## Evaluation

```bash
python evaluate.py \
  --config configs/latent-diffusion/brats_ldm_vae_msg.yaml \
  --ldm_ckpt /path/to/ldm.ckpt \
  --vae_ckpt /path/to/vae.ckpt \
  --msg_ckpt /path/to/msg.ckpt \
  --test_data_path ../data/BraTS20/TestData \
  --ddim_steps 50 \
  --max_batches 0 \
  --out_dir ../eval_outputs/brats20
```

`--max_batches 0` evaluates every subject in `TestData`. Metrics are averaged over the slices of each subject first, then reported as mean ± std across test subjects.
