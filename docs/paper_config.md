# Paper-to-code parameter map

This table checks the values stated in the manuscript against the released configs and code. All model and optimization values match the paper.

## Stage 1a: KL-VAE (Section 3.2)

| Paper | Value | Code |
|---|---|---|
| Base channels | 128 | `ddconfig.ch: 128` |
| Channel multipliers | `[1, 1, 2, 4]` | `ddconfig.ch_mult` |
| Residual blocks per level | 2 | `ddconfig.num_res_blocks: 2` |
| Self-attention | middle block + lowest resolution | `attn_resolutions: [24]` and the mid block |
| Latent | `4 x 24 x 24` | `embed_dim: 4`, `z_channels: 4`, `resolution: 192` |
| Learning rate | `4.5 x 10^-6` | `base_learning_rate: 4.5e-06` |
| LPIPS weight | 1.0 | `lossconfig.perceptual_weight: 1.0` |
| KL weight | `10^-6` | `lossconfig.kl_weight: 1.0e-6` |
| Log-variance init | 0.0 | `lossconfig.logvar_init: 0.0` |
| Batch size | 12 | `data.params.batch_size: 12` |

## Stage 1b: MSG (Section 3.3)

| Paper | Value | Code |
|---|---|---|
| Base channels | 16 / 32 / 64 / 128 | `n_base_filters=16` in `disentangle.py` |
| Style code | 8-D | `StyleEncoder` output and `mlp(8, ...)` |
| Prior `z_c` | 128 x 24 x 24 | `MSFusion` / `fused_multiscale` |
| Learning rate | `10^-4` | `base_learning_rate: 1e-4` |
| Seg weight | 1.0 | `model.py`: `1.0 * seg_loss` |
| KL weight | 0.1 | `model.py`: `0.1 * KL_loss` |
| L1 and gradient weights | 1.0 each | `model.py`: `1.0 * rec_loss_total` |
| Batch size | 12 | `data.params.batch_size: 12` |
| Gaussian noise | `10^-5` | `dataset.py`: `eps = 1e-5` |

## Stage 2: LDM (Section 3.4)

| Paper | Value | Code |
|---|---|---|
| Learning rate | `10^-4` | `base_learning_rate: 1.0e-04` |
| AdamW betas | `(0.9, 0.999)` | `ddpm.py: configure_optimizers` |
| Weight decay | `10^-6` | `ddpm.py: weight_decay=1e-6` |
| Gradient clipping | 1.0 | `gradient_clip_val: 1.0` |
| EMA decay | 0.9999 with warmup | `ldm/modules/ema.py` default |
| Beta schedule | linear, `T = 1000` | `timesteps: 1000` and default linear schedule |
| UNet base width | 160 | `model_channels: 160` |
| UNet multipliers | `[1, 2, 4]` | `channel_mult: [1, 2, 4]` |
| Condition width | 148 = 128 prior + 4 mask + 16 available | `ddpm.py: get_input` |
| UNet input | 164 = 16 noisy + 148 condition | `in_channels: 164` |
| DDIM | `eta = 0`, 50 steps | `evaluate.py` |
| Batch size | 12 | `data.params.batch_size: 12` |
| LR scheduler | none (constant LR) | `use_scheduler: false` |

## Data split caveat

The manuscript states the train/test splits but does not specify a validation split:

- BraTS2020: 296 training / 73 test
- BraTS2021: 991 training / 260 test
- WMH: 60 training / 110 test

The code keeps those pools and carves a subject-level validation subset from `TrainingData` (`val_ratio: 0.1`, `seed: 42`) so that no test subject is used for checkpoint selection. This is the only protocol setting not stated in the paper. With 296 BraTS2020 subjects, the split is 266 optimization / 30 validation; all 73 test subjects remain held out.

Evaluation uses every subject in `TestData`. `stage2_ldm/evaluate.py` defaults to `--max_batches 0`, which means no batch cap. Per-slice metrics are averaged within each subject first, and PSNR/SSIM/MAE are then reported as mean ± std across test subjects.
