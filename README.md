# MSG-LDM

Code release for the structure-guided latent diffusion framework for missing-modality MRI synthesis.

The repository is organized by the three stages described in the manuscript:

| Directory | Manuscript | Role |
|---|---|---|
| `stage1_vae/` | Section 3.2 | KL-regularized autoencoder. Encodes each modality independently into a `4 x 24 x 24` latent and decodes it back. |
| `stage1_msg/` | Section 3.3 | Multi-cue structural guidance (MCE, CMIF, MSDF). Fuses the observed modalities into the structural prior `z_c` (`128 x 24 x 24`). |
| `stage2_ldm/` | Section 3.4 | Structure-conditioned latent diffusion. Completes the missing latents while the observed latents are kept fixed. |

## Pipeline

```
raw 4-modality volume
        |
        |  per modality
        v
+----------------+     z_m (4 x 24 x 24)
|  stage1_vae    |--------------------------+
+----------------+                          |
                                            v
+----------------+     z_c (128 x 24 x 24)  |
|  stage1_msg    |--------------------------+
+----------------+                          |
                                            v
                              +-----------------------------+
                              |  stage2_ldm                 |
                              |  structure-conditioned DDPM |
                              |  observed latents fixed     |
                              +-----------------------------+
                                            |
                                            v
                                  synthesized modality
```

Stage I is trained first. Stage II loads the frozen Stage-I VAE and MSG weights from paths passed on the command line.

## Repository layout

```
.
├── stage1_vae/              # Section 3.2, KL-VAE
│   ├── main.py
│   ├── environment.yaml
│   ├── configs/brats20_vae_recon.yaml
│   └── taming/
├── stage1_msg/              # Section 3.3, MSG
│   ├── main.py
│   ├── model.py
│   ├── disentangle.py
│   ├── losses.py
│   ├── dataset.py
│   ├── utils.py
│   └── config_brats20.yaml
├── stage2_ldm/              # Section 3.4, latent diffusion
│   ├── main.py
│   ├── evaluate.py
│   ├── configs/latent-diffusion/brats_ldm_vae_msg.yaml
│   └── ldm/
├── scripts/                 # training and evaluation entry points
├── data/                    # raw data location (not tracked)
└── docs/
```

## Data layout

The dataset classes read one folder per split. Every case is a sub-directory named after the case ID, and each file follows `<case>_<modality>.nii` or `<case>_<modality>.nii.gz`:

```
data/BraTS20/
├── TrainingData/
│   ├── BraTS20_001/
│   │   ├── BraTS20_001_t1.nii[.gz]
│   │   ├── BraTS20_001_t2.nii[.gz]
│   │   ├── BraTS20_001_t1ce.nii[.gz]
│   │   ├── BraTS20_001_flair.nii[.gz]
│   │   └── BraTS20_001_seg.nii[.gz]
│   └── ...
└── TestData/
    └── ...
```

The same layout is used for BraTS2021; point the `data_path` entries in the configs to the corresponding folders.

## Training / validation / test protocol

`TrainingData` is split into train and validation subsets at the subject level by the dataset classes:

```yaml
val_ratio: 0.1
seed: 42
```

`CustomTrain` and `CustomVal` use the same ratio and seed, so the two subsets are complementary and reproducible. No test subject is used during training or checkpoint selection.

`TestData` is reserved for the final evaluation. `stage2_ldm/evaluate.py` reads it from `--test_data_path` and does not use the training config's `validation` entry.

## Weights

No weights are included in the repository. Stage-II training and evaluation take the frozen Stage-I weights as command-line paths:

```bash
--vae_ckpt /path/to/vae.ckpt
--msg_ckpt /path/to/msg.ckpt
```

The Stage-II model itself is loaded with `--resume` for training or `--ldm_ckpt` for evaluation.

## Training

```bash
bash scripts/train_stage1_vae.sh
bash scripts/train_stage1_msg.sh
bash scripts/train_stage2_ldm.sh /path/to/vae.ckpt /path/to/msg.ckpt
```

The scripts change into the corresponding stage directory, so the relative paths in the configs are resolved from there. Override the config paths directly or pass a local config with `-b`.

## Evaluation

```bash
bash scripts/evaluate_stage1_vae.sh /path/to/vae.ckpt
bash scripts/evaluate_stage2.sh /path/to/ldm.ckpt /path/to/vae.ckpt /path/to/msg.ckpt
```

The first script reports KL-VAE PSNR/SSIM/L1 on the test split. The second runs DDIM inpainting over the single-, double- and triple-missing patterns and writes `summary.json`, `summary.csv` and per-pattern figures to `eval_outputs/`. Both evaluate every subject in `TestData`; `--max_batches 0` means no batch cap. The Stage-II summary reports PSNR/SSIM/MAE as mean ± std over test subjects, after averaging the slices of each subject.

## Environment

The code was developed with Python 3.8 and PyTorch Lightning 1.x. See `requirements.txt` and `stage1_vae/environment.yaml`.

## Notes

- `stage1_vae/` is the paper's KL-VAE. The directory name `taming/` and some class names follow the original taming-transformers code base.
- `stage1_msg/` names follow the manuscript: `FrequencyCueExtractor` implements MCE, `HybridCIFModule` implements CMIF, and `MSFusion` implements MSDF.
- Weights, logs, cached assets, qualitative figures and large data files are intentionally excluded from the repository.
