# Stage 1: MSG (Section 3.3)

This directory contains the multi-cue structural guidance network. It takes the observed modalities, fuses them through MCE, CMIF and MSDF, and produces the structural prior `z_c`.

`z_c` is the `fused_multiscale` tensor of shape `128 x 24 x 24`. It is consumed by Stage II as the structure condition.

## Files

| File | Purpose |
|---|---|
| `main.py` | Training entry point, data module and checkpointing. |
| `model.py` | `VQModel` LightningModule: random modality masking, reconstruction, segmentation and KL losses. |
| `disentangle.py` | `DisentangleModel`, `FrequencyCueExtractor` (MCE), `HybridCIFModule` (CMIF), `MSFusion` (MSDF), `ImageDecoder` and `MaskDecoder`. |
| `losses.py` | `GradLoss`, `BCEDiceLoss` and `kl_loss`. |
| `dataset.py` | BraTS dataset with WT/TC/ET labels and slice sampling. |
| `utils.py` | Config instantiation and collate helpers. |
| `config_brats20.yaml` | Default BraTS2020 training configuration. |

## Paper components

| Manuscript component | Code |
|---|---|
| Multi-cue embedding (MCE) | `FrequencyCueExtractor` in `disentangle.py` |
| Cross-modal interaction fusion (CMIF) | `HybridCIFModule` in `disentangle.py` |
| Multi-scale dynamic fusion (MSDF) | `MSFusion` in `disentangle.py` |
| Structural prior `z_c` | `fused_multiscale` returned by `DisentangleModel.forward` |
| Reconstruction / segmentation supervision | `model.py` and `losses.py` |

## Training

```bash
python main.py -t \
  --base config_brats20.yaml \
  -n brats20_msg \
  --logdir ../logs/stage1_msg
```

The config reads `../data/BraTS20/TrainingData` and splits it into train and validation subsets at the subject level (`val_ratio: 0.1`, `seed: 42`). Missing modalities are sampled during training; the observed ones are kept. `TestData` is not used in this stage.

The style encoder intentionally sees the full four-modality input during Stage-I training. It is used only for the auxiliary reconstruction branch; Stage II consumes `z_c` (`fused_multiscale`), which is computed from the masked content path.

## Output

The trained MSG checkpoint is passed to Stage II with `--msg_ckpt`.
