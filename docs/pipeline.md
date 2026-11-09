# Pipeline interfaces

This note records the tensor interfaces between the three stages. It follows Sections 3.2–3.4 of the manuscript.

## Stage 1a: KL-VAE (Section 3.2)

Input: one modality slice, `1 x 192 x 192`, normalized to `[-1, 1]`.

Output: posterior over a `4 x 24 x 24` latent. The code uses the posterior mode at evaluation time.

The same frozen encoder is applied to each of the four modalities. The four latents are concatenated into `16 x 24 x 24` for Stage II.

## Stage 1b: MSG (Section 3.3)

Input: the four modality slices and the availability vector `lst` of length 4.

Output:

- `fused_multiscale`: `128 x 24 x 24`. This is the structural prior `z_c` consumed by Stage II.
- four reconstructed modalities and a four-class segmentation map (training supervision only).

The encoder path uses `FrequencyCueExtractor` (MCE), `HybridCIFModule` (CMIF) and `MSFusion` (MSDF). Missing modalities are masked in the content path; the style path is defined over the available inputs.

## Stage 2: latent diffusion (Section 3.4)

Input to the UNet at each denoising step:

```
concat[
  noisy target latents,   16 x 24 x 24
  structural prior z_c,  128 x 24 x 24
  target mask,             4 x 24 x 24
  available latents,      16 x 24 x 24
]
```

Total: `164` channels.

During DDIM sampling the available latents are replaced by their clean values at every step, so the observed modalities are not regenerated. Only the target positions are denoised.

## Data flow

1. Encode all four modalities with the frozen VAE.
2. Run the frozen MSG on the observed modalities to obtain `z_c`.
3. Build the target mask from the availability vector.
4. Run DDIM inpainting on the target latents.
5. Decode the completed latents with the frozen VAE decoder.

Steps 1–2 and 5 are shared by training and evaluation; only step 4 differs.

## Split protocol

Each stage reads `TrainingData` and splits it into train and validation subjects with `val_ratio: 0.1` and `seed: 42`. `CustomTrain` and `CustomVal` use the same split, so no validation subject is seen during optimization.

`TestData` is held out from all training and validation code. `stage2_ldm/evaluate.py` reads it from `--test_data_path` and never uses the training config's `validation` entry.
