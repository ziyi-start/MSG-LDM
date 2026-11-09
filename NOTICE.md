# Third-party code

Parts of this repository are adapted from the following MIT-licensed projects:

- **taming-transformers** — the KL-VAE backbone, encoder/decoder blocks and LPIPS loss in `stage1_vae/`.
  https://github.com/CompVis/taming-transformers
- **latent-diffusion** — the latent diffusion backbone, UNet, DDPM/DDIM samplers and utility modules in `stage2_ldm/`.
  https://github.com/CompVis/latent-diffusion

The Stage-I MSG network in `stage1_msg/` and the structure-conditioning changes in `stage2_ldm/ldm/models/msg.py` are specific to this work.
