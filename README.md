This repository contains code for a MICCAI 2026 submission and is provided for double-blind review purposes only.


## Training

The proposed framework is trained in two stages.

In Stage 1 (Structural Feature Learning Training), the model learns disentangled anatomical (structure) representations, followed by Stage 2 (Diffusion Model Training), which trains the latent diffusion model based on the learned structural representations:


# Stage 1: Structural Feature Learning Training
cd VQGAN
python main.py -b configs/imagenet_vqgan_msg.yaml -t True --gpus 3,4,5 --logdir /logs

# Stage 2: Diffusion Model Training
cd LDM
python main.py -b configs/latent-diffusion/brats-ldm-vq-4.yaml -t True --gpus 2 --logdir /logs --scale_lr False