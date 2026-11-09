from setuptools import setup, find_packages

setup(
    name='ldm',
    version='0.0.1',
    description='Latent Diffusion Models',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'pytorch_lightning',
        'omegaconf',
        'einops',
        'monai',
        'nibabel',
    ],
)
