import torch
import torch.nn.functional as F
import pytorch_lightning as pl
try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
except ImportError:
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
    except ImportError:
        from torchmetrics.functional import ssim as ssim_fn

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.distributions import DiagonalGaussianDistribution


class VAEKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"], "VAEKL requires ddconfig.double_z = True"
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.contiguous().float()
        if x.shape[1] != 1 and x.shape[1] > x.shape[2]:
            x = x.permute(0, 3, 1, 2)
        elif x.shape[1] != 1:
            x = x.permute(0, 2, 1, 3)
        return x.contiguous().float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, posterior = self(x)

        aeloss, log_dict_ae = self.loss(x, xrec, posterior, self.global_step, split="train")

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, posterior = self(x)

        aeloss, log_dict_ae = self.loss(x, xrec, posterior, self.global_step, split="val")

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/kl", log_dict_ae["val/kl_loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        mse = torch.mean((x - xrec) ** 2, dim=[1, 2, 3])
        psnr_val = 20 * torch.log10(2.0 / torch.sqrt(mse + 1e-8)).mean()
        ssim_val = ssim_fn(xrec, x, data_range=2.0)
        self.log("val/psnr", psnr_val, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/ssim", ssim_val, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return log_dict_ae

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, posterior = self(x)
        log["inputs"] = x[:, :1]
        log["reconstructions"] = xrec[:, :1]
        return log
