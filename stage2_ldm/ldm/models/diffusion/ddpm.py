"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from functools import partial
from contextlib import contextmanager
from einops import rearrange, repeat
from tqdm import tqdm
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import exists, default, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from pytorch_msssim import ssim


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def on_before_optimizer_step(self, optimizer, optimizer_idx=None):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log('grad_norm', total_norm, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        mask_avg = loss_dict_no_ema.get('val/mask_avg', 0)
        self.log('val/mask_avg_step', mask_avg, prog_bar=True, logger=True, on_step=True, on_epoch=False)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 structural_features_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        self.scheduler_t_max = kwargs.pop("scheduler_t_max", 100)
        self.warmup_epochs = kwargs.pop("warmup_epochs", 5)
        self.val_gen_batches = kwargs.pop("val_gen_batches", 1)
        self.val_gen_steps = kwargs.pop("val_gen_steps", 50)
        self.val_gen_bs = kwargs.pop("val_gen_bs", 2)
        use_scheduler = kwargs.pop("use_scheduler", None)
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        if use_scheduler is not None:
            self.use_scheduler = use_scheduler
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_structural_features(structural_features_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True


    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            print("### USING STD-RESCALING ###")
            x = batch["t1"].to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
    def instantiate_structural_features(self, config):
        # Mask-only ablation: no learned structural prior. The denoiser is then
        # conditioned on the missing mask and the observed latents only.
        if config == "__is_disabled__":
            print("Structural prior disabled (mask-only ablation).")
            self.structural_features_model = None
            self._random_prior = False
            return
        # Capacity-matched control: a random prior with the same 128-channel shape.
        if config == "__random__":
            print("Using a random (capacity-matched) structural prior.")
            self.structural_features_model = None
            self._random_prior = True
            return
        self._random_prior = False
        model = instantiate_from_config(config)
        self.structural_features_model = model.eval()
        self.structural_features_model.train = disabled_train
        for param in self.structural_features_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_first_stage_encoding(self, encoder_posterior):
        encoder_posterior = encoder_posterior
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

 
    def get_input(self, batch, k, return_first_stage_outputs=False,
                force_c_encode=False, cond_key=None,
                return_original_cond=False, bs=None, lst=None):

        modalities = ["t1", "t2", "t1ce", "flair"]
        z_list = []
        x_m_list = []

        # ------------------------------
        # Encode each modality separately
        # ------------------------------
        for m in modalities:
            x_m = batch[m]  # (B,1,H,W)
            if bs is not None:
                x_m = x_m[:bs]
            x_m = x_m.to(self.device)
            x_m_list.append(x_m)

            z_m = self.encode_first_stage(x_m)
            z_m = self.get_first_stage_encoding(z_m)  # (B,C,h,w)
            z_list.append(z_m)

        # ------------------------------
        # Target latent (full)
        # ------------------------------
        z_tgt = torch.cat(z_list, dim=1).detach()  # (B,4C,h,w)
        x_m_lst = torch.cat(x_m_list, dim=1)

        B, total_C, h, w = z_tgt.shape
        modal_num = 4
        C = total_C // modal_num

        # ------------------------------
        # Build modality mask lst
        # ------------------------------
        if lst is None:
            lst = []
            for _ in range(B):
                num_ones = random.choice([1, 2, 3, 4])
                ones_indices = random.sample(range(modal_num), num_ones)
                mask = [1 if i in ones_indices else 0 for i in range(modal_num)]
                lst.append(mask)
            lst = torch.tensor(lst, device=z_tgt.device, dtype=torch.float32)
        else:
            lst = lst[:B].to(device=z_tgt.device, dtype=torch.float32)

        struct_fea = None
        if self.structural_features_model is not None:
            struct_fea = self.structural_features_model.model(x_m_lst, lst)
            fused_struct = struct_fea["fused_multiscale"]  # (B,128,h,w)
        elif getattr(self, "_random_prior", False):
            fused_struct = torch.randn(B, 128, h, w, device=z_tgt.device, dtype=z_tgt.dtype)
        else:
            fused_struct = None

        # target mask: missing modality indicator (B,4,h,w)
        target_mask = 1.0 - lst  # (B,4)
        target_mask = target_mask.unsqueeze(-1).unsqueeze(-1)  # (B,4,1,1)
        target_mask = target_mask.repeat(1, 1, h, w)           # (B,4,h,w)

        # available-modality latents (missing channels zeroed) (B,16,h,w)
        # The condition never contains missing-modality latents; the full z_tgt
        # is used only as the diffusion target during training/sampling.
        mask_expand = lst.repeat_interleave(C, dim=1)[:, :, None, None]  # (B,16,1,1)
        z_avail = z_tgt * mask_expand                                    # (B,16,h,w)

        # concatenate structural features + modality mask + available latents into c_concat
        if fused_struct is None:
            cond_image = torch.cat([target_mask, z_avail], dim=1)            # (B,20,h,w)
        else:
            cond_image = torch.cat([fused_struct, target_mask, z_avail], dim=1)  # (B,148,h,w)

        # ------------------------------
        # Final conditioning dict
        # ------------------------------
        cond = {
            "c_concat": cond_image,
            "c_crossattn": target_mask   # unused when use_spatial_transformer=False
        }

        out = [z_tgt, cond, lst]

        if return_first_stage_outputs:
            out.append(x_m_list)

        return out


    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def shared_step(self, batch, lst=None, **kwargs):
        x, c, lst_ = self.get_input(batch, self.first_stage_key, lst=lst)
        loss, loss_dict = self(x, c)
        prefix = 'train' if self.training else 'val'
        loss_dict[f'{prefix}/mask_avg'] = lst_.mean()  # average available modalities ratio
        return loss, loss_dict

    VAL_MODALITY_PATTERNS = [
        [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0],
        [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0],
        [1, 0, 0, 0], [0, 0, 0, 1],
    ]

    def _build_val_lst(self, batch_size, device):
        patterns = self.VAL_MODALITY_PATTERNS
        rows = [patterns[i % len(patterns)] for i in range(batch_size)]
        return torch.tensor(rows, device=device, dtype=torch.float32)

    @torch.no_grad()
    def _psnr_ssim(self, pred, gt):
        pred = ((pred.clamp(-1., 1.) + 1.) / 2.).float()
        gt = ((gt.clamp(-1., 1.) + 1.) / 2.).float()
        mse = F.mse_loss(pred, gt)
        psnr = (10. * torch.log10(1. / (mse + 1e-8))).item()
        ssim_val = ssim(pred, gt, data_range=1.0, size_average=True).item()
        return psnr, ssim_val

    @torch.no_grad()
    def _val_generation_metrics(self, batch, val_lst):
        z_tgt, cond, lst_, x_src_list = self.get_input(
            batch, self.first_stage_key,
            return_first_stage_outputs=True, bs=self.val_gen_bs, lst=val_lst)
        C = z_tgt.shape[1] // 4
        mod_mask = lst_.repeat_interleave(C, dim=1)[:, :, None, None]
        samples, _ = self.sample_log(
            cond=cond, batch_size=z_tgt.shape[0], ddim=True,
            ddim_steps=self.val_gen_steps, eta=0.0, mask=mod_mask, x0=z_tgt)
        miss_psnr, miss_ssim, avail_psnr, avail_ssim = [], [], [], []
        for i in range(4):
            x_gen = self.decode_first_stage(samples[:, i * C:(i + 1) * C])
            gt = x_src_list[i]
            miss = lst_[:, i] < 0.5
            if bool(miss.any()):
                p, s = self._psnr_ssim(x_gen[miss], gt[miss])
                miss_psnr.append(p)
                miss_ssim.append(s)
            avail = ~miss
            if bool(avail.any()):
                p, s = self._psnr_ssim(x_gen[avail], gt[avail])
                avail_psnr.append(p)
                avail_ssim.append(s)
        out = {}
        if miss_psnr:
            out['val/psnr_missing'] = float(np.mean(miss_psnr))
            out['val/ssim_missing'] = float(np.mean(miss_ssim))
        if avail_psnr:
            out['val/psnr_available'] = float(np.mean(avail_psnr))
            out['val/ssim_available'] = float(np.mean(avail_ssim))
        return out

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_size = batch["t1"].shape[0]
        val_lst = self._build_val_lst(batch_size, self.device)
        _, loss_dict_no_ema = self.shared_step(batch, lst=val_lst)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, lst=val_lst)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
            if batch_idx < self.val_gen_batches:
                loss_dict_ema.update(self._val_generation_metrics(batch, val_lst))
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        mask_avg = loss_dict_no_ema.get('val/mask_avg', 0)
        self.log('val/mask_avg_step', mask_avg, prog_bar=True, logger=True, on_step=True, on_epoch=False)

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        #logvar_t = self.logvar[t].to(self.device)
        logvar_t = self.logvar.to(self.device)[t]

        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates
    
    def _to01(self, x):
        return ((x.clamp(-1., 1.) + 1.) / 2.).float()

    def _norm01(self, x):
        mn = x.amin(dim=(2, 3), keepdim=True)
        mx = x.amax(dim=(2, 3), keepdim=True)
        return (x.float() - mn) / (mx - mn + 1e-8)

    def _rot(self, x):
        return torch.rot90(x, k=1, dims=[2, 3])

    def _mark_border(self, img3, avail, border=4):
        img3 = img3.clone()
        for n in range(img3.shape[0]):
            color = torch.tensor([0., 1., 0.] if avail[n].item() > 0.5 else [1., 0., 0.],
                                 device=img3.device).view(1, 3, 1, 1)
            img3[n:n + 1, :, :border, :] = color
            img3[n:n + 1, :, -border:, :] = color
            img3[n:n + 1, :, :, :border] = color
            img3[n:n + 1, :, :, -border:] = color
        return img3

    @torch.no_grad()
    def _build_result_grid(self, x_src_list, log, lst_, modal_names, target_size=192, border=4):
        N = x_src_list[0].shape[0]
        rows = []
        legend = torch.zeros(1, 3, 14, target_size * len(modal_names), device=self.device)
        for i in range(len(modal_names)):
            color = torch.tensor([0., 1., 0.] if lst_[0, i].item() > 0.5 else [1., 0., 0.],
                                 device=self.device).view(3, 1, 1)
            legend[0, :, :, i * target_size:(i + 1) * target_size] = color
        rows.append(legend)
        for n in range(N):
            gt_row, gen_row, err_row = [], [], []
            for i, name in enumerate(modal_names):
                gt = self._rot(self._to01(x_src_list[i][n:n + 1]))
                gen = self._rot(self._to01(log[f"samples/{name}"][n:n + 1]))
                gt_row.append(self._mark_border(gt.repeat(1, 3, 1, 1), lst_[n:n + 1, i], border))
                gen_row.append(gen.repeat(1, 3, 1, 1))
                err_row.append(self._norm01((gt - gen).abs()).repeat(1, 3, 1, 1))
            rows.append(torch.cat(gt_row, dim=3))
            rows.append(torch.cat(gen_row, dim=3))
            rows.append(torch.cat(err_row, dim=3))
        return torch.cat(rows, dim=2)

    @torch.no_grad()
    def _build_progressive_grid(self, intermediates, lst_, C, target_size=192):
        preds = intermediates.get('pred_x0', None) if isinstance(intermediates, dict) else None
        if not preds:
            return None
        miss = [i for i in range(4) if lst_[0, i].item() < 0.5]
        ti = miss[0] if miss else 0
        imgs = []
        for z in preds:
            x = self.decode_first_stage(z[0:1, ti * C:(ti + 1) * C].cpu())
            imgs.append(self._rot(self._to01(x)).repeat(1, 3, 1, 1))
        return torch.cat(imgs, dim=3)

    @torch.no_grad()
    def _build_cond_grid(self, cond, lst_, target_size=192):
        cc = cond['c_concat'] if isinstance(cond, dict) else cond
        fused = cc[:1, :128].float()
        fused_img = F.interpolate(fused.mean(dim=1, keepdim=True), size=(target_size, target_size),
                                  mode='bilinear', align_corners=False)
        imgs = [self._rot(self._norm01(fused_img)).repeat(1, 3, 1, 1)]
        for i in range(4):
            color = torch.tensor([0., 1., 0.] if lst_[0, i].item() > 0.5 else [1., 0., 0.],
                                 device=cc.device).view(1, 3, 1, 1)
            imgs.append(color.expand(1, 3, target_size, target_size).clone())
        return torch.cat(imgs, dim=3)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=4, sample=True, ddim_steps=100, ddim_eta=0., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None
        modal_names = ["t1", "t2", "t1ce", "flair"]
        log = dict()

        # -------------------------------------------------
        # get inputs (N samples)
        # -------------------------------------------------
        z_tgt, cond, lst_, x_src_list = self.get_input(
            batch, self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=N,
        )
        B, total_C, h, w = z_tgt.shape
        C = total_C // 4
        N = min(B, N)

        # -------------------------------------------------
        # 1) source images with availability border
        # -------------------------------------------------
        for i, name in enumerate(modal_names):
            img3 = self._to01(x_src_list[i]).repeat(1, 3, 1, 1)
            log[f"source/{name}"] = self._mark_border(img3, lst_[:, i])

        # -------------------------------------------------
        # 2) GT latent -> VAE decode (upper-bound reference)
        # -------------------------------------------------
        self.first_stage_model.to("cpu")
        for i, name in enumerate(modal_names):
            x_rec = self.decode_first_stage(z_tgt[:, i * C:(i + 1) * C].cpu())
            log[f"target_recon/{name}"] = x_rec.to(self.device)

        # -------------------------------------------------
        # 3) DDIM sampling with inpainting + intermediates
        # -------------------------------------------------
        intermediates = None
        if sample:
            log_every_t = max(1, int((ddim_steps or 100) // 6))
            mod_mask = lst_.repeat_interleave(C, dim=1)[:, :, None, None]  # (N,16,1,1)
            with self.ema_scope("Plotting"):
                samples, intermediates = self.sample_log(
                    cond=cond, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps,
                    eta=ddim_eta, mask=mod_mask, x0=z_tgt, log_every_t=log_every_t)
            for i, name in enumerate(modal_names):
                x_samp = self.decode_first_stage(samples[:, i * C:(i + 1) * C].cpu())
                log[f"samples/{name}"] = x_samp.to(self.device)

        # -------------------------------------------------
        # 4) process figure: denoising trajectory of a missing modality
        # -------------------------------------------------
        if intermediates is not None and plot_progressive_rows:
            prog = self._build_progressive_grid(intermediates, lst_, C)
            if prog is not None:
                log["progressive_grid"] = prog.to(self.device)

        self.first_stage_model.to(self.device)

        # -------------------------------------------------
        # 5) condition figure (fused_multiscale + availability)
        # -------------------------------------------------
        log["cond_grid"] = self._build_cond_grid(cond, lst_)

        # -------------------------------------------------
        # 6) result figure: GT | prediction | error, per sample
        # -------------------------------------------------
        if sample:
            log["result_grid"] = self._build_result_grid(x_src_list, log, lst_, modal_names)

        if return_keys:
            return {k: log[k] for k in return_keys if k in log}
        return log


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)

        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-6)

        if self.use_scheduler:
            t_max = getattr(self, 'scheduler_t_max', 100)
            warmup = getattr(self, 'warmup_epochs', 5)
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup)
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=t_max - warmup, eta_min=lr * 0.01)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])
            print(f"Setup scheduler: LinearWarmup({warmup}ep) + CosineAnnealingLR(T_max={t_max - warmup})")
            return {
                'optimizer': opt,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat = None, c_crossattn = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x,c_concat], dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            if isinstance(c_crossattn, list):
                cc = torch.cat(c_crossattn, dim=1)  # 沿通道拼接
            else:
                cc = c_crossattn
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            if isinstance(c_crossattn, list):
                cc = torch.cat(c_crossattn, dim=1)  # 沿通道拼接
            else:
                cc = c_crossattn
            xc = torch.cat([x, c_concat], dim=1)
            out = self.diffusion_model(xc, t, context=cc)
            #out = self.diffusion_model(xc, t, y=c_crossattn)

        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


