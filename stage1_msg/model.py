import os
import torch
import pytorch_lightning as pl
import random

from disentangle import DisentangleModel
from losses import GradLoss, BCEDiceLoss, kl_loss


class VQModel(pl.LightningModule):
    def __init__(self, n_modal=4, in_channels=1, max_epochs=100, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.model = DisentangleModel(modal_num=n_modal, in_channels=in_channels)
        self.grad_loss_fn = GradLoss()
        self.seg_criterion = BCEDiceLoss()
        self.monitor = "val/total_loss"
        self.max_epochs = max_epochs
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def forward(self, batch, lst=None):
        x = torch.cat([batch["t1"], batch["t2"], batch["t1ce"], batch["flair"]], dim=1)

        if lst is None:
            batch_size = x.shape[0]
            modal_num = self.model.modal_num
            if os.environ.get("NO_MASK", "0") == "1":
                lst = [[1] * modal_num for _ in range(batch_size)]
            else:
                lst = []
                for _ in range(batch_size):
                    mask = [random.randint(0, 1) for _ in range(modal_num)]
                    if not any(mask):
                        mask[random.randint(0, modal_num - 1)] = 1
                    lst.append(mask)

        return self.model(x, lst)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        reconstructions = self(batch)
        aeloss, log_dict_ae = self.get_loss(batch, reconstructions, split='train')
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    def get_loss(self, batch, reconstructs, split='train'):
        rec_loss = 0
        grad_loss = 0
        rec_name = ['t1', 't2', 't1ce', 'flair']
        for i, name in enumerate(rec_name):
            gen_img = reconstructs['reconstruct_image'][i]
            real_img = batch[name]
            rec_loss += torch.mean(torch.abs(gen_img - real_img))
            grad_loss += self.grad_loss_fn(gen_img, real_img)

        rec_loss_total = rec_loss + grad_loss

        KL_loss = 0
        for i in range(len(reconstructs['mu'])):
            mu = reconstructs['mu'][i]
            logvar = torch.log(reconstructs['sigma'][i].pow(2).clamp(min=1e-6))
            KL_loss += kl_loss(mu, logvar)

        seg_loss = self.seg_criterion(reconstructs['seg'], batch['seg'])

        loss = (
            1.0 * rec_loss_total +
            1.0 * seg_loss +
            0.1 * KL_loss
        )

        dice_scores = self._compute_dice(reconstructs['seg'], batch['seg'])

        log = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/rec_loss": rec_loss.detach(),
            f"{split}/grad_loss": grad_loss.detach(),
            f"{split}/seg_loss": seg_loss.detach(),
            f"{split}/kl_loss": KL_loss.detach(),
            f"{split}/dice_WT": dice_scores['WT'],
            f"{split}/dice_TC": dice_scores['TC'],
            f"{split}/dice_ET": dice_scores['ET'],
        }

        return loss, log

    def _compute_dice(self, input, target):
        smooth = 1e-5
        input_sig = torch.sigmoid(input)
        num = target.size(0)
        class_names = {1: 'WT', 2: 'TC', 3: 'ET'}
        scores = {}
        for c in range(1, target.shape[1]):
            input_c = input_sig[:, c].view(num, -1)
            target_c = target[:, c].view(num, -1)
            keep = target_c.sum(1) > 0
            if keep.sum() == 0:
                scores[class_names[c]] = torch.tensor(0.0, device=input.device)
                continue
            inter = (input_c[keep] * target_c[keep]).sum(1)
            union = input_c[keep].sum(1) + target_c[keep].sum(1)
            dice_c = ((2. * inter + smooth) / (union + smooth)).mean()
            scores[class_names[c]] = dice_c.detach()
        return scores

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        reconstructions = self(batch)
        aeloss, log_dict_ae = self.get_loss(batch, reconstructions, split="val" + suffix)

        self.log(f"val{suffix}/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, sync_dist=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(list(self.model.parameters()), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        _step = os.environ.get("LR_STEP", None)
        if _step is not None:
            sched = torch.optim.lr_scheduler.StepLR(opt_ae, step_size=int(_step), gamma=0.5)
        else:
            _ms = os.environ.get("LR_MILESTONES", "15500,31000,46500")
            milestones = [int(m) for m in _ms.split(",")]
            sched = torch.optim.lr_scheduler.MultiStepLR(opt_ae, milestones=milestones, gamma=0.5)
        return [opt_ae], [{"scheduler": sched, "interval": "step"}]

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = {}

        for key in ['t1', 't2', 't1ce', 'flair']:
            log[f"inputs/{key}"] = batch[key].detach().cpu()

        if only_inputs:
            return log

        batch_size = batch["t1"].shape[0]
        modal_num = self.model.modal_num
        lst = [[1] * modal_num for _ in range(batch_size)]
        outputs = self(batch, lst=lst)

        modality_names = ['t1', 't2', 't1ce', 'flair']
        for i, key in enumerate(modality_names):
            recon = outputs['reconstruct_image'][i].detach().cpu()
            log[f"reconstruction/{key}"] = recon

        pred_seg = torch.sigmoid(outputs['seg']).detach().cpu()
        gt_seg = batch['seg'].detach().cpu()

        def make_color_mask(mask):
            color_map = torch.zeros((3, mask.shape[1], mask.shape[2]))
            color_map[0] = mask[1]
            color_map[1] = mask[2]
            color_map[2] = mask[3]
            color_map = (color_map - color_map.min()) / (color_map.max() - color_map.min() + 1e-5)
            return color_map

        pred_color = torch.stack([make_color_mask(pred_seg[i]) for i in range(pred_seg.size(0))], dim=0)
        gt_color = torch.stack([make_color_mask(gt_seg[i]) for i in range(gt_seg.size(0))], dim=0)

        log["segmentation/pred_color"] = pred_color
        log["segmentation/gt_color"] = gt_color

        return log
