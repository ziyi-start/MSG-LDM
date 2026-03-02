import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import torch.nn as nn
from taming.modules.diffusionmodules.model2 import DisentangleModel
from taming.util import instantiate_from_config

from taming.modules.losses.loss import Modality_Contrastive_Loss,FreqSSIMLoss,BCEDiceLoss
import random  # 推荐放在文件最上面import random  # 推荐放在文件最上面

    
class VQModel(pl.LightningModule):
    def __init__(self, n_modal=4, in_channels=1,ckpt_path=None,ignore_keys=[]):
        super().__init__()
        self.model = DisentangleModel(modal_num=n_modal, in_channels=in_channels)
        self.modality_contrastive_loss = Modality_Contrastive_Loss()
        self.freq_ssim_fn = FreqSSIMLoss()
        self.criterion = BCEDiceLoss()
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

        x = torch.cat([batch["t1"], batch["t2"], batch["t1ce"], batch["flair"]], dim=1)  # (B,4,H,W)

        #掩码
        if lst is None:
            batch_size = x.shape[0]
            modal_num = self.model.modal_num  # 4
            lst = []
            for _ in range(batch_size):
                num_mask = random.choice([0, 1, 2, 3])
                mask_indices = random.sample(range(modal_num), num_mask)
                mask = [0 if i in mask_indices else 1 for i in range(modal_num)]
                lst.append(mask)


        return self.model(x, lst)


    def training_step(self, batch, batch_idx, optimizer_idx=None):
        reconstructions= self(batch)
        aeloss, log_dict_ae = self.get_loss(batch, reconstructions,split='train')
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
 
        return aeloss
    
    def get_loss(self, batch, reconstructs, split='train', lambda_l2=1e-4):
        """
        reconstructs 字典来自 DisentangleModel.forward()
        lambda_l2: L2正则权重
        """

        # ===============================
        # 1️⃣ 重建损失 (L1 + Grad + FreqSSIM)
        # ===============================
        rec_loss = 0
        grad_loss = 0
        freq_ssim_loss = 0
        rec_name = ['t1', 't2', 't1ce', 'flair']
        for i, name in enumerate(rec_name):
            gen_img = reconstructs['reconstruct_image'][i]  # (B,1,H,W)
            real_img = batch[name]                           # (B,1,H,W)

            rec_loss += torch.mean(torch.abs(gen_img - real_img))
            freq_ssim_loss += self.freq_ssim_fn(gen_img, real_img)

        lambda_freq_dct = 1.0  # 权重，可调
        lambda_rec = 1.0  # 权重，可调
        rec_loss_total = lambda_rec*rec_loss + lambda_freq_dct * freq_ssim_loss 

        # ===============================
        # KL损失
        # ===============================
        KL_loss = 0
        for i in range(len(reconstructs['mu'])):
            mu = reconstructs['mu'][i]
            logvar = torch.log(torch.pow(reconstructs['sigma'][i], 2))
            KL_loss += self.kl_loss(mu, logvar)

        # ===============================
        # 分割损失 (BCE + Dice)
        # ===============================
        seg_loss = self.criterion(reconstructs['seg'], batch['seg'])

        # ===============================
        # 风格对比损失（Modality对比）
        # ===============================
        style_features = torch.stack(reconstructs['style'], dim=1).squeeze(-1).squeeze(-1)  # (B,4,8)
        modality_loss = self.modality_contrastive_loss(style_features)

        # ===============================
        # L2 正则（对模型所有可训练参数）
        # ===============================
        l2_loss = 0.0
        for param in self.model.parameters():
            l2_loss += torch.sum(param ** 2)
        l2_loss = lambda_l2 * l2_loss
        # ===============================
        # 🎯 最终损失
        # ===============================
        loss = (
            1.0 * rec_loss_total +          # 重建
            2.0 * seg_loss +                # 分割
            0.05 * KL_loss +                # KL
            0.1 * modality_loss +           # 风格对比
            l2_loss                         # L2正则
        )

        log = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/rec_loss": rec_loss.detach(),
            f"{split}/freq_ssim_loss": freq_ssim_loss.detach(),
            f"{split}/seg_loss": seg_loss.detach(),
            f"{split}/kl_loss": KL_loss.detach(),
            f"{split}/modality_loss": modality_loss.detach(),
            f"{split}/l2_loss": l2_loss.detach(),
        }

        return loss, log


    def kl_loss(self, mu, logvar):
        loss = 0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(logvar) - 1 - logvar, dim=1)
        loss = torch.mean(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        reconstructions= self(batch)
        aeloss, log_dict_ae = self.get_loss(batch, reconstructions,split="val"+suffix)

        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.model.parameters()), lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        """
        记录输入图像、重建图像、融合分割图
        输出格式适配 TensorBoard / WandB
        每个 batch item 都输出一组：四个模态 + 分割图
        """
        log = {}

        # ------------------ 1️⃣ 记录输入模态 ------------------
        # 你目前的 batch 应该是 4 个模态拼到一起输入，如 [B, 4, H, W]
        # 如果 batch 是分开的（batch['t1']...），保持原样
        for key in ['t1', 't2', 't1ce', 'flair']:
            log[f"inputs/{key}"] = batch[key].detach().cpu()

        if only_inputs:
            return log

        # ------------------ 2️⃣ 前向推理 ------------------
        outputs = self(batch)

        # ------------------ 3️⃣ 重建图像 ------------------
        # outputs['reconstruct_image'] 是 list，长度=modal_num，每个 (B,1,H,W)
        modality_names = ['t1', 't2', 't1ce', 'flair']
        for i, key in enumerate(modality_names):
            recon = outputs['reconstruct_image'][i].detach().cpu()  # index 对应模态顺序
            log[f"reconstruction/{key}"] = recon

        # ------------------ 4️⃣ 分割结果 ------------------
        pred_seg = torch.sigmoid(outputs['seg']).detach().cpu()  # [B,4,H,W]
        gt_seg = batch['seg'].detach().cpu()                     # [B,4,H,W]

        # --------- 🔥 把 mask 转成彩色图 ---------
        def make_color_mask(mask):
            """把 [C, H, W] 转成彩色可视化 [3, H, W]"""
            color_map = torch.zeros((3, mask.shape[1], mask.shape[2]))

            # 🚨注意你标签定义：
            # label 1: NCR（坏死肿瘤核心）
            # label 2: ED（水肿）
            # label 4: ET（增强肿瘤）
            # --> 所以你这里映射错了，应改为：
            # 建议：红=NCR，绿=ED，蓝=ET
            color_map[0] = mask[1]  # NCR
            color_map[1] = mask[2]  # ED
            color_map[2] = mask[3]  # ET（第四通道，index=3）

            color_map = (color_map - color_map.min()) / (color_map.max() - color_map.min() + 1e-5)
            return color_map

        # 对每个 batch item 都生成彩色图
        pred_color = torch.stack([make_color_mask(pred_seg[i]) for i in range(pred_seg.size(0))], dim=0)
        gt_color = torch.stack([make_color_mask(gt_seg[i]) for i in range(gt_seg.size(0))], dim=0)

        log["segmentation/pred_color"] = pred_color
        log["segmentation/gt_color"] = gt_color

        return log

