import torch
from torch.autograd import Function, Variable
import torchmetrics
import torch.nn.functional as F
from torch_dct import dct as torch_dct
from pytorch_msssim import ssim
import torch.nn as nn
import torch.nn.functional as F
class Modality_Contrastive_Loss(nn.Module):
    def __init__(self, tau=0.07, max_scale=100.0):
        super().__init__()

        self.scale_param = nn.Parameter(
            torch.log(torch.tensor(1.0 / tau))
        )
        self.upper_bound = torch.log(torch.tensor(max_scale))

    def forward(self, x):
        B, M, D = x.shape

        scale = torch.clamp(self.scale_param, 0.0, self.upper_bound).exp()

        reordered = x.transpose(0, 1).reshape(M * B, D)

        normalized = F.normalize(reordered, dim=1)

        sim_matrix = normalized @ normalized.t()
        sim_matrix = sim_matrix * scale

        group_index = torch.arange(M, device=x.device).repeat_interleave(B)
        mask = group_index[:, None] == group_index[None, :]
        mask = mask.float()
        loss_value = F.binary_cross_entropy_with_logits(sim_matrix, mask)

        return loss_value


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

def dct2(x):
    """
    对输入 x 做 2D DCT
    x: (B,C,H,W)
    返回: (B,C,H,W)
    """
    # 1D DCT 公式实现：DCT-II
    def dct_1d(v):
        return torch_dct(v, norm='ortho')

    # 先 H 方向
    x = dct_1d(x.transpose(-2, -1)).transpose(-2, -1)
    # 再 W 方向
    x = dct_1d(x)
    return x

class FreqSSIMLoss(nn.Module):
    """
    频域 SSIM 损失，基于 DCT
    """
    def __init__(self):
        super(FreqSSIMLoss, self).__init__()

    def forward(self, pred_img, gt_img):
        """
        pred_img, gt_img: (B,C,H,W), float tensor
        """
        # 做 DCT
        pred_dct = dct2(pred_img)
        gt_dct = dct2(gt_img)

        # 取幅度谱
        pred_amp = torch.abs(pred_dct)
        gt_amp = torch.abs(gt_dct)

        # 计算 SSIM
        loss = 1 - ssim(pred_amp, gt_amp, data_range=gt_amp.max() - gt_amp.min())
        return loss