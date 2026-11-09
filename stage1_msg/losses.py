import torch
import torch.nn as nn
import torch.nn.functional as F


class GradLayer(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weight_h', kernel_h)
        self.register_buffer('weight_v', kernel_v)

    def get_gray(self, x):
        if x.shape[1] == 3:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            x_gray = x.mul(convert).sum(dim=1, keepdim=True)
            return x_gray
        return x

    def forward(self, x):
        """Return the *signed* gradients (g_v, g_h) instead of their magnitude."""
        x = self.get_gray(x)
        g_v = F.conv2d(x, self.weight_v, padding=1)
        g_h = F.conv2d(x, self.weight_h, padding=1)
        return g_v, g_h


class GradLoss(nn.Module):
    """Direction-preserving gradient loss.

    The vertical/horizontal derivatives are compared separately so that edge
    orientation is kept. A magnitude-only loss (sqrt(g_v^2 + g_h^2)) is
    invariant to sign flips and therefore cannot penalise wrongly oriented
    edges or checkerboard artefacts.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, target):
        out_v, out_h = self.grad_layer(output)
        tgt_v, tgt_h = self.grad_layer(target)
        return 0.5 * (self.loss(out_v, tgt_v) + self.loss(out_h, tgt_h))


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        num_classes = target.shape[1]
        dice = 0.0
        for c in range(num_classes):
            p = input[:, c].contiguous().view(num, -1)
            l = target[:, c].contiguous().view(num, -1)
            d = (2. * (p * l).sum(1) + smooth) / (p.sum(1) + l.sum(1) + smooth)
            dice = dice + d.sum() / num
        dice = 1 - dice / num_classes
        return 0.5 * bce + dice


def kl_loss(mu, logvar):
    loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1)
    return loss.mean()
