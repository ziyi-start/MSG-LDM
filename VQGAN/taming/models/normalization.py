import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# SPADE 2D
# -----------------------------
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size=3, norm_type='instance'):
        super().__init__()
        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' % norm_type)

        nhidden = 64
        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x):
        normalized = self.param_free_norm(x)
        x = F.interpolate(x, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(x)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

# -----------------------------
# 多模态 SPADE 2D
# -----------------------------
class SPADE_Multimodal(nn.Module):
    def __init__(self, num_classes, norm_nc, label_nc, kernel_size, norm_type='instance'):
        super().__init__()
        self.spades = nn.ModuleList([SPADE(norm_nc, label_nc, kernel_size, norm_type) for _ in range(num_classes)])

    def forward(self, x, y):
        outputs = []
        for i in range(y.shape[0]):
            class_idx = y[i].item()
            if class_idx < len(self.spades):
                output_i = self.spades[class_idx](x[i:i+1])
                outputs.append(output_i)
            else:
                raise ValueError(f'Class {class_idx} is not a recognized class in SPADE_Multimodal')
        x = torch.cat(outputs, dim=0)
        return x

# -----------------------------
# Learnable MixStyle 2D
# -----------------------------
class LearnableMixStyle2D(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = x.var(dim=[2, 3], keepdim=True).add(self.eps).sqrt()
        x_normed = (x - mu) / sig
        out = x_normed * (sig * self.gamma) + (mu + self.beta)
        return out

# -----------------------------
# SPADEResnetBlock 2D + MixStyle
# -----------------------------
class SPADEResnetBlock(nn.Module):
    def __init__(self, num_classes, fin, fout):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        self.norm_0 = SPADE_Multimodal(num_classes, fin, fin, kernel_size=3, norm_type='instance')
        self.norm_1 = SPADE_Multimodal(num_classes, fmiddle, fmiddle, kernel_size=3, norm_type='instance')

        # self.mixstyle_0 = LearnableMixStyle2D(fmiddle)
        # self.mixstyle_1 = LearnableMixStyle2D(fout)
        self.mixstyle_0 = ConditionalMixStyle2D(fmiddle, num_classes)
        self.mixstyle_1 = ConditionalMixStyle2D(fout, num_classes)
    def forward(self, x, y):
        x_s = self.shortcut(x, y)
        dx = self.conv_0(self.actvn(self.norm_0(x, y)))
        #dx = self.mixstyle_0(dx,y)  # 第一次卷积后加 MixStyle
        dx = self.conv_1(self.actvn(self.norm_1(dx, y)))
        #dx = self.mixstyle_1(dx,y)  # 第二次卷积后加 MixStyle
        out = x_s + dx
        #x = self.mixstyle_0(out,y)  # 第一次卷积后加 MixStyle
        return out

    def shortcut(self, x, y):
        if self.learned_shortcut:
            return self.conv_s(x)
        return x

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

# -----------------------------
# Generator 2D
# -----------------------------
class SPADEGenerator(nn.Module):
    def __init__(self, num_classes=5, z_dim=4, nf=128):
        super().__init__()
        self.block = nn.ModuleList([
            SPADEResnetBlock(num_classes, z_dim, nf),
            SPADEResnetBlock(num_classes, nf, nf*2),
            SPADEResnetBlock(num_classes, nf*2, nf),
            SPADEResnetBlock(num_classes, nf, z_dim),
        ])

    def forward(self, x, y):
        for block in self.block:
            x = block(x, y)
        return x



class ConditionalMixStyle2D(nn.Module):
    def __init__(self, channels, num_classes, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 每个模态一个 gamma 和 beta
        self.gamma_embed = nn.Embedding(num_classes, channels)
        self.beta_embed = nn.Embedding(num_classes, channels)

    def forward(self, x, y):
        mu = x.mean(dim=[2,3], keepdim=True)
        sig = x.var(dim=[2,3], keepdim=True).add(self.eps).sqrt()
        x_normed = (x - mu) / sig
        gamma = self.gamma_embed(y).view(y.size(0), -1, 1, 1)
        beta = self.beta_embed(y).view(y.size(0), -1, 1, 1)
        out = x_normed * (sig * gamma) + (mu + beta)
        return out

