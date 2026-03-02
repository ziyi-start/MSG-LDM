import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# 1. SPADE 模块 (2D)
# ===============================
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size=3, norm_type='instance'):
        super().__init__()

        # param-free normalization
        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(f'{norm_type} is not a recognized param-free norm type in SPADE')

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
        # 这里如果输入的条件是另一个 map，这里一般是对 label map 做 resize
        x = F.interpolate(x, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(x)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out

# ===============================
# 2. 多模态 SPADE (2D)
# ===============================
class SPADE_Multimodal(nn.Module):
    def __init__(self, num_classes, norm_nc, label_nc, kernel_size, norm_type='instance'):
        super().__init__()
        self.spades = nn.ModuleList(
            [SPADE(norm_nc, label_nc, kernel_size, norm_type) for _ in range(num_classes)]
        )

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

# ===============================
# 3. SPADE ResNet Block (2D)
# ===============================
class SPADEResnetBlock(nn.Module):
    def __init__(self, num_classes, fin, fout):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # SPADE norm layers
        self.norm_0 = SPADE_Multimodal(num_classes, fin, fin, kernel_size=3, norm_type='instance')
        self.norm_1 = SPADE_Multimodal(num_classes, fmiddle, fmiddle, kernel_size=3, norm_type='instance')

    def forward(self, x, y):
        x_s = self.shortcut(x)
        dx = self.conv_0(self.actvn(self.norm_0(x, y)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, y)))
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

# ===============================
# 4. SPADE Generator (2D)
# ===============================
class SPADEGenerator(nn.Module):
    def __init__(self, num_classes=5, z_dim=4, nf=128):
        super().__init__()
        self.block = nn.ModuleList([
            SPADEResnetBlock(num_classes, z_dim, nf),
            SPADEResnetBlock(num_classes, nf, nf * 2),
            SPADEResnetBlock(num_classes, nf * 2, nf),
            SPADEResnetBlock(num_classes, nf, z_dim),
        ])

    def forward(self, x, y):
        for block in self.block:
            x = block(x, y)
        return x