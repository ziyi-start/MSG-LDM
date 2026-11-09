import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=False, dilation=1, norm_type=True, dropout=0.0, act_type='lrelu'):
        super().__init__()
        if padding == 'SAME':
            p = (kernel_size - 1) // 2
            pad = [nn.ReflectionPad2d(p)]
            conv_p = 0
        else:
            pad = []
            conv_p = 0

        layers = pad + [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                  padding=conv_p, dilation=dilation, bias=bias)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        if norm_type:
            layers.append(nn.InstanceNorm2d(out_channels, affine=False))
        if act_type == 'lrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif act_type == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act_type is None:
            pass
        self.unit = nn.Sequential(*layers)

    def forward(self, x):
        return self.unit(x)


class StyleEncoder(nn.Module):
    def __init__(self, in_channels, n_base_ch_se=32):
        super().__init__()
        self.unit = nn.Sequential(
            GeneralConv2d(in_channels, n_base_ch_se, kernel_size=7, norm_type=True, act_type='lrelu'),
            GeneralConv2d(n_base_ch_se, n_base_ch_se * 2, kernel_size=4, stride=2, norm_type=True, act_type='lrelu'),
            GeneralConv2d(n_base_ch_se * 2, n_base_ch_se * 4, kernel_size=4, stride=2, norm_type=True, act_type='lrelu'),
            GeneralConv2d(n_base_ch_se * 4, n_base_ch_se * 4, kernel_size=4, stride=2, norm_type=True, act_type='lrelu'),
            GeneralConv2d(n_base_ch_se * 4, n_base_ch_se * 4, kernel_size=4, stride=2, norm_type=True, act_type='lrelu'),
        )
        self.unit2 = GeneralConv2d(n_base_ch_se * 4, 8, kernel_size=1,
                                   norm_type=False, act_type=None, bias=True)

    def forward(self, x):
        x = self.unit(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        return self.unit2(x)


class FrequencyCueExtractor(nn.Module):
    """Multi-scale structural cue extractor for feature maps.

    For every input channel it extracts four complementary cues:
      gx / gy : signed Sobel gradients  -> edge strength + orientation
      lap     : signed Laplacian        -> curvature (ridge vs valley)
      var     : local variance          -> texture energy

    Each cue is normalized per channel by a detached running RMS, so it stays
    O(1) across scales / training and flat feature maps are not amplified into
    noise. Output has 4 * num_features channels.
    """

    CUE_NAMES = ("gx", "gy", "lap", "var")

    def __init__(self, num_features, kernel_sizes=(3, 5), momentum=0.01, eps=1e-3):
        super().__init__()
        self.num_features = int(num_features)
        self.kernel_sizes = tuple(kernel_sizes)
        self.momentum = float(momentum)
        self.eps = float(eps)

        for k in self.kernel_sizes:
            sobel_x, sobel_y, laplace = self._build_kernels(k)
            self.register_buffer(f"sobel_x_{k}", sobel_x.view(1, 1, k, k), persistent=False)
            self.register_buffer(f"sobel_y_{k}", sobel_y.view(1, 1, k, k), persistent=False)
            self.register_buffer(f"laplace_{k}", laplace.view(1, 1, k, k), persistent=False)

        self.register_buffer("running_scale", torch.ones(len(self.CUE_NAMES), self.num_features))
        self.cue_scale = nn.Parameter(torch.full((len(self.CUE_NAMES),), 0.1))

    @staticmethod
    def _build_kernels(k):
        if k == 3:
            smooth = torch.tensor([1., 2., 1.])
            deriv = torch.tensor([-1., 0., 1.])
            laplace = torch.tensor([[0., 1., 0.],
                                    [1., -4., 1.],
                                    [0., 1., 0.]])
        elif k == 5:
            smooth = torch.tensor([1., 4., 6., 4., 1.])
            deriv = torch.tensor([-1., -2., 0., 2., 1.])
            laplace = torch.tensor([[0., 0., 1., 0., 0.],
                                    [0., 1., 2., 1., 0.],
                                    [1., 2., -16., 2., 1.],
                                    [0., 1., 2., 1., 0.],
                                    [0., 0., 1., 0., 0.]])
        else:
            raise ValueError(f"unsupported kernel size: {k}")
        sobel_x = torch.outer(smooth, deriv)
        sobel_y = torch.outer(deriv, smooth)
        sobel_x = sobel_x / sobel_x.abs().sum()
        sobel_y = sobel_y / sobel_y.abs().sum()
        laplace = laplace / laplace.abs().sum()
        return sobel_x, sobel_y, laplace

    def _depthwise_filter(self, x, kernel):
        channels = x.shape[1]
        weight = kernel.repeat(channels, 1, 1, 1)
        pad = kernel.shape[-1] // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x, weight, padding=0, groups=channels)

    def _normalize(self, t, index):
        with torch.no_grad():
            scale = t.detach().pow(2).mean(dim=(0, 2, 3)).sqrt()
            if self.training:
                self.running_scale[index].mul_(1.0 - self.momentum).add_(self.momentum * scale)
        scale = self.running_scale[index].clamp(min=self.eps).view(1, -1, 1, 1)
        return t / scale

    def forward(self, x):
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(x)
        laplace = torch.zeros_like(x)
        for k in self.kernel_sizes:
            gx = gx + self._depthwise_filter(x, getattr(self, f"sobel_x_{k}"))
            gy = gy + self._depthwise_filter(x, getattr(self, f"sobel_y_{k}"))
            laplace = laplace + self._depthwise_filter(x, getattr(self, f"laplace_{k}"))
        n_scales = float(len(self.kernel_sizes))
        gx, gy, laplace = gx / n_scales, gy / n_scales, laplace / n_scales

        mean = F.avg_pool2d(x, 3, 1, 1)
        variance = (F.avg_pool2d(x * x, 3, 1, 1) - mean * mean).clamp(min=0.)

        cues = (gx, gy, laplace, variance)
        out = [self._normalize(cue, i) * self.cue_scale[i] for i, cue in enumerate(cues)]
        return torch.cat(out, dim=1)


class EdgeEnhanceSingle(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.edge_extractor = FrequencyCueExtractor(nf)
        self.conv = nn.Conv2d(nf * 5, nf, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, fea):
        edge = self.edge_extractor(fea)
        out = torch.cat([fea, edge], dim=1)
        out = self.act(self.conv(out))
        return out


class ContentEncoder(nn.Module):
    def __init__(self, in_channels, n_base_filters=16):
        super().__init__()
        self.image_cue = FrequencyCueExtractor(in_channels)
        n_cue = in_channels * len(FrequencyCueExtractor.CUE_NAMES)
        self.unit1_0 = GeneralConv2d(in_channels + n_cue, n_base_filters)
        self.unit1 = nn.Sequential(
            GeneralConv2d(n_base_filters, n_base_filters, dropout=0.2),
            GeneralConv2d(n_base_filters, n_base_filters),
        )
        self.unit2_0 = GeneralConv2d(n_base_filters, n_base_filters * 2, stride=2)
        self.unit2 = nn.Sequential(
            GeneralConv2d(n_base_filters * 2, n_base_filters * 2, dropout=0.2),
            GeneralConv2d(n_base_filters * 2, n_base_filters * 2),
        )
        self.unit3_0 = GeneralConv2d(n_base_filters * 2, n_base_filters * 4, stride=2)
        self.unit3 = nn.Sequential(
            GeneralConv2d(n_base_filters * 4, n_base_filters * 4, dropout=0.2),
            GeneralConv2d(n_base_filters * 4, n_base_filters * 4),
        )
        self.unit4_0 = GeneralConv2d(n_base_filters * 4, n_base_filters * 8, stride=2)
        self.unit4 = nn.Sequential(
            GeneralConv2d(n_base_filters * 8, n_base_filters * 8, dropout=0.2),
            GeneralConv2d(n_base_filters * 8, n_base_filters * 8),
        )

        self.edge_s1 = EdgeEnhanceSingle(n_base_filters)
        self.edge_s2 = EdgeEnhanceSingle(n_base_filters * 2)
        self.edge_s3 = EdgeEnhanceSingle(n_base_filters * 4)
        self.edge_s4 = EdgeEnhanceSingle(n_base_filters * 8)

    def forward(self, x):
        cue = self.image_cue(x)
        o1_0 = self.unit1_0(torch.cat([x, cue], dim=1))
        o1 = self.unit1(o1_0) + o1_0
        o1 = self.edge_s1(o1)

        o2_0 = self.unit2_0(o1)
        o2 = self.unit2(o2_0) + o2_0
        o2 = self.edge_s2(o2)

        o3_0 = self.unit3_0(o2)
        o3 = self.unit3(o3_0) + o3_0
        o3 = self.edge_s3(o3)

        o4_0 = self.unit4_0(o3)
        o4 = self.unit4(o4_0) + o4_0
        o4 = self.edge_s4(o4)

        return {'s1': o1, 's2': o2, 's3': o3, 's4': o4}


class linear(nn.Module):
    def __init__(self, in_ch, ch):
        super().__init__()
        self.unit = nn.Sequential(nn.Flatten(), nn.Linear(in_ch, ch))

    def forward(self, x):
        return self.unit(x)


class mlp(nn.Module):
    def __init__(self, in_ch, channel):
        super().__init__()
        self.channel = channel
        self.unit = nn.Sequential(
            linear(in_ch, channel), nn.LeakyReLU(0.2, inplace=True),
            linear(channel, channel), nn.LeakyReLU(0.2, inplace=True),
        )
        self.get_mu = linear(channel, channel)
        self.get_sigma = linear(channel, channel)

    def forward(self, style):
        s = self.unit(style)
        mu = self.get_mu(s)
        sigma = F.softplus(self.get_sigma(s)) + 1e-3
        mu = mu.view(-1, self.channel, 1, 1)
        sigma = sigma.view(-1, self.channel, 1, 1)
        return mu, sigma


class adaptive_resblock(nn.Module):
    def __init__(self, input_channel, channel):
        super().__init__()
        self.conv1 = GeneralConv2d(input_channel, channel)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = GeneralConv2d(channel, channel)
        self.affine = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel, 2 * channel, 1)
        )

    def adaptive_instance_norm(self, content, gamma, beta):
        c_mean = torch.mean(content, dim=(2, 3), keepdim=True)
        c_std = torch.std(content, dim=(2, 3), keepdim=True) + 1e-5
        return gamma * ((content - c_mean) / c_std) + beta

    def forward(self, x_init, style_z):
        affine_params = self.affine(style_z)
        gamma, beta = torch.chunk(affine_params, 2, dim=1)
        x = self.adaptive_instance_norm(self.conv1(x_init), gamma, beta)
        x = self.lrelu(x)
        x = self.adaptive_instance_norm(self.conv2(x), gamma, beta)
        return x + x_init


class ImageDecoder(nn.Module):
    def __init__(self, input_channel, mlp_ch=128, img_ch=1, scale=4):
        super().__init__()
        channel = mlp_ch
        self.scale = scale
        self.ar1 = adaptive_resblock(input_channel, channel)
        self.ar2 = adaptive_resblock(channel, channel)
        self.ar3 = adaptive_resblock(channel, channel)
        self.ar4 = adaptive_resblock(channel, channel)
        self.mlp = mlp(8, channel)
        self.up_blocks = nn.ModuleList()
        in_ch = channel
        for _ in range(scale - 1):
            out_ch = in_ch // 2
            self.up_blocks.append(nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_ch, out_ch, 5, bias=True),
                nn.InstanceNorm2d(out_ch, affine=True, eps=1e-6),
                nn.ReLU(inplace=True),
            ]))
            in_ch = out_ch
        self.conv_final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, img_ch, 7, bias=True),
        )

    def forward(self, style, content):
        mu, sigma = self.mlp(style)
        if self.training:
            eps = torch.randn_like(sigma)
            style_z = mu + eps * sigma
        else:
            style_z = mu
        x = self.ar1(content, style_z)
        x = self.ar2(x, style_z)
        x = self.ar3(x, style_z)
        x = self.ar4(x, style_z)
        for up, pad, conv, norm, act in self.up_blocks:
            x = up(x)
            x = pad(x)
            x = conv(x)
            x = norm(x)
            x = act(x)
        return torch.tanh(self.conv_final(x)), mu, sigma


class MaskDecoder(nn.Module):
    def __init__(self, input_channel, n_base_filters=16, num_cls=4):
        super().__init__()
        self.ups = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.conv3 = nn.ModuleList()
        in_ch = input_channel
        out_ch = n_base_filters * 4
        for i in range(3):
            self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.conv1.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.3),
            ))
            self.conv2.append(nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.3),
            ))
            self.conv3.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
            out_ch //= 2
        self.conv_seg = nn.Conv2d(in_ch, num_cls, kernel_size=1, bias=True)

    def forward(self, inp):
        skips = [inp['e4_out'], inp['e3_out'], inp['e2_out'], inp['e1_out']]
        out = skips[0]
        for i in range(3):
            out = self.ups[i](out)
            out = self.conv1[i](out)
            out = torch.cat([out, skips[i + 1]], dim=1)
            out = self.conv2[i](out)
            out = self.conv3[i](out)
        return self.conv_seg(out)


class HybridCIFModule(nn.Module):
    def __init__(self, channels_per_modal, reduction=4):
        super().__init__()
        self.c_per_modal = channels_per_modal
        self.c_total = channels_per_modal * 4

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Linear(self.c_total, self.c_total // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c_total // reduction, self.c_total, bias=False),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(self.c_total, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=1),
            nn.Sigmoid()
        )

        self.gate1 = nn.Conv2d(self.c_per_modal, self.c_per_modal, kernel_size=1)
        self.gate2 = nn.Conv2d(self.c_per_modal, self.c_per_modal, kernel_size=1)
        self.gate3 = nn.Conv2d(self.c_per_modal, self.c_per_modal, kernel_size=1)
        self.gate4 = nn.Conv2d(self.c_per_modal, self.c_per_modal, kernel_size=1)

        self.final_fusion = nn.Sequential(
            nn.Conv2d(self.c_total, self.c_per_modal, kernel_size=1),
            nn.InstanceNorm2d(self.c_per_modal),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x1, x2, x3, x4, mask=None):
        b, c, h, w = x1.size()
        x_all = torch.cat([x1, x2, x3, x4], dim=1)

        c_weight = self.gap(x_all).view(b, self.c_total)
        c_weight = self.channel_att(c_weight).view(b, self.c_total, 1, 1)
        if mask is not None:
            m_exp = mask.repeat_interleave(self.c_per_modal, dim=1).unsqueeze(-1).unsqueeze(-1)
            c_weight = c_weight * m_exp
        x_c_weighted = x_all * c_weight

        s_weight = self.spatial_att(x_all)
        if mask is not None:
            s_weight = s_weight * mask.unsqueeze(-1).unsqueeze(-1)

        f1, f2, f3, f4 = torch.split(x_c_weighted, self.c_per_modal, dim=1)
        f1_refined = f1 * s_weight[:, 0:1]
        f2_refined = f2 * s_weight[:, 1:2]
        f3_refined = f3 * s_weight[:, 2:3]
        f4_refined = f4 * s_weight[:, 3:4]

        f_list = [f1_refined, f2_refined, f3_refined, f4_refined]
        f_sum = f1_refined + f2_refined + f3_refined + f4_refined

        g1 = torch.sigmoid(self.gate1(x1))
        g2 = torch.sigmoid(self.gate2(x2))
        g3 = torch.sigmoid(self.gate3(x3))
        g4 = torch.sigmoid(self.gate4(x4))

        if mask is not None:
            m1 = mask[:, 0:1].unsqueeze(-1).unsqueeze(-1)
            m2 = mask[:, 1:2].unsqueeze(-1).unsqueeze(-1)
            m3 = mask[:, 2:3].unsqueeze(-1).unsqueeze(-1)
            m4 = mask[:, 3:4].unsqueeze(-1).unsqueeze(-1)
            eff_g = [g1 * m1 + (1 - m1), g2 * m2 + (1 - m2),
                     g3 * m3 + (1 - m3), g4 * m4 + (1 - m4)]
            n_present = mask.sum(dim=1, keepdim=True)
        else:
            eff_g = [g1, g2, g3, g4]
            n_present = f1_refined.new_full((b, 1), float(len(f_list)))

        outs = []
        for i, fi in enumerate(f_list):
            if mask is not None:
                denom = (n_present - mask[:, i:i+1]).clamp(min=1)
            else:
                denom = (n_present - 1).clamp(min=1)
            context = (f_sum - fi) / denom.view(b, 1, 1, 1)
            outs.append(fi + eff_g[i] * context)

        out_fused = self.final_fusion(torch.cat(outs, dim=1))
        return out_fused


class MultiScaleDynamicFusion(nn.Module):
    def __init__(self, c1, c2, c3, c4, heads=4):
        super().__init__()
        self.c4 = c4
        self.proj1 = nn.Conv2d(c1, c4, 1)
        self.proj2 = nn.Conv2d(c2, c4, 1)
        self.proj3 = nn.Conv2d(c3, c4, 1)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(c4 * 3, c4, kernel_size=1),
            nn.InstanceNorm2d(c4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.norm = nn.LayerNorm(c4)
        self.q = nn.Conv2d(c4, c4, 1)
        self.k = nn.Conv2d(c4, c4, 1)
        self.v = nn.Conv2d(c4, c4, 1)
        self.heads = heads
        self.scale = (c4 // heads) ** -0.5
        self.out = nn.Conv2d(c4, c4, 1)

    def forward(self, c1, c2, c3, c4):
        B, C, H, W = c4.shape
        size = (H, W)
        c1 = F.interpolate(self.proj1(c1), size=size, mode='area')
        c2 = F.interpolate(self.proj2(c2), size=size, mode='area')
        c3 = F.interpolate(self.proj3(c3), size=size, mode='area')
        structure = self.fuse_conv(torch.cat([c1, c2, c3], dim=1))

        x = c4.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        q = self.q(x)
        k = self.k(structure)
        v = self.v(structure)

        def reshape(x):
            return x.reshape(B, self.heads, C // self.heads, H * W)

        q, k, v = map(reshape, (q, k, v))
        attn = torch.softmax(
            torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale, dim=-1
        )
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.out(out)
        return c4 + 0.5 * out


class DisentangleModel(nn.Module):
    def __init__(self, modal_num, in_channels, n_base_filters=16):
        super().__init__()
        self.modal_num = modal_num
        mlp_ch = 128

        self.content_encoder = ContentEncoder(in_channels, n_base_filters)

        self.style_encoder_list = nn.ModuleList(
            [StyleEncoder(in_channels) for _ in range(modal_num)]
        )

        self.mask_decoder = MaskDecoder(
            input_channel=n_base_filters * 8,
            n_base_filters=n_base_filters,
            num_cls=4
        )

        self.hybrid_cif_s1 = HybridCIFModule(n_base_filters)
        self.hybrid_cif_s2 = HybridCIFModule(n_base_filters * 2)
        self.hybrid_cif_s3 = HybridCIFModule(n_base_filters * 4)
        self.hybrid_cif_s4 = HybridCIFModule(n_base_filters * 8)

        self.image_decoder = nn.ModuleList(
            [ImageDecoder(n_base_filters * 8, mlp_ch) for _ in range(modal_num)]
        )

        self.ms_fusion = MultiScaleDynamicFusion(
            c1=n_base_filters,
            c2=n_base_filters * 2,
            c3=n_base_filters * 4,
            c4=n_base_filters * 8,
        )

    def forward(self, inputs, lst):
        assert inputs.shape[1] % self.modal_num == 0
        inputs = torch.chunk(inputs, self.modal_num, dim=1)
        device = inputs[0].device
        mask = torch.tensor(lst, dtype=torch.float32, device=device)

        masked_inputs = [inputs[i] * mask[:, i:i+1, None, None] for i in range(self.modal_num)]
        content = [self.content_encoder(masked_inputs[i]) for i in range(self.modal_num)]
        # The style encoder intentionally sees the full input during Stage-I
        # training. It only feeds the auxiliary reconstruction branch; Stage II
        # consumes z_c (fused_multiscale), which comes from the masked content path.
        style = [self.style_encoder_list[i](inputs[i]) for i in range(self.modal_num)]

        s1 = [c['s1'] for c in content]
        s2 = [c['s2'] for c in content]
        s3 = [c['s3'] for c in content]
        s4 = [c['s4'] for c in content]

        fused_c1 = self.hybrid_cif_s1(*s1, mask=mask)
        fused_c2 = self.hybrid_cif_s2(*s2, mask=mask)
        fused_c3 = self.hybrid_cif_s3(*s3, mask=mask)
        fused_c4 = self.hybrid_cif_s4(*s4, mask=mask)

        fused_multiscale = self.ms_fusion(fused_c1, fused_c2, fused_c3, fused_c4)

        recon, mus, sigmas = [], [], []
        for i in range(self.modal_num):
            img, mu, sigma = self.image_decoder[i](style[i], fused_multiscale)
            recon.append(img)
            mus.append(mu)
            sigmas.append(sigma)

        seg = self.mask_decoder({
            'e1_out': fused_c1, 'e2_out': fused_c2,
            'e3_out': fused_c3, 'e4_out': fused_c4,
        })

        return {
            'style': style, 'reconstruct_image': recon,
            'mu': mus, 'sigma': sigmas,
            'content_c1': fused_c1, 'content_c2': fused_c2,
            'content_c3': fused_c3, 'content_c4': fused_c4,
            'seg': seg, 'fused_multiscale': fused_multiscale,
        }
