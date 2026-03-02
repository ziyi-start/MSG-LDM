import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type=True, dropout=0.0, act=True):
        super(GeneralConv2d, self).__init__()
        p = kernel_size // 2
        self.unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=p, stride=stride, bias=bias, dilation=dilation)
        )
        self.unit.add_module('drop', nn.Dropout(p=dropout))
        self.unit.add_module('norm', nn.InstanceNorm2d(out_channels))
        self.unit.add_module('activation', nn.LeakyReLU(0.01, inplace=True))

    def forward(self, inputs):
        return self.unit(inputs)
class StyleEncoder(nn.Module):
    def __init__(self, in_channels, n_base_ch_se=32):
        super(StyleEncoder, self).__init__()

        self.unit = nn.Sequential(
            GeneralConv2d(in_channels, n_base_ch_se, kernel_size=7, stride=1),
            GeneralConv2d(n_base_ch_se, n_base_ch_se*2, kernel_size=4, stride=2),
            GeneralConv2d(n_base_ch_se*2, n_base_ch_se*4, kernel_size=4, stride=2),
            GeneralConv2d(n_base_ch_se*4, n_base_ch_se*4, kernel_size=4, stride=2),
            GeneralConv2d(n_base_ch_se*4, n_base_ch_se*4, kernel_size=4, stride=2),
        )
        self.unit2 = GeneralConv2d(n_base_ch_se*4, 8, kernel_size=1, stride=1, norm_type=False, act=False)

    def forward(self, inputs):
        output = self.unit(inputs)
        output = torch.mean(output, dim=(2, 3), keepdim=True)
        return self.unit2(output)

class ContentEncoder(nn.Module):
    def __init__(self, in_channels, n_base_filters=32):
        super(ContentEncoder, self).__init__()
        self.unit1_0 = GeneralConv2d(in_channels, n_base_filters)
        self.unit1 = nn.Sequential(
            GeneralConv2d(n_base_filters, n_base_filters, dropout=0.3),
            GeneralConv2d(n_base_filters, n_base_filters),
        )

        self.unit2_0 = GeneralConv2d(n_base_filters, n_base_filters*2, stride=2)
        self.unit2 = nn.Sequential(
            GeneralConv2d(n_base_filters*2, n_base_filters*2, dropout=0.3),
            GeneralConv2d(n_base_filters*2, n_base_filters*2),
        )

        self.unit3_0 = GeneralConv2d(n_base_filters*2, n_base_filters*4, stride=2)
        self.unit3 = nn.Sequential(
            GeneralConv2d(n_base_filters*4, n_base_filters*4, dropout=0.3),
            GeneralConv2d(n_base_filters*4, n_base_filters*4 ),
        )

        self.unit4_0 = GeneralConv2d(n_base_filters * 4, n_base_filters*8, stride=2)
        self.unit4 = nn.Sequential(
            GeneralConv2d(n_base_filters*8, n_base_filters * 8, dropout=0.3),
            GeneralConv2d(n_base_filters * 8, n_base_filters * 8),
        )
        self.edge_s1 = EdgeEnhanceSingle(n_base_filters)
        self.edge_s2 = EdgeEnhanceSingle(n_base_filters * 2)
        self.edge_s3 = EdgeEnhanceSingle(n_base_filters * 4)
        self.edge_s4 = EdgeEnhanceSingle(n_base_filters * 8)

    def forward(self, inputs):
        # ===== s1 =====
        output1_0 = self.unit1_0(inputs)
        output1 = self.unit1(output1_0) + output1_0
        output1 = self.edge_s1(output1)   

        # ===== s2 =====
        output2_0 = self.unit2_0(output1)
        output2 = self.unit2(output2_0) + output2_0
        output2 = self.edge_s2(output2)

        # ===== s3 =====
        output3_0 = self.unit3_0(output2)
        output3 = self.unit3(output3_0) + output3_0
        output3 = self.edge_s3(output3)

        # ===== s4 =====
        output4_0 = self.unit4_0(output3)
        output4 = self.unit4(output4_0) + output4_0
        output4 = self.edge_s4(output4)   

        return {
            's1': output1,
            's2': output2,
            's3': output3,
            's4': output4,
        }


class linear(nn.Module):
    def __init__(self, in_ch, ch):

        super(linear, self).__init__()
        self.unit = nn.Sequential(nn.Flatten(),
                                  nn.Linear(in_ch, ch))
    def forward(self, inputs):
        return self.unit(inputs)

class mlp(nn.Module):
    def __init__(self, in_ch, channel):
        super(mlp, self).__init__()
        self.channel = channel
        self.unit = nn.Sequential(
            linear(in_ch, channel),
            nn.LeakyReLU(0.01),
            linear(channel, channel),
            nn.LeakyReLU(0.01),
        )
        self.get_mu = linear(channel, channel)
        self.get_sigma = linear(channel, channel)

    def forward(self, style):
        s = self.unit(style)
        mu = self.get_mu(s)
        sigma = self.get_sigma(s)

        mu = mu.view(-1, self.channel, 1, 1)
        sigma = sigma.view(-1, self.channel, 1, 1)

        return mu, sigma

class adaptive_resblock(nn.Module):
    def __init__(self,input_channel, channel):
        super(adaptive_resblock, self).__init__()
        self.conv1 = GeneralConv2d(input_channel, channel)
        self.lrelu = nn.LeakyReLU(0.01)
        self.conv2 = GeneralConv2d(channel, channel)


    def forward(self, x_init, mu, sigma):
        x = self.adaptive_instance_norm(self.conv1(x_init), mu, sigma)
        x = self.lrelu(x)
        x = self.adaptive_instance_norm(self.conv2(x), mu, sigma)
        return x + x_init


    def adaptive_instance_norm(self, content, gamma, beta):
        c_mean = torch.mean(content, dim=(2, 3), keepdim=True)
        c_std = torch.std(content, dim=(2, 3), keepdim=True)
        return gamma * ((content - c_mean) / c_std) + beta


class ImageDecoder(nn.Module):
    def __init__(self, input_channel, mlp_ch=128, img_ch=1, scale=4):
        super(ImageDecoder, self).__init__()
        channel = mlp_ch
        self.scale = scale
        self.ar1 = adaptive_resblock(input_channel, channel)
        self.ar2 = adaptive_resblock(channel, channel)
        self.ar3 = adaptive_resblock(channel, channel)
        self.ar4 = adaptive_resblock(channel, channel)

        self.mlp = mlp(8, channel)
        self.features = nn.Sequential()
        in_channel = channel
        out_channel = channel
        self.lrelu = nn.LeakyReLU(0.01)
        for i in range(scale-1):
            out_channel = in_channel // 2
            up_block = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            conv_block = nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2)
            norm_block = nn.InstanceNorm2d(out_channel)
            active_block = nn.LeakyReLU(0.01, inplace=True)
            self.features.add_module('upblock%d' % (i+1), up_block)
            self.features.add_module('convblock%d' % (i+1), conv_block)
            self.features.add_module('normblock%d' % (i+1), norm_block)
            self.features.add_module('activeblock%d' % (i+1), active_block)
            in_channel = out_channel

        self.conv_final = nn.Conv2d(in_channel, img_ch, kernel_size=7, stride=1, padding=3)


    def forward(self, style, content):
        mu, sigma = self.mlp(style)
        x = self.ar1(content, mu, sigma)
        x = self.ar2(x, mu, sigma)
        x = self.ar3(x, mu, sigma)
        x = self.ar4(x, mu, sigma)

        for i in range(self.scale - 1):
            x = getattr(self.features, 'upblock%d' % (i+1))(x)
            x = getattr(self.features, 'convblock%d' % (i+1))(x)
            x = getattr(self.features, 'normblock%d' % (i+1))(x)
            x = getattr(self.features, 'activeblock%d' % (i+1))(x)
            x = self.lrelu(x)

        x = self.conv_final(x)
        x= torch.tanh(x)
        return x, mu, sigma

class MaskDecoder(nn.Module):
    def __init__(self, input_channel, n_base_filters=16, num_cls=4):
        super(MaskDecoder, self).__init__()

        self.features = nn.Sequential()
        in_channel = input_channel
        out_channel = n_base_filters * 4
        for i in range(3):
            up_block = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            conv_block1 = GeneralConv2d(in_channel, out_channel)
            conv_block2 = GeneralConv2d(out_channel*2, out_channel)
            conv_block3 = GeneralConv2d(out_channel, out_channel, kernel_size=1)

            self.features.add_module('upblock%d' % (i+1), up_block)
            self.features.add_module('convblock1%d' % (i+1), conv_block1)
            self.features.add_module('convblock2%d' % (i+1), conv_block2)
            self.features.add_module('convblock3%d' % (i+1), conv_block3)

            in_channel = out_channel
            out_channel = out_channel // 2

        self.conv_seg = GeneralConv2d(in_channel, num_cls, kernel_size=1, norm_type=False, act=False)

    def forward(self, inp):
        input = [inp['e4_out'], inp['e3_out'], inp['e2_out'], inp['e1_out']]

        out = input[0]
        for i in range(3):
            out = getattr(self.features, 'upblock%d' % (i + 1))(out)
            out = getattr(self.features, 'convblock1%d' % (i + 1))(out)
            out = torch.cat([out, input[i+1]], dim=1)
            out = getattr(self.features, 'convblock2%d' % (i + 1))(out)
            out = getattr(self.features, 'convblock3%d' % (i + 1))(out)
        seg = self.conv_seg(out)

        return seg


class DisentangleModel(nn.Module):
    def __init__(self, modal_num, in_channels, n_base_filters=32):
        super().__init__()

        mlp_ch = n_base_filters * 8
        self.modal_num = modal_num
        self.sigmoid = nn.Sigmoid()
        # ===== encoders =====
        self.content_encoder_list = nn.ModuleList(
            [ContentEncoder(in_channels, n_base_filters) for _ in range(modal_num)]
        )
        self.style_encoder_list = nn.ModuleList(
            [StyleEncoder(in_channels) for _ in range(modal_num)]
        )

        self.mask_decoder = MaskDecoder(
            input_channel=n_base_filters * 8,
            n_base_filters=n_base_filters,
            num_cls=4
        )

        # ===== attention =====
        self.attm1 = GeneralConv2d(n_base_filters * modal_num, modal_num)
        self.attm2 = GeneralConv2d(n_base_filters * 2 * modal_num, modal_num)
        self.attm3 = GeneralConv2d(n_base_filters * 4 * modal_num, modal_num)
        self.attm4 = GeneralConv2d(n_base_filters * 8 * modal_num, modal_num)

        # ===== fusion =====
        self.fusion1 = GeneralConv2d(n_base_filters * modal_num, n_base_filters, 1)
        self.fusion2 = GeneralConv2d(n_base_filters * 2 * modal_num, n_base_filters * 2, 1)
        self.fusion3 = GeneralConv2d(n_base_filters * 4 * modal_num, n_base_filters * 4, 1)
        self.fusion4 = GeneralConv2d(n_base_filters * 8 * modal_num, n_base_filters * 8, 1)

        self.image_decoder = nn.ModuleList(
            [ImageDecoder(n_base_filters * 8, mlp_ch) for _ in range(modal_num)]
        )

        self.ms_fusion = MultiScaleDynamicFusion(
            c1=n_base_filters,
            c2=n_base_filters * 2,
            c3=n_base_filters * 4,
            c4=n_base_filters * 8,
        )

    def mask_content_features(self, content_list, lst):
        B = len(lst)
        M = self.modal_num

        # (B,M) → (1,M,1,1,1)
        mask = torch.tensor(lst, dtype=torch.float32, device=content_list[0]['s1'].device)
        mask = mask.transpose(0,1)[:, :, None, None, None]   # (M,B,1,1,1)

        for m in range(M):
            for k in ["s1", "s2", "s3", "s4"]:
                # content_list[m][k]: (B,C,H,W)
                feat = content_list[m][k]
                content_list[m][k] = feat * mask[m]   

        return content_list

    def forward(self, inputs, lst):
        assert inputs.shape[1] % self.modal_num == 0, "Input channels must be divisible by modal_num"
        inputs = torch.chunk(inputs, chunks=self.modal_num, dim=1)
        content_list = [self.content_encoder_list[idx](modal) for idx, modal in enumerate(inputs)]
        style_list = [self.style_encoder_list[idx](modal) for idx, modal in enumerate(inputs)]
    
        content_list = self.mask_content_features(content_list, lst)
        content_s1_list = [content['s1'] for content in content_list]
        content_s2_list = [content['s2'] for content in content_list]
        content_s3_list = [content['s3'] for content in content_list]
        content_s4_list = [content['s4'] for content in content_list]

        device = inputs[0].device  
        
        content_share_c1_concat = torch.cat(content_s1_list, dim=1)
        content_share_c1_attmap = self.attm1(content_share_c1_concat)
        content_share_c1_attmap = self.sigmoid(content_share_c1_attmap)
        content_share_c1 = torch.cat([s1_enh * content_share_c1_attmap[:, idx, :, :].unsqueeze(1)
                                    for idx, s1_enh in enumerate(content_s1_list)], dim=1)
        content_share_c1 = self.fusion1(content_share_c1)

        content_share_c2_concat = torch.cat(content_s2_list, dim=1)
        content_share_c2_attmap = self.attm2(content_share_c2_concat)
        content_share_c2_attmap = self.sigmoid(content_share_c2_attmap)
        content_share_c2 = torch.cat([s2_enh * content_share_c2_attmap[:, idx, :, :].unsqueeze(1)
                                    for idx, s2_enh in enumerate(content_s2_list)], dim=1)
        content_share_c2 = self.fusion2(content_share_c2)

        content_share_c3_concat = torch.cat(content_s3_list, dim=1)
        content_share_c3_attmap = self.attm3(content_share_c3_concat)
        content_share_c3_attmap = self.sigmoid(content_share_c3_attmap)
        content_share_c3 = torch.cat([s3 * content_share_c3_attmap[:, idx, :, :].unsqueeze(1) for idx, s3 in enumerate(content_s3_list)], dim=1)
        content_share_c3 = self.fusion3(content_share_c3)

        content_share_c4_concat = torch.cat(content_s4_list, dim=1)
        content_share_c4_attmap = self.attm4(content_share_c4_concat)
        content_share_c4_attmap = self.sigmoid(content_share_c4_attmap)
        content_share_c4 = torch.cat([s4 * content_share_c4_attmap[:, idx, :, :].unsqueeze(1) for idx, s4 in enumerate(content_s4_list)], dim=1)
        content_share_c4 = self.fusion4(content_share_c4)
    
        fused_multiscale = self.ms_fusion(
            content_share_c1,
            content_share_c2,
            content_share_c3,
            content_share_c4
        )

        reconstruct_image_list, mu_list, sigma_list = zip(
                *[self.image_decoder[idx](style_list[idx], fused_multiscale) for idx in range(self.modal_num)]
            )
        
        mask_de_input = {
                'e1_out': content_share_c1,
                'e2_out': content_share_c2,
                'e3_out': content_share_c3,
                'e4_out': content_share_c4,
            }
        
        seg = self.mask_decoder(mask_de_input)
        content_c4_all_modal = torch.stack(
            [content_list[m]['s4'] for m in range(self.modal_num)], dim=1  # (B, M, C4, H/8, W/8)
        )
        return {
                'style': style_list,
                'content': content_list,
                'reconstruct_image': reconstruct_image_list,
                'mu': mu_list,
                'sigma': sigma_list,
                'content_c4': content_share_c4,
                'content_c3': content_share_c3,
                'content_c2': content_share_c2,
                'content_c1': content_share_c1,
                'seg': seg,
                'content_c4_modal': content_c4_all_modal, # per-modality version for contrastive loss
                'fused_multiscale':fused_multiscale
            }

class MultiScaleDynamicFusion(nn.Module):
    def __init__(self, c1, c2, c3, c4, heads=4):
        super().__init__()
        self.c4 = c4

    
        self.proj1 = nn.Conv2d(c1, c4, 1)
        self.proj2 = nn.Conv2d(c2, c4, 1)
        self.proj3 = nn.Conv2d(c3, c4, 1)

      
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
        c1 = F.interpolate(self.proj1(c1), size=size, mode='bilinear', align_corners=False)
        c2 = F.interpolate(self.proj2(c2), size=size, mode='bilinear', align_corners=False)
        c3 = F.interpolate(self.proj3(c3), size=size, mode='bilinear', align_corners=False)

      
        structure = c1 + c2 + c3

     
        x = c4.permute(0, 2, 3, 1)       # B,H,W,C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

    
        q = self.q(x)
        k = self.k(structure)
        v = self.v(structure)

      
        def reshape(x):
            return x.reshape(B, self.heads, C // self.heads, H * W)

        q, k, v = map(reshape, (q, k, v))

        attn = torch.softmax(
            torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale,
            dim=-1
        )

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)

    
        out = self.out(out)
        return c4 + 0.5 * out

class LearnableGaussianKernel(nn.Module):
    def __init__(self, kernel_size=3, init_sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.tensor(init_sigma))

        coords = torch.arange(kernel_size) - kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer("xx", xx.float())
        self.register_buffer("yy", yy.float())

    def forward(self):
        sigma = torch.clamp(self.sigma, min=1e-3)

        kernel = torch.exp(-(self.xx**2 + self.yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()   

        return kernel   # [k, k]
class DynamicGetGradient(nn.Module):

    def __init__(self, kernel_size=3, init_sigma=1.0):
        super().__init__()
        self.gaussian = LearnableGaussianKernel(
            kernel_size=kernel_size,
            init_sigma=init_sigma
        )

    def forward(self, x):
        B, C, H, W = x.shape

        kernel = self.gaussian()                  # [k, k]
        kernel = kernel.unsqueeze(0).unsqueeze(0) # [1,1,k,k]
        kernel = kernel.repeat(C, 1, 1, 1)        # depthwise

        smooth = F.conv2d(
            x, kernel,
            padding=kernel.shape[-1] // 2,
            groups=C
        )

        edge = x - smooth
        return edge

class EdgeEnhanceSingle(nn.Module):

    def __init__(self, nf, kernel_size=3, init_sigma=1.0):
        super().__init__()
        self.edge_extractor = DynamicGetGradient(
            kernel_size=kernel_size,
            init_sigma=init_sigma
        )
        self.conv = nn.Conv2d(nf * 2, nf, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.act = nn.LeakyReLU(0.05, inplace=True)   


    def forward(self, fea):
        edge = self.edge_extractor(fea)
        out = torch.cat([fea, edge], dim=1)
        out = self.act(self.conv(out))
        return out


