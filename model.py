import torch
import torch.nn as nn

from hadamard import fwht, ifwht
from quaternion import QuaternionConv

class SoftThresholding(nn.Module):
    def __init__(self):
        super(SoftThresholding, self).__init__()
        self.T = None

    def forward(self, x):
        if self.T is None:
            self.T = nn.Parameter(torch.rand(x.shape[-2:]) / 10, requires_grad=True)
        return torch.sign(x) * torch.maximum(torch.abs(x) - self.T.to(x.device), torch.tensor(0.0, device=x.device))

class PolynomialThresholding(nn.Module):
    def __init__(self, in_channels, a_poly=[0.707,	0.014,	-0.008,	0.999,	0.940]):
        super().__init__()
        self.N = len(a_poly)
        self.a_poly = torch.tensor(a_poly, dtype=torch.float32)
        self.delta = nn.Parameter(torch.rand(1, in_channels, 1) / 10)

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.reshape(B, C, -1)
        delta_expanded = self.delta.view(1, -1, 1).to(x.device)

        abs_x = torch.abs(x_reshaped)
        sign_x = torch.sign(x_reshaped)

        # Using continuous activation for better gradient flow
        condition = torch.sigmoid(abs_x - delta_expanded)

        polynomial_terms = [x_reshaped ** (2 * k + 1) for k in range(self.N - 2)]
        polynomial_terms = torch.stack(polynomial_terms, dim=-1)  # Stack to create additional dimension

        f_x = torch.zeros((B, C, H * W, self.N), device=x.device)
        f_x[..., :-2] = polynomial_terms * (1 - condition.unsqueeze(-1))
        f_x[..., -2] = x_reshaped * condition
        f_x[..., -1] = -delta_expanded * sign_x * condition

        new_coeffs = torch.matmul(f_x, self.a_poly.to(x.device))
        return new_coeffs.view(B, C, H, W)


class WHTConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pods=1, residual=True, poly=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pods = pods
        self.conv = torch.nn.ModuleList([QuaternionConv(in_channels, out_channels, kernel_size=1, stride=1, bias=False) for _ in range(self.pods)])
        if poly:
            self.ST = torch.nn.ModuleList([PolynomialThresholding(out_channels) for _ in range(self.pods)])
        else:
            self.ST = torch.nn.ModuleList([SoftThresholding() for _ in range(self.pods)])
        self.residual = residual

    def forward(self, x):
        height, width = x.shape[-2:]
        height_pad = find_min_power(height)
        width_pad = find_min_power(width)

        f0 = x
        if width_pad > width or height_pad > height:
            f0 = torch.nn.functional.pad(f0, (0, width_pad - width, 0, height_pad - height))

        f1 = fwht(f0, axis=-1)
        f2 = fwht(f1, axis=-2)

        f3 = [f2 for _ in range(self.pods)]
        f4 = [self.conv[i](f3[i]) for i in range(self.pods)]
        f5 = [self.ST[i](f4[i]) for i in range(self.pods)]

        f6 = torch.stack(f5, dim=-1).sum(dim=-1)

        f7 = ifwht(f6, axis=-1)
        f8 = ifwht(f7, axis=-2)

        y = f8[..., :height, :width]

        if self.residual:
            y = y + x
        return y

class QuaternionChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super(QuaternionChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # compressed feature
        self.conv1 = QuaternionConv(channels, channels // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True) # squeezed quaternion weights
        self.conv2 = QuaternionConv(channels // reduction_ratio, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.conv1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.conv2(avg_out)
        return self.sigmoid(avg_out)

class QuaternionSpatialAttention(nn.Module):
    def __init__(self, channels):
        super(QuaternionSpatialAttention, self).__init__()
        self.conv1 = QuaternionConv(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = QuaternionConv(channels, channels // 8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = QuaternionConv(channels // 8, channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return self.sigmoid(x)

class QuaternionAttentionBlock(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super(QuaternionAttentionBlock, self).__init__()
        self.channel_attention = QuaternionChannelAttention(channels, reduction_ratio)
        self.spatial_attention = QuaternionSpatialAttention(channels)

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        # this is an important thing!!!! maybe I could replace it with quaternion multiplication
        return torch.mul((1 - sa), ca) + torch.mul(sa, x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, bias=False):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.projection(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.residual_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            QuaternionConv(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            WHTConv2D(channels, channels, pods=1, residual=False, poly=True),
            nn.BatchNorm2d(channels)
        )
        self.shortcut = nn.Sequential(
            QuaternionConv(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.attention = nn.Sequential(
            QuaternionConv(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = x.clone()
        out = self.residual_layers(x) + self.shortcut(x)
        return self.attention(out) + shortcut

class Downsample(nn.Module):
    def __init__(self, channels):
        super(Downsample, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2)
        )

    def forward(self, x):
        return self.layers(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.scale_factor = 2
        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2)
        )

    def forward(self, x):
        #temp = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * self.scale_factor, x.shape[3] * self.scale_factor)).cuda()
        #output = [F.interpolate(x[i], scale_factor=self.scale_factor, mode='bilinear') for i in range(x.shape[0])]
        #out = torch.stack(output, dim=0)
        out = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.layers(out)

class QHNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=24, enc_blocks=[4, 4, 6, 6], dec_blocks=[4, 4, 6, 6], bias=False):
        super(QHNet, self).__init__()

        self.patch_embedding = PatchEmbedding(in_channels=in_channels, out_channels=base_channels)
        self.encoder1 = nn.Sequential(*[ResBlock(channels=int(base_channels * 1)) for _ in range(enc_blocks[0])])

        self.downsample1 = Downsample(base_channels)  # From Level 1 to Level 2
        self.encoder2 = nn.Sequential(*[ResBlock(channels=int(base_channels * 2 ** 1)) for _ in range(enc_blocks[1])])

        self.downsample2 = Downsample(int(base_channels * 2 ** 1))  # From Level 2 to Level 3
        self.encoder3 = nn.Sequential(*[ResBlock(channels=int(base_channels * 2 ** 2)) for _ in range(enc_blocks[2])])

        self.decoder3 = nn.Sequential(*[ResBlock(channels=int(base_channels * 2 ** 2)) for _ in range(dec_blocks[2])])

        self.upsample2 = Upsample(int(base_channels * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_channels2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(int(base_channels * 2 ** 2), int(base_channels * 2 ** 1), kernel_size=1, bias=bias),
            nn.BatchNorm2d(int(base_channels * 2 ** 1))
        )

        self.decoder2 = nn.Sequential(*[ResBlock(channels=int(base_channels * 2 ** 1)) for _ in range(dec_blocks[1])])

        self.upsample1 = Upsample(int(base_channels * 2 ** 1))  # From Level 2 to Level 1
        self.decoder1 = nn.Sequential(*[ResBlock(channels=int(base_channels * 2 ** 1)) for _ in range(dec_blocks[0])])

        self.refinement = QuaternionAttentionBlock(channels=int(base_channels * 2 ** 1), reduction_ratio=8)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=int(base_channels * 2 ** 1), out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        shortcut = x.clone()
        enc1 = self.patch_embedding(x)
        enc1_out = self.encoder1(enc1)

        enc2 = self.downsample1(enc1_out)
        enc2_out = self.encoder2(enc2)

        enc3 = self.downsample2(enc2_out)
        enc3_out = self.encoder3(enc3)

        dec3_out = self.decoder3(enc3_out)

        dec2 = self.upsample2(dec3_out)
        dec2 = torch.cat([dec2, enc2_out], dim=1)
        dec2 = self.reduce_channels2(dec2)
        dec2_out = self.decoder2(dec2)

        dec1 = self.upsample1(dec2_out)
        dec1 = torch.cat([dec1, enc1_out], dim=1)
        dec1_out = self.decoder1(dec1)

        refined = self.refinement(dec1_out)
        output = self.output(refined) + shortcut
        return output

