# Ultralytics YOLO, AGPL-3.0 license
"""Custom modules used by the YOLO11 DWR-EIEStem-CARAFE-P2 ablation study."""

import numpy as np
import torch
import torch.nn as nn

from .block import C3k, C3k2
from .conv import Conv

__all__ = ("CARAFE", "C3k2_DWR", "DWR", "EIEStem", "SobelConv")


class DWR(nn.Module):
    """Dilation-wise residual block."""

    def __init__(self, dim):
        super().__init__()
        self.conv_3x3 = Conv(dim, dim // 2, 3)
        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)
        self.conv_1x1 = Conv(dim * 2, dim, k=1)

    def forward(self, x):
        """Apply parallel dilated convolutions and a residual connection."""
        x_reduced = self.conv_3x3(x)
        x1 = self.conv_3x3_d1(x_reduced)
        x2 = self.conv_3x3_d3(x_reduced)
        x3 = self.conv_3x3_d5(x_reduced)
        return self.conv_1x1(torch.cat((x1, x2, x3), dim=1)) + x


class C3k_DWR(C3k):
    """C3k block whose bottlenecks are replaced by DWR blocks."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        hidden_channels = int(c2 * e)
        self.m = nn.Sequential(*(DWR(hidden_channels) for _ in range(n)))


class C3k2_DWR(C3k2):
    """YOLO11 C3k2 block using DWR blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_DWR(self.c, self.c, 2, shortcut, g) if c3k else DWR(self.c) for _ in range(n)
        )


class SobelConv(nn.Module):
    """Channel-wise Sobel filtering used by EIEStem."""

    def __init__(self, channel):
        super().__init__()
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(
            channel, channel, kernel_size=3, padding=1, groups=channel, bias=False
        )
        self.sobel_kernel_y_conv3d = nn.Conv3d(
            channel, channel, kernel_size=3, padding=1, groups=channel, bias=False
        )
        self.sobel_kernel_x_conv3d.weight.data = kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = kernel_y.clone()

        # Keep the original experimental behavior for checkpoint compatibility.
        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        """Apply horizontal and vertical Sobel filters."""
        x_3d = x[:, :, None, :, :]
        return (self.sobel_kernel_x_conv3d(x_3d) + self.sobel_kernel_y_conv3d(x_3d))[:, :, 0]


class EIEStem(nn.Module):
    """Edge-information-enhancement stem with Sobel and max-pooling branches."""

    def __init__(self, inc, hidc, ouc):
        super().__init__()
        self.conv1 = Conv(inc, hidc, 3, 2)
        self.sobel_branch = SobelConv(hidc)
        self.pool_branch = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True),
        )
        self.conv2 = Conv(hidc * 2, hidc, 3, 2)
        self.conv3 = Conv(hidc, ouc, 1)

    def forward(self, x):
        """Extract and fuse edge and pooled appearance features."""
        x = self.conv1(x)
        x = torch.cat((self.sobel_branch(x), self.pool_branch(x)), dim=1)
        return self.conv3(self.conv2(x))


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures upsampling."""

    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        super().__init__()
        self.scale = scale
        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)
        self.upsmp = nn.Upsample(scale_factor=scale, mode="nearest")
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=k_up // 2 * scale)

    def forward(self, x):
        """Upsample a feature map using content-aware reassembly kernels."""
        batch, channels, height, width = x.size()
        out_height, out_width = height * self.scale, width * self.scale

        weights = self.comp(x)
        weights = self.enc(weights)
        weights = self.pix_shf(weights)
        weights = torch.softmax(weights, dim=1)

        x = self.upsmp(x)
        x = self.unfold(x)
        x = x.view(batch, channels, -1, out_height, out_width)
        return torch.einsum("bkhw,bckhw->bchw", weights, x)
