import torch
import torch.nn as nn
import torch.nn.functional as F

# I used this paper: https://paperswithcode.com/method/u-net
# This UNet is a good baseline model, and we should mess around with scaling it to 3d, using attention, using Pinn's, and using adaptive kernels


class DoubleConv(nn.Module):
    """
    3x3 Convolution -> Batch Normalization -> ReLU -> 3x3 Convolution -> Batch Normalization -> ReLU
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Max Pool 2x2 -> Double Convolution
    Dimensionality should half, while channels double
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Transpose Convolution + Concatenation -> Double Convolution
    Dimensionality should be doubled, and channels should half
    """

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    1x1 Convolution -> Output
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv3D(nn.Module):
    """
    3x3 Convolution -> Batch Normalization -> ReLU -> 3x3 Convolution -> Batch Normalization -> ReLU
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv_3d = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv_3d(x)

class Down3D(nn.Module):
    """
    Max Pool 2x2 -> Double Convolution
    Dimensionality should half, while channels double
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv_3d = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv_3d(x)

class Up3D(nn.Module):
    """
    Transpose Convolution + Concatenation -> Double Convolution
    Dimensionality should be doubled, and channels should half
    """

    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()

        if trilinear:
            self.up_3d = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv_3d = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up_3d = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv_3d = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_3d(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv_3d(x)

class OutConv3D(nn.Module):
    """
    1x1 Convolution -> Output
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv_3d = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv_3d(x)


class ResidualBlock(nn.Module):
    """
    Create residual connections between different blocks
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.double_conv(x) + self.residual(x)

class ResidualBlock3D(nn.Module):
    """3D Residual Block"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.double_conv = DoubleConv3D(in_channels, out_channels)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.double_conv(x) + self.residual(x)

class DeepSupervision(nn.Module):
    """
    helps intermediate outputs contribute to loss
    """
    def __init__(self, out_channels):
        super(DeepSupervision, self).__init__()
        self.out_conv = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)

class DeepSupervision3D(nn.Module):
    """
    helps intermediate outputs contribute to loss
    """
    def __init__(self, out_channels):
        super(DeepSupervision3D, self).__init__()
        self.out_conv = nn.LazyConv3d(out_channels, 1)

    def forward(self, x):
        return self.out_conv(x)

class AttentionBlock(nn.Module):
    """
    Standard attention block to help model find important parts of an image
    """

    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.gate = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta_x = self.W(x)
        phi_g = self.gate(g)
        f = self.relu(theta_x + phi_g)
        psi_f = self.sigmoid(self.psi(f))
        return x * psi_f

class AttentionBlock3D(nn.Module):
    """
    Standard 3Dattention block to help model find important parts of an image
    """

    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock3D, self).__init__()
        self.W = nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.gate = nn.Conv3d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta_x = self.W(x)
        phi_g = self.gate(g)
        f = self.relu(theta_x + phi_g)
        psi_f = self.sigmoid(self.psi(f))
        return x * psi_f
