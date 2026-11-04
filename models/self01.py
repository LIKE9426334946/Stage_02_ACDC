# 自己写的UNet+ResNet18网络

import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        residual = self.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = identity + residual
        return self.relu(out)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SelfModel_01(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2), BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2), BasicBlock(512, 512)
        )

        self.up4 = DecoderBlock(512, 256, 256)
        self.up3 = DecoderBlock(256, 128, 128)
        self.up2 = DecoderBlock(128, 64, 64)
        self.up1 = DecoderBlock(64, 64, 32)
        self.refine = DoubleConv(32, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x_0):
        stem = self.stem(x_0)
        e1 = self.layer1(self.maxpool(stem))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        x = self.layer4(e3)

        x = self.up4(x, e3)
        x = self.up3(x, e2)
        x = self.up2(x, e1)
        x = self.up1(x, stem)
        x = F.interpolate(x, size=x_0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = SelfModel_01(3, 21)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")  # 38.16M
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        model(x)
        torch.onnx.export(model, (x,), "onnx/02.onnx", external_data=False)
