# 自己写的UNet+ResNet18网络

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, k_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        # 每个块独立学习一张空间权重图，两个维度分别对应高和宽。
        self.k_11 = nn.Parameter(torch.randn(k_size, k_size) * 0.01)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, x_0):
        identity = x
        residual = self.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        if self.downsample is not None:
            identity = self.downsample(x)
        size = residual.shape[-2:]
        x_small = F.interpolate(x_0, size=size, mode="bilinear", align_corners=False)
        x_small = x_small.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        k = self.k_11
        if k.shape != size:
            # 输入分辨率变化时，插值后的权重仍能把梯度传回原参数。
            k = F.interpolate(k[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
        input_branch = (x_small * k).to(dtype=residual.dtype)

        out = identity + residual + input_branch  # (B, 1, H, W) 广播到每个特征通道
        return self.relu(out)


class ResidualStage(nn.Sequential):
    def __init__(self, in_channels, out_channels, k_size, stride=1):
        super().__init__(
            BasicBlock(in_channels, out_channels, stride=stride, k_size=k_size),
            BasicBlock(out_channels, out_channels, k_size=k_size),
        )

    def forward(self, x, x_0):
        for block in self:
            x = block(x, x_0)
        return x


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


class SelfModel_04(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 以 256×256 输入为基准，每个阶段的两个块使用相同大小、各自独立的权重。
        self.layer1 = ResidualStage(64, 64, k_size=64)
        self.layer2 = ResidualStage(64, 128, k_size=32, stride=2)
        self.layer3 = ResidualStage(128, 256, k_size=16, stride=2)
        self.layer4 = ResidualStage(256, 512, k_size=8, stride=2)

        self.up4 = DecoderBlock(512, 256, 256)
        self.up3 = DecoderBlock(256, 128, 128)
        self.up2 = DecoderBlock(128, 64, 64)
        self.up1 = DecoderBlock(64, 64, 32)
        self.refine = DoubleConv(32, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x_0):
        stem = self.stem(x_0)
        e1 = self.layer1(self.maxpool(stem), x_0)
        e2 = self.layer2(e1, x_0)
        e3 = self.layer3(e2, x_0)
        x = self.layer4(e3, x_0)

        x = self.up4(x, e3)
        x = self.up3(x, e2)
        x = self.up2(x, e1)
        x = self.up1(x, stem)
        x = F.interpolate(x, size=x_0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = SelfModel_04(3, 21).eval()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        y = model(x)
        print("输出形状", tuple(y.shape))

    try:
        import onnx  # 仅导出时需要

        onnx_path = Path("onnx/02.onnx")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model, (x,), str(onnx_path), external_data=False)
        print("已导出", onnx_path)
    except ModuleNotFoundError as exc:
        if exc.name not in {"onnx", "onnxscript"}:
            raise
        print("如需导出 ONNX，请运行 python -m pip install onnx onnxscript 后重试。")
