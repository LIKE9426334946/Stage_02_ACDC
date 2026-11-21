import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = self.mlp(self.avg_pool(x))
        max_ = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg + max_)

        return x * attention


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        feature = torch.cat([avg, max_], dim=1)

        attention = self.sigmoid(self.conv(feature))

        return x * attention


class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction)

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x


import segmentation_models_pytorch as smp


class UNetResNet18CBAM(nn.Module):

    def __init__(self, in_channels=3, classes=1, encoder_weights="imagenet"):
        super().__init__()

        # 原始 U-Net
        base = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

        self.encoder = base.encoder
        self.decoder = base.decoder
        self.segmentation_head = base.segmentation_head

        # 每个 Encoder 阶段配置一个 CBAM
        channels = self.encoder.out_channels[1:]

        self.attentions = nn.ModuleList([CBAM(c) for c in channels])

    def forward(self, x):

        # Encoder
        features = self.encoder(x)

        # 注意力增强
        features = [features[0]] + [attention(feature) for attention, feature in zip(self.attentions, features[1:])]

        # Decoder
        decoder_output = self.decoder(features)

        # 分割头
        output = self.segmentation_head(decoder_output)

        return output


if __name__ == "__main__":
    model = UNetResNet18CBAM(in_channels=3, classes=1)

    x = torch.randn(2, 3, 256, 256)

    y = model(x)

    print(y.shape)
    with torch.no_grad():
        model(x)
        torch.onnx.export(model, (x,), "onnx/03.onnx", external_data=False)
