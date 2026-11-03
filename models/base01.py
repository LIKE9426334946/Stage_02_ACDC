import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class BaseModel_01(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

    def forward(self, x):
        return self.unet(x)


if __name__ == "__main__":
    model = BaseModel_01(1, 4)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")  # 14.33M
    x = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        model(x)
        torch.onnx.export(model, (x,), "onnx/01.onnx", external_data=False)
