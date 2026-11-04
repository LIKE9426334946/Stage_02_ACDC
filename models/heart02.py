import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralBranch(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x


class FusionBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        return x


class FusionMiddle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x


class FusionBottom(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, int(out_channels / 2), 3, padding=1)
        self.bn3 = nn.BatchNorm2d(int(out_channels / 2))

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        return x


class FusionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x


class HeartNetwork_02(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.left = nn.Conv2d(in_channels, 64, 7, padding=3, stride=2)
        self.right = nn.Conv2d(in_channels, 64, 7, padding=3, stride=2)

        self.left_bn = nn.BatchNorm2d(64)
        self.right_bn = nn.BatchNorm2d(64)

        # 第一阶段
        self.left_branch_1 = LateralBranch(64, 64)
        self.right_branch_1 = LateralBranch(64, 64)
        self.center_branch_1 = FusionMiddle(128, 64)
        self.left_fusion_1 = FusionBranch(64, 128)
        self.right_fusion_1 = FusionBranch(64, 128)

        # 第二阶段
        self.left_branch_2 = LateralBranch(128, 128)
        self.right_branch_2 = LateralBranch(128, 128)
        self.center_branch_2 = FusionMiddle(256, 128)
        self.left_fusion_2 = FusionBranch(128, 256)
        self.right_fusion_2 = FusionBranch(128, 256)

        # 第三阶段
        self.left_branch_3 = LateralBranch(256, 256)
        self.right_branch_3 = LateralBranch(256, 256)
        self.center_branch_3 = FusionMiddle(512, 256)
        self.left_fusion_3 = FusionBranch(256, 512)
        self.right_fusion_3 = FusionBranch(256, 512)

        # 第四阶段
        self.left_branch_4 = LateralBranch(512, 512)
        self.right_branch_4 = LateralBranch(512, 512)
        self.center_branch_4 = FusionMiddle(1024, 512)

        # bottom
        self.bottom = FusionBottom(512, 1024)

        # up
        self.up_fusion_1 = FusionUp(512, 256)
        self.up_fusion_2 = FusionUp(256, 128)
        self.up_fusion_3 = FusionUp(128, 64)
        self.up_fusion_4 = FusionUp(64, 32)

        self.head = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, x):
        left_1 = F.relu(self.left_bn(self.left(x)))
        right_1 = F.relu(self.right_bn(self.right(x)))

        # ----------------
        center_1 = torch.cat([left_1, right_1], dim=1)
        center_feature_1 = self.center_branch_1(center_1)
        left_11 = self.left_branch_1(left_1) + left_1 + center_feature_1

        right_11 = self.right_branch_1(right_1) + right_1 + center_feature_1
        left_2 = self.left_fusion_1(left_11)
        right_2 = self.right_fusion_1(right_11)

        # ----------------
        center_2 = torch.cat([left_2, right_2], dim=1)
        center_feature_2 = self.center_branch_2(center_2)
        left_22 = self.left_branch_2(left_2) + left_2 + center_feature_2

        right_22 = self.right_branch_2(right_2) + right_2 + center_feature_2
        left_3 = self.left_fusion_2(left_22)
        right_3 = self.right_fusion_2(right_22)

        # ----------------
        center_3 = torch.cat([left_3, right_3], dim=1)
        center_feature_3 = self.center_branch_3(center_3)
        left_33 = self.left_branch_3(left_3) + left_3 + center_feature_3
        right_33 = self.right_branch_3(right_3) + right_3 + center_feature_3

        left_4 = self.left_fusion_3(left_33)
        right_4 = self.right_fusion_3(right_33)

        # ----------------
        center_4 = torch.cat([left_4, right_4], dim=1)
        bottom_1 = (
            self.left_branch_4(left_4)
            + left_4
            + self.center_branch_4(center_4)
            + self.right_branch_4(right_4)
            + right_4
        )

        bottom_11 = self.bottom(bottom_1)

        up_1 = left_4 + right_4 + bottom_11
        up_11 = self.up_fusion_1(up_1)

        up_2 = left_3 + right_3 + up_11
        up_22 = self.up_fusion_2(up_2)

        up_3 = left_2 + right_2 + up_22
        up_33 = self.up_fusion_3(up_3)

        up_4 = left_1 + right_1 + up_33
        up_44 = self.up_fusion_4(up_4)

        out = self.head(up_44)
        return out


if __name__ == "__main__":
    model = HeartNetwork_02(1, 4)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")  # 38.16M
    x = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        model(x)
        torch.onnx.export(model, (x,), "onnx/01.onnx", external_data=False)
