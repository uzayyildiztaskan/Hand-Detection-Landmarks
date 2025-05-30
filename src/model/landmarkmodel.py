import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class HandLandmarkModel(nn.Module):
    def __init__(self, num_keypoints=21, grid_size=32):
        super(HandLandmarkModel, self).__init__()

        self.num_keypoints = num_keypoints
        self.grid_size = grid_size
        self.num_anchors = 2
        self.features_per_anchor = 1 + 4 + (3 * num_keypoints)
        self.output_channels = self.features_per_anchor * 2

        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
            nn.MaxPool2d(2),  # 256 -> 128
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),  # 128 -> 64
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2),  # 64 -> 32
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2)
        )

        self.prediction_head = nn.Conv2d(256, self.output_channels, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.backbone(x)                          # Shape: [B, 256, S, S]
        x = self.prediction_head(x)                   # Shape: [B, output_channels, S, S]
        x = x.permute(0, 2, 3, 1)                     # Shape: [B, S, S, output_channels]

        x = x.view(batch_size, x.size(1), x.size(2), self.num_anchors, self.features_per_anchor)          # Shape: [B, S, S, num_anchors, features_per_anchor]

        confidence = torch.sigmoid(x[..., 0:1])       # [B, S, S, num_anchors, 1]
        rest = x[..., 1:]                             # [B, S, S, num_anchors, features_per_anchor - 1]

        x = torch.cat([confidence, rest], dim=-1)     # [B, S, S, num_anchors, features_per_anchor]

        return x