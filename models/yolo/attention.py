import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAModule(nn.Module):
    """
    Efficient Multi-scale Attention (EMA) module.
    Uses parallel processing to capture channel-specific features through
    convolutional operations while maintaining original dimensionality.
    """

    def __init__(self, channels):
        super(EMAModule, self).__init__()

        # 3x3 convolution branch
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Final fusion 1x1 convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Process through 3x3 branch
        branch3x3 = self.conv3x3(x)

        # Process through 1x1 branch
        branch1x1 = self.conv1x1(x)

        # Concatenate both branches
        concat_features = torch.cat([branch3x3, branch1x1], dim=1)

        # Fuse features with 1x1 convolution
        output = self.fusion(concat_features)

        # Add residual connection
        return output + x
