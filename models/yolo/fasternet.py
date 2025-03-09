import torch
import torch.nn as nn
import torch.nn.functional as F


class PConv(nn.Module):
    """
    Partial Convolution layer for our FasterNet architecture.
    Exploits the similarity between different channels by only applying
    convolution to a subset of input channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=0.5):
        super(PConv, self).__init__()
        self.ratio = ratio
        self.part_channels = int(in_channels * ratio)
        self.conv = nn.Conv2d(self.part_channels, self.part_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Split channels into two parts
        part1, part2 = torch.split(
            x, [self.part_channels, x.size(1) - self.part_channels], dim=1)

        # Apply convolution only to the first part
        part1 = self.conv(part1)

        # Concatenate both parts again
        x = torch.cat([part1, part2], dim=1)

        # Apply batch normalization and ReLU
        x = self.bn(x)
        x = self.relu(x)
        return x


class PWConv(nn.Module):
    """
    Point-Wise Convolution (1x1 convolution) with batch normalization and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super(PWConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FasterNetBlock(nn.Module):
    """
    FasterNet block consisting of one PConv layer and two PWConv layers 
    as described in the paper.
    """

    def __init__(self, in_channels, out_channels, ratio=0.5):
        super(FasterNetBlock, self).__init__()
        self.pconv = PConv(in_channels, in_channels, ratio=ratio)
        self.pw_conv1 = PWConv(in_channels, out_channels)
        self.pw_conv2 = PWConv(out_channels, out_channels)

    def forward(self, x):
        x = self.pconv(x)
        x = self.pw_conv1(x)
        x = self.pw_conv2(x)
        return x


class FasterNet(nn.Module):
    """
    FasterNet backbone implementation integrating with YOLOv8.
    """

    def __init__(self, in_channels=3, base_channels=64, num_blocks=[2, 2, 6, 2], ratio=0.5):
        super(FasterNet, self).__init__()

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Build FasterNet backbone
        self.stages = nn.ModuleList()
        in_ch = base_channels

        for i, num_block in enumerate(num_blocks):
            stage = []
            out_ch = base_channels * (2 ** i)

            # Downsample at the beginning of each stage except the first one
            if i > 0:
                stage.append(nn.Conv2d(in_ch, out_ch, kernel_size=3,
                             stride=2, padding=1, bias=False))
                stage.append(nn.BatchNorm2d(out_ch))
                stage.append(nn.ReLU(inplace=True))
                in_ch = out_ch

            # Add FasterNet blocks
            for _ in range(num_block):
                stage.append(FasterNetBlock(in_ch, out_ch, ratio=ratio))
                in_ch = out_ch

            self.stages.append(nn.Sequential(*stage))

        self.output_channels = [base_channels *
                                (2 ** i) for i in range(len(num_blocks))]

    def forward(self, x):
        x = self.stem(x)
        features = []

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


def create_fasternet_backbone(in_channels=3, base_channels=64):
    """
    Create a FasterNet backbone suitable for integration with YOLOv8.

    Args:
        in_channels: Number of input channels (3 for RGB)
        base_channels: Base number of channels (scales up in deeper layers)

    Returns:
        FasterNet backbone model
    """
    return FasterNet(in_channels=in_channels, base_channels=base_channels)
