import torch
import torch.nn as nn
import torch.nn.functional as F
from models.yolo.attention import EMAModule
from models.yolo.fasternet import FasterNetBlock


class C2fEMFast(nn.Module):
    """
    Improved C2f module with EMA integration and FasterNet for faster and lighter processing.
    More efficient C2f component
    for YOLOv8x with enhanced normalization and activation layers.
    """

    def __init__(self, c1, c2, n=1, shortcut=True):
        super(C2fEMFast, self).__init__()
        self.c = c2 // 2  # Half the number of output channels
        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, bias=False)
        self.cv2 = nn.Conv2d(2 * self.c, c2, kernel_size=1, bias=False)

        # Create a list of bottlenecks with FasterNet followed by EMA attention
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                # Add FasterNet block
                FasterNetBlock(self.c, self.c, ratio=0.5),
                EMAModule(self.c)
            ) for _ in range(n)
        ])

        self.shortcut = shortcut and c1 == c2
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply initial 1x1 convolution
        y = self.cv1(x)

        # Split the tensor in half along the channel dimension
        a, b = torch.split(y, self.c, dim=1)

        # Process through bottlenecks with residual connections
        for bottleneck in self.bottlenecks:
            a = a + bottleneck(a)

        # Concatenate processed features with the second half
        y = torch.cat((a, b), dim=1)

        # Apply final 1x1 convolution
        y = self.cv2(y)

        # Apply batch normalization and ReLU
        y = self.bn(y)
        y = self.relu(y)

        # Add residual connection if applicable
        if self.shortcut:
            y = y + x

        return y


class C2fTranslearn(nn.Module):
    """
    C2f module for transfer learning, used in the neck section when 
    C2f-EM-Fast is used in the backbone or vice versa.
    """

    def __init__(self, c1, c2, n=1, shortcut=True):
        super(C2fTranslearn, self).__init__()
        self.c = c2 // 2  # Half the number of output channels
        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, bias=False)
        self.cv2 = nn.Conv2d(2 * self.c, c2, kernel_size=1, bias=False)

        # Create a list of standard bottlenecks
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.c, self.c, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(self.c),
                nn.ReLU(inplace=True)
            ) for _ in range(n)
        ])

        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        # Apply initial 1x1 convolution
        y = self.cv1(x)

        # Split the tensor in half along the channel dimension
        a, b = torch.split(y, self.c, dim=1)

        # Process through bottlenecks with residual connections
        for bottleneck in self.bottlenecks:
            a = a + bottleneck(a)

        # Concatenate processed features with the second half
        y = torch.cat((a, b), dim=1)

        # Apply final 1x1 convolution
        y = self.cv2(y)

        # Add residual connection if applicable
        if self.shortcut:
            y = y + x

        return y
