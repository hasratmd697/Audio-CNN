import torch
import torch.nn as nn

# Define a residual block (similar to ResNet blocks)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, fmap_dict=None, prefix=""):
        # Forward pass through conv1 -> bn -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        # Forward through conv2 -> bn
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply shortcut connection
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut  # Residual connection

        # Store feature map before ReLU if needed
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add

        # Final activation
        out = torch.relu(out_add)

        # Store feature map after ReLU if needed
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out

        return out


# Define the full CNN architecture using Residual blocks
class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet-style block groups
        # Each layer contains multiple residual blocks
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for _ in range(3)])
        self.layer2 = nn.ModuleList([
            ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1)
            for i in range(4)
        ])
        self.layer3 = nn.ModuleList([
            ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1)
            for i in range(6)
        ])
        self.layer4 = nn.ModuleList([
            ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1)
            for i in range(3)
        ])

        # Global average pooling, dropout, and final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feature_maps=False):
        # If feature maps are not requested (standard forward)
        if not return_feature_maps:
            x = self.conv1(x)
            for block in self.layer1:
                x = block(x)
            for block in self.layer2:
                x = block(x)
            for block in self.layer3:
                x = block(x)
            for block in self.layer4:
                x = block(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.dropout(x)
            x = self.fc(x)
            return x

        # If feature maps are requested (for visualization/debugging)
        else:
            feature_maps = {}

            # Initial conv layer
            x = self.conv1(x)
            feature_maps["conv1"] = x

            # Residual blocks with feature map tracking
            for i, block in enumerate(self.layer1):
                x = block(x, feature_maps, prefix=f"layer1.block{i}")
            feature_maps["layer1"] = x

            for i, block in enumerate(self.layer2):
                x = block(x, feature_maps, prefix=f"layer2.block{i}")
            feature_maps["layer2"] = x

            for i, block in enumerate(self.layer3):
                x = block(x, feature_maps, prefix=f"layer3.block{i}")
            feature_maps["layer3"] = x

            for i, block in enumerate(self.layer4):
                x = block(x, feature_maps, prefix=f"layer4.block{i}")
            feature_maps["layer4"] = x

            # Pooling and classification
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)

            # Return both final output and intermediate feature maps
            return x, feature_maps
