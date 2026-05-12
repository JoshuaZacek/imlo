from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, inChannels, outChannels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannels)

        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outChannels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs):
        residual = self.shortcut(inputs)

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)

        outputs += residual
        outputs = self.relu(outputs)

        return outputs

class PetBreedClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stages = nn.Sequential(
            self._createStage(64, 64, blocks=2, stride=1),
            self._createStage(64, 128, blocks=2, stride=2),
            self._createStage(128, 256, blocks=2, stride=2),
            self._createStage(256, 512, blocks=2, stride=2),
        )

        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 37),
        )

    def _createStage(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.head(inputs)
        outputs = self.stages(outputs)
        outputs = self.tail(outputs)
        return outputs