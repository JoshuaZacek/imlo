from torch import nn

class PetBreedClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(32 * 112 * 112, 37)

    def forward(self, inputs):
        inputs = self.features(inputs)

        inputs = inputs.view(inputs.size(0), -1)
        outputs = self.fc1(inputs)

        return outputs