from torch import nn

class PetBreedClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 112 * 112, 37)

    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.pool(inputs)
        inputs = self.relu(inputs)

        inputs = inputs.view(-1, 32 * 112 * 112)
        outputs = self.fc1(inputs)

        return outputs