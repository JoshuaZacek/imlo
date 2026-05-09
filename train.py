# imports
import torch
from torchvision import datasets, transforms
from model import NeuralNet

# constants
EPOCHS = 10

# select compute device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"using device: {device}")

# load dataset
dataTransformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
data = datasets.OxfordIIITPet(
    root="data",
    split="trainval",
    target_types="category",
    download=True,
    transform=dataTransformations
)
dataLoader = torch.utils.data.DataLoader(
    data,
    batch_size=32,
    shuffle=True
)

# load model
model = NeuralNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model training
for epoch in range(EPOCHS):
    model.train()

    for inputs, labels in dataLoader:
        # move data to compute device
        images = inputs.to(device)
        labels = labels.to(device)

        # update model weights
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"{epoch + 1}/{EPOCHS}")

# save trained model
torch.save(
    model.state_dict(),
    "model.pth"
)
print("training complete")
