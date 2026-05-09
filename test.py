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
    split="test",
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
model.load_state_dict(
    torch.load("model.pth", map_location=device)
)
model.eval()

# run model with test dataset
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataLoader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

 # calculate accuracy
accuracy = 100 * correct / total
print(f"test accuracy: {accuracy:.2f}%")