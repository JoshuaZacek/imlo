# imports
import torch
from model import PetBreedClassifier
from dataloader import initDataLoader

# select compute device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"using device: {device}")

# load dataset
dataLoader = initDataLoader("test")

# load model
model = PetBreedClassifier().to(device)
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