# imports
import torch
from model import PetBreedClassifier
from dataloader import initDataLoader
from eval import evaluateModel

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
accuracy, loss = evaluateModel(model, dataLoader, torch.nn.CrossEntropyLoss(), device)
print(f"test accuracy: {accuracy:.2f}% | test loss: {loss:.4f}")