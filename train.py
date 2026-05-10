# imports
import torch
from dataloader import initDataLoader
from model import PetBreedClassifier

# constants
EPOCHS = 30
SEED = 42

# set random seed for reproducibility
torch.manual_seed(SEED)

# select compute device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"using device: {device}")

# load dataset
dataLoader = initDataLoader("train")

# load model
model = PetBreedClassifier().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model training
for epoch in range(EPOCHS):
    model.train()

    for images, labels in dataLoader:
        # move data to compute device
        images = images.to(device)
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
