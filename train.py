# imports
import torch
from accuracy import calculateAccuracy
from dataloader import initDataLoader
from model import PetBreedClassifier

# constants
EPOCHS = 10
SEED = 42

# set random seed for reproducibility
torch.manual_seed(SEED)

# select compute device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"using device: {device}")

# load dataset
trainingDataLoader = initDataLoader("train")
testDataLoader = initDataLoader("test")

# load model
model = PetBreedClassifier().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model training
for epoch in range(EPOCHS):
    model.train()

    for images, labels in trainingDataLoader:
        # move data to compute device
        images = images.to(device)
        labels = labels.to(device)

        # update model weights
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    testAccuracy = calculateAccuracy(model, testDataLoader, device)
    trainAccuracy = calculateAccuracy(model, trainingDataLoader, device)

    print(f"epoch {epoch + 1}/{EPOCHS} | test accuracy: {testAccuracy:.2f}% | training accuracy: {trainAccuracy:.2f}%")

# save trained model
torch.save(
    model.state_dict(),
    "model.pth"
)
print("training complete")
