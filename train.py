# imports
import torch
from accuracy import calculateAccuracy
from dataloader import initDataLoader
from model import PetBreedClassifier

# constants
EPOCHS = 30
SEED = 42

# reproducibility
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

# select compute device (Apple silicon, Nvidia GPU, or CPU)
deviceType = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(deviceType)
print(f"using device: {device}")

# load datasets
trainingDataLoader = initDataLoader("train", augment=True)
testDataLoader = initDataLoader("test", augment=False)
trainingEvalDataLoader = initDataLoader("train", augment=False)

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
    trainAccuracy = calculateAccuracy(model, trainingEvalDataLoader, device)

    print(f"epoch {epoch + 1}/{EPOCHS} | test accuracy: {testAccuracy:.2f}% | training accuracy: {trainAccuracy:.2f}%")

# save trained model
torch.save(
    model.state_dict(),
    "model.pth"
)
print("training complete")
