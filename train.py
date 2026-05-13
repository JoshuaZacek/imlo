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
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
lrScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=0.000001,
)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    lrScheduler.step()

    testAccuracy = calculateAccuracy(model, testDataLoader, device)
    trainAccuracy = calculateAccuracy(model, trainingEvalDataLoader, device)
    currentLearningRate = optimizer.param_groups[0]["lr"]

    print(f"epoch {epoch + 1}/{EPOCHS} | lr: {currentLearningRate:.6f} | test accuracy: {testAccuracy:.2f}% | training accuracy: {trainAccuracy:.2f}%")

# save trained model
torch.save(
    model.state_dict(),
    "model.pth"
)
print("training complete")
