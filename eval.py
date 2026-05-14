#imports
import torch

# accuracy calculation function
def evaluateModel(model, dataloader, criterion, device):
    model.eval()

    correct = 0
    total = 0
    loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss += criterion(outputs, labels).item()
    
    averageLoss = loss / len(dataloader)
    accuracy = 100 *(correct / total)

    return accuracy, averageLoss