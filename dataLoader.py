import torch
from torchvision import datasets, transforms

def initDataLoader(mode):
    dataTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    data = datasets.OxfordIIITPet(
        root="data",
        split="trainval" if mode == "train" else "test",
        target_types="category",
        download=True,
        transform=dataTransforms
    )
    dataLoader = torch.utils.data.DataLoader(
        data,
        batch_size=32,
        shuffle=True
    )

    return dataLoader