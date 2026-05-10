import torch
from torchvision import datasets, transforms

def initDataLoader(mode):
    dataTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4783, 0.4459, 0.3957], std=[0.2601, 0.2548, 0.2627]),
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