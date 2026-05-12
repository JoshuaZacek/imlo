import torch
from torchvision import datasets, transforms

def getTransforms(augment):
    # seperate transforms for training and testing datasets because of data augmentation for training dataset
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4783, 0.4459, 0.3957], std=[0.2601, 0.2548, 0.2627]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4783, 0.4459, 0.3957], std=[0.2601, 0.2548, 0.2627]),
        ])

def initDataLoader(mode, augment=False):
    dataTransforms = getTransforms(augment)

    data = datasets.OxfordIIITPet(
        root="data",
        split="trainval" if mode == "train" else "test",
        target_types="category",
        download=True,
        transform=dataTransforms
    )
    dataLoader = torch.utils.data.DataLoader(
        data,
        batch_size=64,
        shuffle=(True if mode == "train" else False),
        pin_memory=torch.cuda.is_available(),
        num_workers=2 if torch.cuda.is_available() else 0
    )

    return dataLoader