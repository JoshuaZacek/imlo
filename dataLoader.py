import torch
from pet_dataset import PetDataset

def initDataLoader(mode, augment=False):
    data = PetDataset(
        root="data",
        split="trainval" if mode == "train" else "test",
        augment=augment
    )

    dataLoader = torch.utils.data.DataLoader(
        data,
        batch_size=64,
        shuffle=(mode == "train"),
        pin_memory=torch.cuda.is_available(),
        num_workers=8 if torch.cuda.is_available() else 0
    )

    return dataLoader