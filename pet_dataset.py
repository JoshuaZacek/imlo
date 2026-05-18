import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import datasets, transforms
import torch

class PetDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, augment=False):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="category",
            download=True
        )
        self.root = root
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        imagePath = self.dataset._images[index]
        imageName = os.path.splitext(os.path.basename(imagePath))[0]

        trimapPath = os.path.join(self.root, "oxford-iiit-pet", "annotations", "trimaps", f"{imageName}.png")
        trimap = Image.open(trimapPath)
        trimap = np.array(trimap)

        # get trimap mask
        mask = (trimap != 3).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # data augmentation
        if self.augment:
            # Random resized crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.7, 1.0), ratio=(0.75, 1.33))
            image = TF.resized_crop(image, i, j, h, w, size=(224, 224), interpolation=InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, size=(224, 224), interpolation=InterpolationMode.NEAREST)

            # Random horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random rotation between -10 and 10 degrees
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

            # Color jitter
            image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)(image)
        else:
            image = TF.resize(image, (224, 224), interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (224, 224), interpolation=InterpolationMode.NEAREST)

        # normalize image
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.4783, 0.4459, 0.3957], std=[0.2601, 0.2548, 0.2627])
        
        # add mask as a 4th channel to the image tensor
        mask = (mask - 0.5) / 0.5
        image = torch.cat([image, mask], dim=0)

        return image, label