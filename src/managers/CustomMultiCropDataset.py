from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class CustomMultiCropDataset(Dataset):
    def __init__(self, dataset, scales, crop_size):
        self.dataset = dataset
        self.scales = scales
        self.crop_size = crop_size

        self.transformations = []
        for scale in scales:
            self.transformations.append(
                transforms.Compose([
                    transforms.Resize((scale, scale)),
                    transforms.FiveCrop(crop_size),  # Get top-left, top-right, bottom-left, bottom-right, and center crops
                    transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),  # Convert crops to tensor
                    transforms.Lambda(lambda crops: crops + [transforms.functional.hflip(crop) for crop in crops])  # Add flipped crops
                ])
            )

    def __len__(self):
        return len(self.dataset) * len(self.scales) * 10  # 5 crops + 5 flipped crops for each scale

    def __getitem__(self, idx):
        dataset_idx = idx // (len(self.scales) * 10)
        scale_idx = (idx % (len(self.scales) * 10)) // 10
        crop_idx = idx % 10

        item = self.dataset[dataset_idx]
        image = Image.fromarray(np.array(item['image'])).convert("RGB")
        label = item['label']

        transform = self.transformations[scale_idx]
        crops = transform(image)
        crop = crops[crop_idx]

        return crop, label