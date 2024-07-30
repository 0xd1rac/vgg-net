import torch
import src.managers as managers
import src.model_components as model_components
import torch.nn as nn
import json
from src.common_imports import *
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from MultiCropEvalStats import MultiCropEvalStats

class CustomTransformDataset(Dataset):
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


def eval(s_min,
         s_max,
        test_scales,
         models_lis,
         device, 
         batch_size,
         num_workers,
        ):
    dataset = load_dataset("zh-plus/tiny-imagenet")
    test_ds = dataset["valid"]
    test_ds = test_ds.select(range(80))
    custom_test_ds = CustomTransformDataset(test_ds, test_scales, IMG_DIM)
    test_dl= DataLoader(custom_test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    stats_lis = list()
    for model in models_lis:
        print(f"[INFO] Evaluating model: {model.name}")
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for crops, labels in test_dl:
                labels = labels.to(device)
                crops = crops.to(device)  # Move crops to device

                # Forward pass through the model
                outputs = model(crops)

                # Convert outputs to probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())

                # Get the predictions (max over the probabilities)
                _, preds = probs.max(1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)

        top_1_err = managers.MetricManager.get_top_1_err(all_preds, all_labels)
        top_5_err = managers.MetricManager.get_top_5_err(all_probs, all_labels)
        acc = managers.MetricManager.get_accuracy(all_preds, all_labels)

        stats_lis.append(MultiCropEvalStats(
            model_name=model.name,
            train_min_scale=s_min,
            train_max_scale=s_max,
            test_scales=test_scales,
            top_1_err=top_1_err,
            top_5_err=top_5_err,
            acc=acc

        ))

        print(f"[INFO] Top 1 Err: {top_1_err}")
        print(f"[INFO] Top 5 Err: {top_5_err}")
        print(f"[INFO] Accuracy: {acc}\n")
    
    print("=" * 50)
    return stats_lis


if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_WORKERS = 0
    WEIGHTS_FOLDER = "weights-classification"
    MULTISCALE_WEIGHTS = f"{WEIGHTS_FOLDER}/multiscale"
    SCALE_44_WEIGHTS = f"{WEIGHTS_FOLDER}/scale_44"
    SCALE_54_WEIGHTS = f"{WEIGHTS_FOLDER}/scale_54"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_DIM = 32
    RESULTS_FOLDER = "classification-results/multi_crop"

    # scaling 

    # Central crop
    # Corner crops (top-left, top-right, bottom-left, bottom-right)
    # Flipped versions of these crops (horizontally flipped)

    # Three test scales Q were considered: {256, 384, 512}.
    # [44, 48, 54]

    # each image => 6 x f3 crops 
    # each scale = 6 crops 
    # print(test_dl)
    test_scales = [44, 48, 54]
    train_scale_44_models = managers.ModelManager.load_models(SCALE_44_WEIGHTS)

    stats_lis = eval(s_min=44,
         s_max=44,
         test_scales=test_scales,
         models_lis=train_scale_44_models,
         device=DEVICE,
         batch_size=BATCH_SIZE,
         num_workers=NUM_WORKERS
        )





