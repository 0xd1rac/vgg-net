from src.common_imports import *
from datasets import load_dataset
from torch.utils.data import DataLoader
from .TransformManager import TransformManager
from typing import Callable

from torch.utils.data import DataLoader
from typing import Callable

class DataManager:
    @staticmethod
    def get_train_dl(train_transform: Callable,
                     batch_size: int,
                     num_workers: int,
                     dataset_url: str = "zh-plus/tiny-imagenet"
                     ) -> DataLoader:
        """
        Get the DataLoader for the training dataset.

        Args:
        train_transform (Callable): Transformations to apply to the training dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
        dataset_url (str): URL or identifier for the dataset.

        Returns:
        DataLoader: DataLoader for the training dataset.
        """
        
        dataset = load_dataset(dataset_url)
        train_ds = dataset['train']
        train_ds.set_transform(lambda batch: TransformManager.apply_transforms(batch, train_transform))
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
        return train_dl
    
    @staticmethod
    def get_test_dl(test_transform: Callable,
                    batch_size: int,
                    num_workers: int,
                    dataset_url: str = "zh-plus/tiny-imagenet"
                    ) -> DataLoader:
        """
        Get the DataLoader for the test dataset.

        Args:
        test_transform (Callable): Transformations to apply to the test dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
        dataset_url (str): URL or identifier for the dataset.

        Returns:
        DataLoader: DataLoader for the test dataset.
        """
        
        dataset = load_dataset(dataset_url)
        test_ds = dataset["valid"]
        test_ds.set_transform(lambda batch: TransformManager.apply_transforms(batch, test_transform))
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
        return test_dl


    @staticmethod
    def get_test_dl_multi_crop(test_transform: Callable,
                    batch_size: int,
                    num_workers: int,
                    dataset_url: str = "zh-plus/tiny-imagenet"
                    ) -> DataLoader:
        """
        Get the DataLoader for the test dataset.

        Args:
        test_transform (Callable): Transformations to apply to the test dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
        dataset_url (str): URL or identifier for the dataset.

        Returns:
        DataLoader: DataLoader for the test dataset.
        """
        
        dataset = load_dataset(dataset_url)
        test_ds = dataset["valid"]
        test_ds = test_ds.select(range(80))
        
        # Applying multi-crop transform within the dataset
        test_ds.set_transform(lambda batch: TransformManager.apply_transforms_multi_crop(batch, test_transform))
        
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=DataManager.multi_crop_collate_fn)
        return test_dl

    @staticmethod
    def multi_crop_collate_fn(batch):
        """
        Custom collate function to handle multi-crop batches.

        Args:
        batch: List of tuples (crops, label).

        Returns:
        crops: List of crops.
        labels: Tensor of labels.
        """
        crops, labels = zip(*batch)
        # Flatten the list of lists of crops
        print(f"crops type: {type(crops)}, crops content: {crops[:2]}")  # Print the first two for brevity
        print(f"labels type: {type(labels)}, labels content: {labels[:2]}")  # Print the first two for brevity
        crops = [crop for sublist in crops for crop in sublist]
        print(labels)
        return crops, torch.tensor(labels, dtype=torch.long)
