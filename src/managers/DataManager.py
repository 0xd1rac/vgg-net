from src.common_imports import *
from datasets import load_dataset
from torch.utils.data import DataLoader
from .TransformManager import TransformManager
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
        train_ds = train_ds.select(range(5))

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
        test_ds = test_ds.select(range(80))
        test_ds.set_transform(lambda batch: TransformManager.apply_transforms(batch, test_transform))
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
        return test_dl
