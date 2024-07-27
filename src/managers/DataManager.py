from src.common_imports import *
from datasets import load_dataset
from .TransformManager import TransformManager

class DataManager:
    @staticmethod
    def get_train_dl(train_transform,
                     batch_size,
                     num_workers,
                     dataset_url="zh-plus/tiny-imagenet"
                     ):
        
        dataset = load_dataset(dataset_url)
        train_ds = dataset['train']
        train_ds = train_ds.select(range(5))

        train_ds.set_transform(lambda batch: TransformManager.apply_transforms(batch, train_transform))
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
        return train_dl