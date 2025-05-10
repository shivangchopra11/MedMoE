import os
from typing import Any, Dict, Optional, Tuple
import subprocess
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# from torchvision.datasets import MNIST
from src.data.components.slake import SlakeDataset, collate_fn
from src.data.components.chexpert import CheXpertImageDataset
from utils.utils import build_transformation
from omegaconf import DictConfig


class ChexpertDataModule(LightningDataModule):
    def __init__(self, 
        data_dir: str,
        transformations: DictConfig,
        sample_frac: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """Initialize a `SLAKEDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.batch_size_per_device = batch_size
        self.sample_frac = sample_frac
        self.dataset = CheXpertImageDataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transformations = transformations

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        print('Downloading')
        if os.path.exists("datasets/chexpert") == False:
            rc = subprocess.call("python datasets/download_chexpert.py datasets", shell=True)

    def train_dataloader(self):
        transform = build_transformation(self.transformations, "train")
        dataset = self.dataset(self.data_dir, self.sample_frac, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        transform = build_transformation(self.transformations, "valid")
        dataset = self.dataset(self.data_dir, self.sample_frac, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=False,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        transform = build_transformation(self.transformations, "test")
        dataset = self.dataset(self.data_dir, self.sample_frac, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=self.pin_memory,
            shuffle=False,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    _ = CheXpertDataModule()
