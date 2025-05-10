import os
from typing import Any, Dict, Optional, Tuple
import subprocess
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.components.slake import SlakeDataset, collate_fn
from omegaconf import DictConfig
from utils.utils import build_transformation


class SLAKEDataModule(LightningDataModule):

    def __init__(
        self,
        transformations: DictConfig = None,
        data_dir: str = "datasets/slake",
        label_type: str = None,
        content_type: str = None,
        modality: str = None,
        language: str = "en",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        img_id_limit: int = None,
    ) -> None:
        """Initialize a `SLAKEDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.collate_fn = collate_fn
        self.content_type = content_type # "Abnormality" if "abnormal" in label_type else None # modality, abnormality, location, organ
        self.label_type = label_type
        self.language = language 
        self.modality = modality
        self.img_id_limit = img_id_limit
        self.transformations = transformations
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        if "abnormal" in self.label_type.lower():
            return 2
        elif self.label_type == "organ":
            return 104 # from en_organ_rel.csv
        elif self.label_type == "modality":
            return 3 # from en_organ_rel.csv
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        if os.path.exists("datasets/slake") == False:
            rc = subprocess.call("datasets/download_slake.sh")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            transform = build_transformation(self.transformations, "train")
            mask_transform = build_transformation(self.transformations, "train", mask=True)
            self.data_train = SlakeDataset(
                dataset_root_path = self.hparams.data_dir, 
                split='train', 
                mask_transform=mask_transform,
                transform=transform,
                label_type=self.label_type,
                content_type=self.content_type, 
                language=self.language,
                modality=self.modality,
                img_id_limit=self.img_id_limit
            )
            transform = build_transformation(self.transformations, "valid")
            self.data_val = SlakeDataset(
                dataset_root_path=self.hparams.data_dir, 
                transform=transform,
                mask_transform=mask_transform,
                split='validate', 
                label_type=self.label_type,
                content_type=self.content_type, 
                language=self.language,
                modality=self.modality,
                img_id_limit=self.img_id_limit
            )
            transform = build_transformation(self.transformations, "test")
            self.data_test = SlakeDataset(
                dataset_root_path=self.hparams.data_dir,
                transform=transform,
                mask_transform=mask_transform,
                split='test', 
                label_type=self.label_type,
                content_type=self.content_type, 
                language=self.language,
                modality=self.modality,
                img_id_limit=self.img_id_limit
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = SLAKEDataModule()
