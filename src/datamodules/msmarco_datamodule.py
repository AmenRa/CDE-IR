from typing import Callable, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..datasets.train_dataset import TrainDataset
from ..datasets.train_dataset_rolling import TrainDatasetRolling


class MSMARCO_DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_dataloader_params: Dict = None,
        train_collate_fn: Callable = None,
        val_dataloader_params: Dict = None,
        val_collate_fn: Callable = None,
        test_dataloader_params: Dict = None,
        test_collate_fn: Callable = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_dataloader_params = train_dataloader_params
        self.train_collate_fn = train_collate_fn
        self.val_dataloader_params = val_dataloader_params
        self.val_collate_fn = val_collate_fn
        self.test_dataloader_params = test_dataloader_params
        self.test_collate_fn = test_collate_fn

    def prepare_data(self):
        # TODO: download dataset, load dataset into Redis
        return

    def setup(self, stage: Optional[str] = None):
        # Assign train set for use in dataloader
        if stage in (None, "fit"):
            self.train_set = TrainDatasetRolling(self.data_dir)

        # Assign val set for use in dataloader
        if stage in "validate":
            self.val_set = None

        # Assign test set for use in dataloader
        if stage in "test":
            self.test_set = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            collate_fn=self.train_collate_fn,
            **self.train_dataloader_params,
        )

    def val_dataloader(self) -> DataLoader:
        # TODO: placeholder to make the training working!
        return None

    def test_dataloader(self) -> DataLoader:
        # TODO: placeholder to make the training working!
        return None
