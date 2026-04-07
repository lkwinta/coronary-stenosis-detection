import torch
from torch.utils.data import Dataset

import albumentations as A

from .dca1 import DCA1Dataset, get_dca1_pairs
from .fs_cad import FSCADDataset, get_fs_cad_pairs


class DCA1FSCADDataset(Dataset):
    def __init__(
        self,
        dca1_image_dir: str,
        fs_cad_image_dir: str,
        dca1_pairs: list[tuple[str, str]] | None = None,
        fs_cad_pairs: list[tuple[str, str]] | None = None,
        transform: A.Compose | None = None,
    ) -> None:
        if dca1_pairs is None:
            dca1_pairs = get_dca1_pairs(dca1_image_dir)

        if fs_cad_pairs is None:
            fs_cad_pairs = get_fs_cad_pairs(fs_cad_image_dir)

        self.dca1_dataset = DCA1Dataset(
            image_dir=dca1_image_dir,
            pairs=dca1_pairs,
            transform=transform,
        )
        self.fs_cad_dataset = FSCADDataset(
            image_dir=fs_cad_image_dir,
            pairs=fs_cad_pairs,
            transform=transform,
        )

        self.indices = torch.randperm(len(self.dca1_dataset) + len(self.fs_cad_dataset))

    def __len__(self) -> int:
        return len(self.dca1_dataset) + len(self.fs_cad_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = self.indices[idx].item()

        if idx < len(self.dca1_dataset):
            return self.dca1_dataset[idx]
        else:
            return self.fs_cad_dataset[idx - len(self.dca1_dataset)]
