import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


def get_fs_cad_pairs(image_dir: str) -> list[tuple[str, str]]:
    files_A_dir = os.path.join(image_dir, "A")
    files_GT_dir = os.path.join(image_dir, "GT")

    files_A = set(os.listdir(files_A_dir))
    files_GT = set(os.listdir(files_GT_dir))

    pairs = []

    for file_GT in files_GT:
        if file_GT.startswith(".") or not file_GT.endswith(".png"):
            continue

        if file_GT in files_A:
            pairs.append((file_GT, file_GT))

    return pairs


class FSCADDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        pairs: list[str] = None,
        transform: A.Compose | None = None,
    ) -> None:
        self.image_dir = image_dir
        if pairs is None:
            self.pairs = get_fs_cad_pairs(image_dir)
        else:
            self.pairs = pairs
        self.transform = transform

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_img, file_mask = self.pairs[idx]

        image_path = os.path.join(self.image_dir, "A", file_img)
        mask_path = os.path.join(self.image_dir, "GT", file_mask)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not load image: {file_img}")
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {file_mask}")

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = np.expand_dims(image, axis=0)  # [1, H, W]
        mask = np.expand_dims(mask, axis=0)  # [1, H, W]

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()

    def __len__(self) -> int:
        return len(self.pairs)
