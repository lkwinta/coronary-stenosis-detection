from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DCA1Dataset(Dataset):
    """
    PyTorch dataset for loading DCA1 coronary angiography images and masks.

    This dataset expects grayscale angiography images stored in `.pgm` format,
    with corresponding binary ground-truth masks stored as `{stem}_gt.pgm`
    in the same directory.

    Each sample is loaded as:
    - image: single-channel float tensor of shape `[1, H, W]`
    - mask: single-channel float tensor of shape `[1, H, W]`

    Images are normalized to the range `[0, 1]`.
    Masks are binarized using a threshold of 127.

    Optionally, an augmentation/transform pipeline may be applied. The transform
    is expected to follow the Albumentations convention and accept `image` and
    `mask` keyword arguments.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing DCA1 images and masks.
    stems : list[str] | None, optional
        List of file stems identifying samples to load. For a stem `"12"`,
        the dataset expects:
        - image: `"12.pgm"`
        - mask: `"12_gt.pgm"`
    transform : callable, optional
        Transform or augmentation callable applied jointly to image and mask.
        Typically an `albumentations.Compose` object.

    Raises
    ------
    FileNotFoundError
        If an image or its corresponding mask cannot be loaded.
    """

    def __init__(
        self,
        image_dir: str,
        stems: list[str] = None,
        transform=None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        image_dir : str
            Path to the directory containing `.pgm` images and `_gt.pgm` masks.
        stems : list[str] | None, optional
            List of sample identifiers to include in the dataset.
        transform : callable, optional
            Transform applied jointly to image and mask.
        """
        self.image_dir = Path(image_dir)
        self.stems = stems
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of image-mask pairs.
        """
        return len(self.stems)

    def __getitem__(self, idx: int):
        """
        Load a single image-mask pair.

        Parameters
        ----------
        idx : int
            Index of the sample to load.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple `(image, mask)` where:
            - `image` is a float tensor of shape `[1, H, W]`
            - `mask` is a float tensor of shape `[1, H, W]`

        Notes
        -----
        Processing steps:
        1. Load image and mask in grayscale.
        2. Normalize image to `[0, 1]`.
        3. Binarize mask using threshold `> 127`.
        4. Apply optional transform jointly to image and mask.
        5. Add channel dimension.
        6. Convert to PyTorch tensors.

        Raises
        ------
        FileNotFoundError
            If the image or mask file does not exist or cannot be read.
        """
        stem = self.stems[idx]

        img_path = self.image_dir / f"{stem}.pgm"
        mask_path = self.image_dir / f"{stem}_gt.pgm"

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image file: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask file: {mask_path}")

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = np.expand_dims(image, axis=0)  # Shape: [1, H, W]
        mask = np.expand_dims(mask, axis=0)  # Shape: [1, H, W]

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()
