from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class ArcadeSyntaxBinaryDataset(Dataset):
    """
    PyTorch dataset for loading ARCADE SYNTAX images as binary vessel-region masks.

    This dataset reads grayscale angiography images from the ARCADE `syntax`
    split and converts all COCO annotations for a given image into a single
    binary mask. Any annotated SYNTAX vessel segment is mapped to foreground.

    Each sample is returned as:
    - image: single-channel float tensor of shape `[1, H, W]`
    - mask: single-channel float tensor of shape `[1, H, W]`

    Images are normalized to the range `[0, 1]`.
    Masks are binary and represent the union of all annotated vessel regions
    for the selected image.

    This is useful when ARCADE SYNTAX is used as a weak pretraining dataset
    for vessel-region localization before fine-tuning on a dataset with
    full binary vessel masks such as DCA1.

    Parameters
    ----------
    root : str
        Path to the root ARCADE syntax directory. The dataset expects the
        following structure:

        - `{root}/{split}/images/`
        - `{root}/{split}/annotations/{split}.json`

    split : str, optional
        Dataset split to use, e.g. `"train"` or `"val"`. Default is `"train"`.
    transform : callable, optional
        Transform or augmentation callable applied jointly to image and mask.
        Typically an `albumentations.Compose` object.

    Notes
    -----
    This dataset uses `coco.imgToAnns[img_id]` to retrieve annotations for
    an image, which is safer when working with ARCADE training annotations
    if duplicate annotation IDs are present.

    Raises
    ------
    FileNotFoundError
        If an image referenced in the annotation file cannot be loaded.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        """
        Initialize the dataset.

        Parameters
        ----------
        root : str
            Root directory containing ARCADE SYNTAX data.
        split : str, optional
            Dataset split name. Default is `"train"`.
        transform : callable, optional
            Transform applied jointly to image and mask.
        """
        self.root = Path(root) / split
        self.image_dir = self.root / "images"
        self.ann_file = self.root / "annotations" / f"{split}.json"
        self.transform = transform

        self.coco = COCO(str(self.ann_file))

        # Only keep images that actually have annotations
        self.image_ids = sorted(list(self.coco.imgToAnns.keys()))

    def __len__(self):
        """
        Return the number of annotated images in the dataset.

        Returns
        -------
        int
            Number of image-mask pairs available.
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        """
        Load a single image and its merged binary SYNTAX mask.

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
        1. Load the grayscale image referenced by the COCO entry.
        2. Normalize the image to `[0, 1]`.
        3. Retrieve all annotations associated with the image.
        4. Convert each annotation to a mask and merge them into one binary mask.
        5. Apply optional transform jointly to image and mask.
        6. Add channel dimension.
        7. Convert both to PyTorch tensors.

        Raises
        ------
        FileNotFoundError
            If the image file cannot be read.
        """
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]

        img_path = self.image_dir / img_info["file_name"]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image file: {img_path}")

        image = image.astype(np.float32) / 255.0

        anns = self.coco.imgToAnns[img_id]

        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for ann in anns:
            ann_mask = self.coco.annToMask(ann)
            mask = np.maximum(mask, ann_mask.astype(np.uint8))

        mask = mask.astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = np.expand_dims(image, axis=0)  # Shape: [1, H, W]
        mask = np.expand_dims(mask, axis=0)  # Shape: [1, H, W]

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()
