from coronary_analysis.models.segmentation import CoronaryUNetPP
from coronary_analysis.transforms import get_val_transforms
from coronary_analysis.utils import get_device

import cv2
import numpy as np
import torch


def load_segmentation_model(
    model_path: str,
    device: torch.device | None = None,
    encoder_name: str = "resnet34",
) -> CoronaryUNetPP:
    if device is None:
        device = get_device()

    model = CoronaryUNetPP(encoder_name=encoder_name)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    return model


def predict_mask(
    image: np.ndarray,
    model: CoronaryUNetPP,
    device: torch.device | None = None,
    img_size: int = 256,
    threshold: float = 0.5,
) -> np.ndarray:
    if device is None:
        device = get_device()

    original_shape = image.shape[:2]
    transform = get_val_transforms(img_size)
    augmented = transform(
        image=image.astype(np.float32) / 255.0,
        mask=np.zeros_like(image, dtype=np.float32),
    )
    img_tensor = (
        torch.from_numpy(augmented["image"])
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    proba = model.predict_proba(img_tensor)
    mask = (proba > threshold).float()
    mask = mask.squeeze().cpu().numpy().astype(np.uint8)

    if mask.shape != original_shape:
        mask = cv2.resize(
            mask,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask
