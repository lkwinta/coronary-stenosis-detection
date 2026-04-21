import cv2
import albumentations as A


def get_lm_cad_train_transforms(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(-0.03, 0.03),
                rotate=(-12, 12),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
        ],
        additional_targets={"image_b": "image"},
    )


def get_lm_cad_val_transforms(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        ],
        additional_targets={"image_b": "image"},
    )
