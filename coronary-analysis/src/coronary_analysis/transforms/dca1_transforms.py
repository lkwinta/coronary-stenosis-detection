import cv2
import albumentations as A


def get_train_transforms(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(
                shift_limit=0.03,
                scale_limit=0.08,
                rotate_limit=12,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=1.0,
                    ),
                    A.CLAHE(
                        clip_limit=2.0,
                        tile_grid_size=(8, 8),
                        p=1.0,
                    ),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                ],
                p=0.25,
            ),
            A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=1.0),
        ]
    )


def get_val_transforms(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=1.0),
        ]
    )
