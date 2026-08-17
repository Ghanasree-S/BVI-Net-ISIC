"""Training-time augmentation: random flips/rotation + brightness/contrast jitter.
Applied only to the train split -- val/test stay unaugmented for honest evaluation.
"""
import albumentations as A


def get_train_augment():
    """Builds the training-only augmentation pipeline.

    Hyperparameters (each is a probability `p` of applying that
    transform, and a `*_limit` controlling its strength):
        HorizontalFlip / VerticalFlip (p=0.5 each): mirrors the image --
            lesions have no canonical orientation, so this is safe and
            doubles effective data diversity.
        RandomRotate90 (p=0.5): rotates by a random multiple of 90 degrees.
        ShiftScaleRotate (p=0.5): shift_limit=0.05 (max 5% translation),
            scale_limit=0.1 (max 10% zoom in/out), rotate_limit=20 (max
            20-degree rotation) -- simulates camera/lesion positioning
            variation. border_mode=0 fills new pixels with black rather than
            reflecting/wrapping the image.
        RandomBrightnessContrast (p=0.5): brightness_limit=0.2,
            contrast_limit=0.2 -- simulates lighting variation across
            dermoscopy devices.
        HueSaturationValue (p=0.3): hue_shift_limit=10, sat_shift_limit=15,
            val_shift_limit=10 -- simulates skin-tone and color-calibration
            variation across images.

    Returns:
        albumentations.Compose: callable as augment(image=..., mask=...).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5,
                            border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    ])
