"""Data augmentation pipelines using Albumentations."""

import logging
from typing import Callable, Dict, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(config: dict) -> A.Compose:
    """Create training augmentation pipeline.
    
    Args:
        config: Configuration dictionary with 'augmentation' and 'data' sections.
        
    Returns:
        Albumentations Compose transform.
    """
    aug_cfg = config.get('augmentation', {})
    data_cfg = config.get('data', {})
    image_size = data_cfg.get('image_size', 518)

    # Every probability below is read from config. Previously `rotate_p`,
    # `brightness_contrast_p`, `coarse_dropout_p`, `elastic_transform_p` and
    # `clahe` were hardcoded, which meant the A10_no_augmentation variant still
    # applied ElasticTransform, CLAHE and CoarseDropout on every batch and was
    # therefore not measuring what its name claims. The defaults here are the
    # exact former hardcoded values, so every other variant is unchanged.
    ops = []

    def _add(op, p):
        """Append an op only when it can actually fire (p > 0)."""
        if p > 0:
            ops.append(op)

    # --- Geometric transforms ---
    _add(A.HorizontalFlip(p=aug_cfg.get('horizontal_flip', 0.5)),
         aug_cfg.get('horizontal_flip', 0.5))
    _add(A.VerticalFlip(p=aug_cfg.get('vertical_flip', 0.3)),
         aug_cfg.get('vertical_flip', 0.3))

    rotation_limit = aug_cfg.get('rotation_limit', 15)
    rotate_p = aug_cfg.get('rotate_p', 0.5)
    # A zero rotation limit makes the op a no-op; skip it outright so the
    # "no augmentation" variant really has an empty geometric stage.
    _add(A.Rotate(limit=rotation_limit, border_mode=0, p=rotate_p),
         rotate_p if rotation_limit else 0.0)

    # --- Color transforms ---
    brightness_limit = aug_cfg.get('brightness_limit', 0.2)
    contrast_limit = aug_cfg.get('contrast_limit', 0.2)
    bc_p = aug_cfg.get('brightness_contrast_p', 0.5)
    _add(
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=bc_p,
        ),
        bc_p if (brightness_limit or contrast_limit) else 0.0,
    )
    _add(A.GaussianBlur(blur_limit=(3, 7), p=aug_cfg.get('gaussian_blur_p', 0.3)),
         aug_cfg.get('gaussian_blur_p', 0.3))

    # --- Occlusion / deformation ---
    coarse_dropout_p = aug_cfg.get('coarse_dropout_p', 0.3)
    _add(
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(image_size // 40, image_size // 20),
            hole_width_range=(image_size // 40, image_size // 20),
            fill=0,
            p=coarse_dropout_p,
        ),
        coarse_dropout_p,
    )
    elastic_p = aug_cfg.get('elastic_transform_p', 0.3)
    _add(A.ElasticTransform(p=elastic_p), elastic_p)
    clahe_p = aug_cfg.get('clahe', 0.3)
    _add(A.CLAHE(clip_limit=aug_cfg.get('clahe_clip_limit', 2.0), p=clahe_p), clahe_p)

    # Normalize and convert to tensor (never optional)
    ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    ops.append(ToTensorV2())

    transform = A.Compose(ops)

    num_aug = len(ops) - 2
    if num_aug == 0:
        logger.info("Training transform created with NO augmentation (normalize only)")
    else:
        logger.info(f"Training transform created with {num_aug} active augmentation(s)")
    return transform


def get_val_transform(config: dict) -> A.Compose:
    """Create validation/test transform pipeline (no augmentation).
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Albumentations Compose transform.
    """
    transform = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    
    logger.info("Validation transform created")
    return transform


def get_inference_transform(image_size: int = 518) -> A.Compose:
    """Create inference transform pipeline.
    
    Args:
        image_size: Target size for resizing.
        
    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def denormalize(image: np.ndarray) -> np.ndarray:
    """Denormalize image from ImageNet normalization.
    
    Args:
        image: Normalized image array (H, W, C) or (C, H, W).
        
    Returns:
        Denormalized image in [0, 255] range.
    """
    if image.shape[0] == 3:  # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    
    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


if __name__ == "__main__":
    # Test augmentation pipeline
    logging.basicConfig(level=logging.INFO)
    
    # Mock config
    config = {
        'data': {'image_size': 518},
        'augmentation': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.3,
            'rotation_limit': 15,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'gaussian_blur_p': 0.3
        }
    }
    
    train_transform = get_train_transform(config)
    val_transform = get_val_transform(config)
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (518, 518), dtype=np.uint8)
    
    transformed = train_transform(image=dummy_image, mask=dummy_mask)
    
    print(f"Image shape: {transformed['image'].shape}")
    print(f"Mask shape: {transformed['mask'].shape}")
    print(f"Image dtype: {transformed['image'].dtype}")
    print(f"Mask dtype: {transformed['mask'].dtype}")
