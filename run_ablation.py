"""
Ablation study runner for DINOv2-DenseFPN-UNet.

Each variant modifies ONE component vs. the full model. The model itself
(``models/``, ``loss/``) is never touched by this script -- it only builds
configs, drives training, and records metrics.

Usage:
    python run_ablation.py --mode eval_only   # evaluate existing checkpoints only
    python run_ablation.py --mode full        # train + eval all variants
    python run_ablation.py --mode quick       # short training run for ablations

Protocol notes (see README "Ablation protocol" for the full rationale):
  * Checkpoint selection uses a held-out VALIDATION split carved from the
    training pool. The ``test`` split is scored exactly once, at the end, and
    never influences which epoch is kept.
  * Every variant trains under the identical budget, seed, and schedule, so
    each row is a controlled one-factor change against A1.
  * Each variant runs in its own subprocess by default, so a crash or an OOM
    in one variant cannot take down the study or leak memory into the next.
"""

import argparse
import copy
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Ensure project root is in sys.path so local modules (data, models, loss, etc.) can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["Background", "Dust", "RunDown", "Scratch"]


class _ProtocolMismatch(Exception):
    """Raised internally when a checkpoint predates the current protocol."""

# ---------------------------------------------------------------------------
# torch.amp compatibility
# ---------------------------------------------------------------------------
# torch>=2.4 deprecates torch.cuda.amp.autocast / GradScaler in favour of the
# device-generic torch.amp API. Route through one shim so the run log is not
# drowned in FutureWarnings on the RTX 3060 box (torch 2.6+ / CUDA 12.6).

def _autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG that affects a run.

    Ablation deltas are only interpretable if the variants differ by the one
    factor under test and not by initialisation or shuffling order.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic, distinct seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "model": {
        "encoder": "dinov2_vitb14",
        "encoder_frozen": True,
        "skip_layers": [3, 7, 11],
        "decoder_channels": [256, 128, 64],
        "num_classes": 4,
    },
    "training": {
        "batch_size": 8,
        "num_epochs": 15,           # fast ablation epochs (frozen encoder)
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_epochs": 3,         # now actually honoured (see build_scheduler)
        "gradient_clip": 1.0,
        "accumulation_steps": 1,
        "unfreeze_encoder_epoch": 999,  # keep frozen for speed
        "unfreeze_last_n_blocks": None,  # None = unfreeze the whole encoder
        "encoder_lr": 1e-5,
        "scheduler": "cosine_warmup",   # or "warm_restarts" (the historical one)
        "min_lr_factor": 0.01,
        "seed": 42,
    },
    "loss": {
        "dice_weight": 0.5,
        "focal_weight": 0.5,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "class_weights": [0.3, 3.0, 2.0, 2.0],
    },
    "data": {
        "image_size": 518,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "num_workers": 0,
        "resize_mode": "stretch",
        "cache_dir": None,
        # Fraction of the training pool held out for checkpoint selection.
        "val_fraction": 0.1,
    },
    "augmentation": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.3,
        "rotation_limit": 15,
        "rotate_p": 0.5,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "brightness_contrast_p": 0.5,
        "gaussian_blur_p": 0.3,
        "coarse_dropout_p": 0.3,
        "clahe": 0.3,
        "elastic_transform_p": 0.3,
    },
    "paths": {
        "data_root": "./data/data",
        "save_dir": "./checkpoints/ablation",
        "log_dir": "./logs/ablation",
    },
}

# Every augmentation probability, for the "turn everything off" variant. Kept
# as a named constant so A10 cannot silently drift out of sync with the list of
# augmentations actually applied in data/augmentation.py.
NO_AUGMENTATION = {
    "horizontal_flip": 0.0,
    "vertical_flip": 0.0,
    "rotation_limit": 0,
    "rotate_p": 0.0,
    "brightness_limit": 0.0,
    "contrast_limit": 0.0,
    "brightness_contrast_p": 0.0,
    "gaussian_blur_p": 0.0,
    "coarse_dropout_p": 0.0,
    "clahe": 0.0,
    "elastic_transform_p": 0.0,
}

ABLATION_VARIANTS = [
    {
        "name": "A1_full_model",
        "description": "Full model — DINOv2-ViT-B/14 + DenseFPN-UNet (3-scale) + combined loss",
        # Trained fresh under the same budget as every other variant so the
        # ablation deltas are interpretable. The 200-epoch production
        # checkpoint is reported separately via --with-pretrained-baseline.
        "checkpoint": None,
        "config_overrides": {},
    },
    {
        "name": "A2_single_scale",
        "description": "Single-scale features (only deep layer [11], no FPN)",
        "checkpoint": None,
        "config_overrides": {
            "model": {"skip_layers": [11]},
        },
    },
    {
        "name": "A3_two_scale",
        "description": "Two-scale features ([7, 11], mid + deep)",
        "checkpoint": None,
        "config_overrides": {
            "model": {"skip_layers": [7, 11]},
        },
    },
    {
        "name": "A4_partial_unfreeze",
        # The previous "A4_frozen_encoder" set encoder_frozen=True and
        # unfreeze_encoder_epoch=999 -- both already the BASE_CONFIG values, so
        # it was byte-identical to the A1 baseline and carried zero information.
        # Replaced with the missing middle point of the freeze axis:
        #   A1 = frozen throughout | A4 = last 4 blocks | A9 = whole encoder.
        "description": "Encoder partially unfrozen: last 4 ViT blocks fine-tuned from epoch 5",
        "checkpoint": None,
        "config_overrides": {
            "model": {"encoder_frozen": True},
            "training": {
                "unfreeze_encoder_epoch": 5,
                "unfreeze_last_n_blocks": 4,
                "encoder_lr": 5e-6,
            },
        },
    },
    {
        "name": "A5_dice_only",
        "description": "Loss: Dice only (no Focal loss)",
        "checkpoint": None,
        "config_overrides": {
            "loss": {"dice_weight": 1.0, "focal_weight": 0.0},
        },
    },
    {
        "name": "A6_focal_only",
        "description": "Loss: Focal only (no Dice loss)",
        "checkpoint": None,
        "config_overrides": {
            "loss": {"dice_weight": 0.0, "focal_weight": 1.0},
        },
    },
    {
        "name": "A7_no_class_weights",
        "description": "Uniform class weights [1, 1, 1, 1] (no class balancing)",
        "checkpoint": None,
        "config_overrides": {
            "loss": {"class_weights": [1.0, 1.0, 1.0, 1.0]},
        },
    },
    {
        "name": "A8_small_decoder",
        "description": "Smaller decoder channels [128, 64, 32]",
        "checkpoint": None,
        "config_overrides": {
            "model": {"decoder_channels": [128, 64, 32]},
        },
    },
    {
        "name": "A9_vitb_unfrozen",
        "description": "ViT-B encoder fully unfrozen from epoch 5 (aggressive fine-tuning)",
        "checkpoint": None,
        "config_overrides": {
            "model": {"encoder_frozen": True},
            "training": {
                "unfreeze_encoder_epoch": 5,
                "unfreeze_last_n_blocks": None,
                "encoder_lr": 5e-6,
                # Halved batch with matching accumulation keeps the effective
                # batch at 8 while fitting the unfrozen ViT-B in 12 GB.
                "batch_size": 4,
                "accumulation_steps": 2,
            },
        },
    },
    {
        "name": "A10_no_augmentation",
        "description": "No augmentation (all geometric/photometric transforms disabled)",
        "checkpoint": None,
        "config_overrides": {
            "augmentation": dict(NO_AUGMENTATION),
        },
    },
    {
        "name": "A11_high_gamma",
        "description": "Focal loss gamma=4.0 (harder focus on difficult samples)",
        "checkpoint": None,
        "config_overrides": {
            "loss": {"focal_gamma": 4.0},
        },
    },
    {
        "name": "A12_lower_lr",
        "description": "Lower learning rate (1e-5 instead of 5e-5)",
        "checkpoint": None,
        "config_overrides": {
            "training": {"learning_rate": 1e-5},
        },
    },
    {
        "name": "A13_higher_lr",
        "description": "Higher learning rate (1e-4)",
        "checkpoint": None,
        "config_overrides": {
            "training": {"learning_rate": 1e-4},
        },
    },
]

# Reported alongside the study, never as one of its rows: the deployed
# 200-epoch production checkpoint. Enabled with --with-pretrained-baseline.
PRETRAINED_REFERENCE = {
    "name": "A0_pretrained_reference",
    "description": "Reference only — deployed 200-epoch production checkpoint (NOT protocol-matched)",
    "checkpoint": "checkpoints/black_best_model.pth",
    "config_overrides": {},
}


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def build_config(variant: dict) -> dict:
    return deep_update(BASE_CONFIG, variant.get("config_overrides", {}))


# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_data_root(data_root: Path) -> Path:
    """Resolve the historical './data' vs './data/data' nesting."""
    if (data_root / "train").exists():
        return data_root
    if (data_root / "data" / "train").exists():
        return data_root / "data"
    if (data_root.parent / "train").exists():
        return data_root.parent
    raise FileNotFoundError(
        f"No train/ directory found under {data_root} or {data_root / 'data'}. "
        f"Expected <root>/train/images and <root>/train/labels."
    )


def _list_images(image_dir: Path) -> List[Path]:
    """List images case-insensitively, deduped by lowercase name."""
    seen, out = set(), []
    for p in sorted(image_dir.iterdir()) if image_dir.exists() else []:
        if p.suffix.lower() in IMAGE_EXTS and p.name.lower() not in seen:
            seen.add(p.name.lower())
            out.append(p)
    return sorted(out)


def _label_signature(label_dir: Path, image_path: Path) -> Tuple[int, ...]:
    """Set of class ids present in an image's label file, as a sortable key.

    Used to stratify the validation split so that rare classes are represented
    in it. Without this, a seeded random split can easily produce a validation
    set with zero RunDown -- which is exactly the state the shipped `valid/`
    directory was in, making it useless for selecting on that class.
    """
    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        return ()
    ids = set()
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    ids.add(int(parts[0]))
    except Exception as e:
        logger.warning(f"Could not parse {label_path}: {e}")
    return tuple(sorted(ids))


def build_splits(
    data_root: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    pool_valid_into_train: bool = True,
) -> Dict[str, Any]:
    """Build deterministic train / val / test splits.

    The shipped layout has three directories, but the previous runner searched
    ``["test", "val", "valid"]`` in that order and therefore selected ``test``
    as its validation set. Best-epoch selection ran against the test split and
    the same split was then reported -- the reported number was the maximum
    test score over all epochs, not a held-out measurement.

    Here:
      * ``train/`` and (optionally) ``valid/`` are pooled,
      * a stratified, seeded ``val_fraction`` of that pool is held out for
        checkpoint selection,
      * ``test/`` is never touched until the final evaluation.

    Returns:
        Dict with 'train'/'val'/'test' entries, each holding image path lists
        plus the label directory, and a 'summary' block for logging.
    """
    data_root = resolve_data_root(data_root)

    train_img_dir = data_root / "train" / "images"
    train_lbl_dir = data_root / "train" / "labels"
    if not train_img_dir.exists():
        raise FileNotFoundError(f"Missing {train_img_dir}")

    pool: List[Tuple[Path, Path]] = [(p, train_lbl_dir) for p in _list_images(train_img_dir)]

    pooled_valid = 0
    if pool_valid_into_train:
        for name in ("valid", "val"):
            vdir = data_root / name / "images"
            if vdir.exists():
                vlbl = data_root / name / "labels"
                extra = [(p, vlbl) for p in _list_images(vdir)]
                pool.extend(extra)
                pooled_valid += len(extra)
                break

    # Test split: held out, scored once.
    test_img_dir, test_lbl_dir = None, None
    for name in ("test", "valid", "val"):
        d = data_root / name / "images"
        if d.exists():
            test_img_dir, test_lbl_dir = d, data_root / name / "labels"
            break
    if test_img_dir is None:
        raise FileNotFoundError(
            f"No test/ (or valid/) directory under {data_root}; cannot form a held-out test set."
        )
    test_files = _list_images(test_img_dir)

    # --- stratified, seeded validation carve-out ---
    groups: Dict[Tuple[int, ...], List[Tuple[Path, Path]]] = {}
    for img, lbl in pool:
        groups.setdefault(_label_signature(lbl, img), []).append((img, lbl))

    rng = random.Random(seed)
    val_pairs: List[Tuple[Path, Path]] = []
    train_pairs: List[Tuple[Path, Path]] = []
    for key in sorted(groups.keys()):
        members = sorted(groups[key], key=lambda t: str(t[0]))
        rng.shuffle(members)
        n_val = int(round(len(members) * val_fraction))
        # Never let a stratum vanish from either side.
        if len(members) >= 2:
            n_val = max(1, min(n_val, len(members) - 1))
        else:
            n_val = 0
        val_pairs.extend(members[:n_val])
        train_pairs.extend(members[n_val:])

    def _class_counts(pairs_or_files, lbl_dir=None):
        counts: Dict[int, int] = {}
        for item in pairs_or_files:
            img, lbl = item if isinstance(item, tuple) else (item, lbl_dir)
            for cid in _label_signature(lbl, img):
                counts[cid] = counts.get(cid, 0) + 1
        return {CLASS_NAMES[c + 1] if c + 1 < len(CLASS_NAMES) else f"class_{c}": n
                for c, n in sorted(counts.items())}

    summary = {
        "data_root": str(data_root),
        "pooled_valid_images": pooled_valid,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_test": len(test_files),
        "val_fraction": val_fraction,
        "split_seed": seed,
        "train_images_with_class": _class_counts(train_pairs),
        "val_images_with_class": _class_counts(val_pairs),
        "test_images_with_class": _class_counts(test_files, test_lbl_dir),
    }

    return {
        "train": {"files": [p for p, _ in train_pairs],
                  "label_dirs": [d for _, d in train_pairs],
                  "image_dir": train_img_dir, "label_dir": train_lbl_dir},
        "val": {"files": [p for p, _ in val_pairs],
                "label_dirs": [d for _, d in val_pairs],
                "image_dir": train_img_dir, "label_dir": train_lbl_dir},
        "test": {"files": test_files, "label_dirs": [test_lbl_dir] * len(test_files),
                 "image_dir": test_img_dir, "label_dir": test_lbl_dir},
        "summary": summary,
    }


class _MultiDirDefectDataset:
    """Thin wrapper letting one dataset span images from several label dirs.

    Pooling ``train/`` and ``valid/`` means a single split can contain images
    whose labels live in different directories. Rather than copy files around
    on disk, hold one DefectDataset per label directory and index across them.
    """

    def __init__(self, groups: List[Any]):
        self.groups = groups
        self.index: List[Tuple[int, int]] = []
        for gi, g in enumerate(groups):
            for li in range(len(g)):
                self.index.append((gi, li))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        gi, li = self.index[idx]
        return self.groups[gi][li]


def make_dataset(split: Dict[str, Any], config: dict, transform, is_training: bool):
    """Build a Dataset for one split, honouring resize_mode and the cache."""
    from data.dataset import DefectDataset

    num_classes = config["model"]["num_classes"]
    data_cfg = config["data"]

    by_dir: Dict[Path, List[Path]] = {}
    for img, lbl in zip(split["files"], split["label_dirs"]):
        by_dir.setdefault(lbl, []).append(img)

    groups = []
    for lbl_dir in sorted(by_dir.keys(), key=str):
        files = sorted(by_dir[lbl_dir], key=str)
        groups.append(
            DefectDataset(
                image_dir=str(files[0].parent),
                label_dir=str(lbl_dir),
                image_size=data_cfg["image_size"],
                transform=transform,
                num_classes=num_classes - 1,
                is_training=is_training,
                resize_mode=data_cfg.get("resize_mode", "stretch"),
                cache_dir=data_cfg.get("cache_dir"),
                file_list=files,
            )
        )

    if len(groups) == 1:
        return groups[0]
    return _MultiDirDefectDataset(groups)


def make_loader(dataset, config: dict, shuffle: bool, batch_size: Optional[int] = None,
                drop_last: bool = False):
    """Build a DataLoader with worker settings that survive Windows."""
    data_cfg = config["data"]
    num_workers = int(data_cfg.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(config["training"].get("seed", 42))

    kwargs = dict(
        batch_size=batch_size or config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    if num_workers > 0:
        # persistent_workers avoids re-spawning processes every epoch, which on
        # Windows (spawn start method) costs seconds per epoch.
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    elif shuffle:
        kwargs.update(generator=generator)

    return torch.utils.data.DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def compute_lr_scale(epoch: int, n_epochs: int, warmup_epochs: int, min_lr_factor: float) -> float:
    """Linear warmup then cosine decay, returned as a multiplier on the base LR.

    ``warmup_epochs`` was declared in configs/config.yaml but no code ever read
    it. A decoder trained from scratch on top of a frozen ViT benefits
    noticeably from the warmup, and the previous
    ``CosineAnnealingWarmRestarts(T_0=n_epochs//3)`` restarted the LR to its
    maximum twice during a 100-epoch run -- which is why best epochs landed
    anywhere between 15 and 83 across variants.
    """
    warmup_epochs = max(0, min(int(warmup_epochs), max(n_epochs - 1, 0)))
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    denom = max(n_epochs - warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / denom
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(min_lr_factor + (1.0 - min_lr_factor) * cosine)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, test_loader, device, num_classes, class_names, desc="Evaluating"):
    """Run evaluation over a loader, returning the full metric dictionary."""
    from utils.metrics import SegmentationMetrics

    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, class_names=class_names)
    times: List[float] = []
    images_seen = 0

    for batch in tqdm(test_loader, desc=desc, leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with _autocast(enabled=torch.cuda.is_available()):
            out = model(imgs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        images_seen += imgs.shape[0]
        pred = out.argmax(dim=1)
        metrics.update(pred, masks)

    results = metrics.compute()
    total_time = float(np.sum(times)) if times else 0.0
    # Per-image latency measured against images actually seen, rather than
    # assuming every batch was full (the last batch usually is not).
    results["avg_inference_ms"] = (total_time * 1000.0 / images_seen) if images_seen else 0.0
    results["throughput_img_per_s"] = (images_seen / total_time) if total_time > 0 else 0.0
    results["num_samples"] = images_seen
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _release_model(model) -> None:
    """Tear down a model so its memory is actually reclaimed.

    DINOv2Encoder registers forward hooks whose closures capture the encoder,
    creating a reference cycle between the module and its own hooks. Combined
    with the cached activations in ``encoder.features``, that keeps a full
    ViT-B (plus its AdamW state) alive far longer than expected. Running 13
    variants in one process is how the study ended up dying on a host-RAM
    allocation of 4.45 MiB.
    """
    import gc
    try:
        enc = getattr(model, "encoder", None)
        if enc is not None:
            for hook in getattr(enc, "_hooks", []) or []:
                try:
                    hook.remove()
                except Exception:
                    pass
            if hasattr(enc, "_hooks"):
                enc._hooks = []
            if hasattr(enc, "features"):
                enc.features.clear()
    except Exception as e:
        logger.debug(f"Encoder teardown notice: {e}")

    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def train_variant(
    config: dict,
    variant_name: str,
    splits: Dict[str, Any],
    num_epochs: int = None,
    resume: bool = True,
    force_resume: bool = False,
    mongo_syncer: Optional[Any] = None,
    dry_run_sync: bool = False,
    run_id: Optional[str] = None,
):
    """Train a single ablation variant and return (best_checkpoint, best_val_iou)."""
    import gc
    import torch.nn as nn
    from torch.optim import AdamW

    from data.augmentation import get_train_transform, get_val_transform
    from models.model import build_model
    from loss.losses import build_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_cfg = config["training"]
    model_cfg = config["model"]
    num_classes = model_cfg["num_classes"]
    class_names = CLASS_NAMES[:num_classes]

    set_seed(t_cfg.get("seed", 42))
    logger.info(f"Training {variant_name} on {device}")

    train_ds = make_dataset(splits["train"], config, get_train_transform(config), True)
    val_ds = make_dataset(splits["val"], config, get_val_transform(config), False)

    train_loader = make_loader(train_ds, config, shuffle=True, drop_last=True)
    val_loader = make_loader(val_ds, config, shuffle=False)

    logger.info(
        f"[{variant_name}] train={len(train_ds)} images, "
        f"val={len(val_ds)} images (held out for checkpoint selection)"
    )

    model = build_model(config).to(device)
    criterion = build_loss(config)
    n_epochs = num_epochs or t_cfg["num_epochs"]

    # Param groups: separate encoder LR
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    base_decoder_lr = t_cfg["learning_rate"]
    base_encoder_lr = t_cfg.get("encoder_lr", t_cfg["learning_rate"])
    encoder_active = not model_cfg.get("encoder_frozen", True)

    param_groups = [
        {"params": decoder_params, "lr": base_decoder_lr},
        {"params": encoder_params, "lr": base_encoder_lr if encoder_active else 0.0},
    ]
    optimizer = AdamW(param_groups, weight_decay=t_cfg["weight_decay"])
    scaler = _grad_scaler(enabled=torch.cuda.is_available())

    scheduler_name = t_cfg.get("scheduler", "cosine_warmup")
    legacy_scheduler = None
    if scheduler_name == "warm_restarts":
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        legacy_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=max(n_epochs // 3, 5), T_mult=1
        )

    save_dir = Path(config["paths"]["save_dir"]) / variant_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    best_ckpt = save_dir / "best.pth"
    latest_ckpt = save_dir / "latest.pth"
    history: List[Dict[str, float]] = []

    # Resume support
    start_epoch = 0
    ckpt_to_load = latest_ckpt if latest_ckpt.exists() else (best_ckpt if best_ckpt.exists() else None)

    if resume and ckpt_to_load:
        try:
            logger.info(f"[{variant_name}] Found existing checkpoint: {ckpt_to_load.name}. Loading state...")
            ckpt = torch.load(ckpt_to_load, map_location=device, weights_only=False)

            # Refuse to resume a checkpoint produced under a different protocol.
            # Weights selected against the test split, or trained on a different
            # LR schedule, would silently contaminate a corrected run.
            prev_protocol = (ckpt.get("config", {}) or {}).get("_protocol")
            cur_protocol = config.get("_protocol")
            if prev_protocol != cur_protocol and not force_resume:
                logger.warning(
                    f"[{variant_name}] Ignoring checkpoint {ckpt_to_load.name}: it was produced "
                    f"under protocol {prev_protocol!r}, but this run uses {cur_protocol!r}. "
                    f"Training from scratch so the two are not mixed. "
                    f"Pass --force-resume to override, or delete "
                    f"{save_dir} to silence this."
                )
                del ckpt
                raise _ProtocolMismatch()

            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if legacy_scheduler is not None and "scheduler_state_dict" in ckpt:
                legacy_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            best_iou = ckpt.get("best_iou", 0.0)
            history = ckpt.get("history", []) or []
            start_epoch = ckpt.get("epoch", -1) + 1
            logger.info(
                f"[{variant_name}] Resuming from epoch {start_epoch + 1}/{n_epochs} "
                f"(best val mIoU so far: {best_iou:.4f})"
            )
        except _ProtocolMismatch:
            start_epoch, best_iou, history = 0, 0.0, []
        except Exception as e:
            logger.warning(f"[{variant_name}] Failed to load checkpoint {ckpt_to_load}: {e}. Starting fresh.")
            start_epoch, best_iou, history = 0, 0.0, []

    if start_epoch >= n_epochs:
        logger.info(f"[{variant_name}] Training already complete ({start_epoch}/{n_epochs} epochs).")
        return str(best_ckpt if best_ckpt.exists() else latest_ckpt), best_iou

    unfreeze_epoch = t_cfg.get("unfreeze_encoder_epoch", 999)
    unfreeze_last_n = t_cfg.get("unfreeze_last_n_blocks")

    def _apply_unfreeze():
        if unfreeze_last_n:
            for p in model.encoder.parameters():
                p.requires_grad = False
            model.unfreeze_encoder_last_n(int(unfreeze_last_n))
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Initial freeze state (accounting for a resume past the unfreeze epoch)
    for p in model.encoder.parameters():
        p.requires_grad = encoder_active
    encoder_unfrozen = encoder_active
    if start_epoch >= unfreeze_epoch:
        _apply_unfreeze()
        encoder_unfrozen = True
        logger.info(f"[{variant_name}] Resuming at epoch {start_epoch} with encoder unfrozen")

    for epoch in range(start_epoch, n_epochs):
        if epoch == unfreeze_epoch and not encoder_unfrozen:
            _apply_unfreeze()
            encoder_unfrozen = True
            scope = f"last {unfreeze_last_n} blocks" if unfreeze_last_n else "all blocks"
            logger.info(f"[{variant_name}] Epoch {epoch}: encoder unfrozen ({scope})")

        # Learning rate for this epoch
        if legacy_scheduler is None:
            scale = compute_lr_scale(
                epoch, n_epochs, t_cfg.get("warmup_epochs", 0), t_cfg.get("min_lr_factor", 0.01)
            )
            optimizer.param_groups[0]["lr"] = base_decoder_lr * scale
            optimizer.param_groups[1]["lr"] = (base_encoder_lr * scale) if encoder_unfrozen else 0.0
        elif encoder_unfrozen:
            optimizer.param_groups[1]["lr"] = max(optimizer.param_groups[1]["lr"], base_encoder_lr)
        current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"[{variant_name}] Ep {epoch+1}/{n_epochs}", leave=False)
        for step, batch in enumerate(pbar):
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with _autocast(enabled=torch.cuda.is_available()):
                out = model(imgs)
                loss = criterion(out, masks) / t_cfg["accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % t_cfg["accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), t_cfg["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step_loss = loss.item() * t_cfg["accumulation_steps"]
            total_loss += step_loss
            pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{current_lr:.2e}"})

        # Flush any tail gradients when the epoch length is not a multiple of
        # accumulation_steps, so the last partial cycle is not silently dropped.
        if len(train_loader) % t_cfg["accumulation_steps"] != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), t_cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if legacy_scheduler is not None:
            legacy_scheduler.step()

        # Validate on the held-out validation split (never the test split)
        val_metrics = evaluate_model(
            model, val_loader, device, num_classes, class_names, desc=f"val ep{epoch+1}"
        )
        miou = val_metrics["mean_iou"]
        train_loss = total_loss / max(len(train_loader), 1)
        logger.info(
            f"[{variant_name}] Epoch {epoch+1}/{n_epochs} | Loss={train_loss:.4f} | "
            f"val mIoU={miou:.4f} | val mIoU(defect)={val_metrics['mean_iou_defect']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_mean_iou": miou,
            "val_mean_iou_defect": val_metrics["mean_iou_defect"],
            "val_mean_dice": val_metrics["mean_dice"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "lr": current_lr,
        })

        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_iou": max(best_iou, miou),
            "config": config,
            "history": history,
            "variant": variant_name,
        }
        if legacy_scheduler is not None:
            ckpt_dict["scheduler_state_dict"] = legacy_scheduler.state_dict()

        torch.save(ckpt_dict, latest_ckpt)
        del ckpt_dict
        if miou > best_iou:
            best_iou = miou
            # Copy the file just written rather than serialising the whole
            # state a second time. With an unfrozen ViT-B the checkpoint is
            # ~1.1 GB, and re-serialising it doubles peak host RAM on exactly
            # the epochs that already carry the most memory pressure. The
            # stored best_iou is already max(best_iou, miou) == miou here.
            shutil.copyfile(latest_ckpt, best_ckpt)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if mongo_syncer:
            try:
                device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                mongo_syncer.sync_heartbeat(
                    status="RUNNING",
                    variant=variant_name,
                    current_epoch=epoch + 1,
                    total_epochs=n_epochs,
                    loss=train_loss,
                    val_iou=miou,
                    device_info=device_name,
                    run_id=run_id,
                    dry_run=dry_run_sync,
                )
            except Exception as e:
                logger.debug(f"[{variant_name}] heartbeat notice: {e}")

        gc.collect()

    logger.info(f"[{variant_name}] Training complete. Best val mIoU={best_iou:.4f}")

    _release_model(model)
    del optimizer, scaler, train_loader, val_loader, train_ds, val_ds
    gc.collect()

    return str(best_ckpt if best_ckpt.exists() else latest_ckpt), best_iou


def run_evaluation_only(
    checkpoint_path: str,
    config: dict,
    splits: Dict[str, Any],
    also_eval_val: bool = True,
) -> dict:
    """Load a checkpoint and score it on the held-out test split."""
    from data.augmentation import get_val_transform
    from models.model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" in ckpt and ckpt["config"]:
        # Model geometry must come from the checkpoint or the state dict will
        # not load; everything else stays as configured for this run.
        config = deep_update(config, {"model": ckpt["config"].get("model", {})})

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_classes = config["model"]["num_classes"]
    class_names = CLASS_NAMES[:num_classes]
    val_tf = get_val_transform(config)

    test_ds = make_dataset(splits["test"], config, val_tf, False)
    test_loader = make_loader(test_ds, config, shuffle=False, batch_size=4)
    results = evaluate_model(model, test_loader, device, num_classes, class_names, desc="test")

    if also_eval_val and splits["val"]["files"]:
        val_ds = make_dataset(splits["val"], config, val_tf, False)
        val_loader = make_loader(val_ds, config, shuffle=False, batch_size=4)
        val_results = evaluate_model(model, val_loader, device, num_classes, class_names, desc="val")
        # Reported alongside the test numbers so selection quality is visible.
        for k, v in val_results.items():
            results[f"val_{k}"] = v
        del val_ds, val_loader

    results["checkpoint_epoch"] = ckpt.get("epoch", -1)
    results["selected_on"] = "validation_split"
    results["best_val_iou_at_selection"] = ckpt.get("best_iou", None)
    if ckpt.get("history"):
        results["epochs_trained"] = len(ckpt["history"])

    _release_model(model)
    del test_ds, test_loader, ckpt
    return results


def print_results_table(all_results: list):
    cols = (
        f"{'Variant':<26} {'mIoU':>7} {'mIoU_d':>7} {'Dice':>7} {'PixAcc':>7} "
        f"{'Dust':>7} {'RunDn':>7} {'Scr':>7} {'fwIoU':>7} {'ms/img':>7}"
    )
    print("\n" + "=" * len(cols))
    print("ABLATION STUDY RESULTS  (mIoU_d = defect-only mean, background excluded)")
    print("=" * len(cols))
    print(cols)
    print("-" * len(cols))

    def f(v):
        if v is None:
            return "    n/a"
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return "    n/a"
        return "    nan" if np.isnan(fv) else f"{fv:>7.4f}"

    for r in all_results:
        name = r["variant"][:25]
        m = r.get("metrics") or {}
        if not m:
            print(f"{name:<26} {'FAILED: ' + str(r.get('error', 'no metrics'))[:60]}")
            continue
        print(
            f"{name:<26} {f(m.get('mean_iou'))} {f(m.get('mean_iou_defect'))} "
            f"{f(m.get('mean_dice'))} {f(m.get('pixel_accuracy'))} "
            f"{f(m.get('iou_Dust'))} {f(m.get('iou_RunDown'))} {f(m.get('iou_Scratch'))} "
            f"{f(m.get('frequency_weighted_iou'))} {float(m.get('avg_inference_ms', 0)):>7.2f}"
        )
    print("=" * len(cols))

    # Surface classes that cannot be measured, so a low mean is not mistaken
    # for a low-performing model.
    zero_support = set()
    thin_support = {}
    for r in all_results:
        m = r.get("metrics") or {}
        for n in CLASS_NAMES:
            px = m.get(f"support_px_{n}")
            if px == 0:
                zero_support.add(n)
            elif isinstance(px, int) and 0 < px < 20000:
                thin_support[n] = px
    if zero_support:
        print(f"  NOTE: the test split has NO ground-truth pixels for "
              f"{', '.join(sorted(zero_support))}. Any score shown for those classes is "
              f"pure false positives; see mean_iou_gt_present for the mean without them.")
    for n, px in sorted(thin_support.items()):
        print(f"  NOTE: only {px:,} ground-truth pixels for {n} in the test split — "
              f"its IoU is a high-variance estimate, read the mean with that in mind.")


# ---------------------------------------------------------------------------
# Subprocess isolation
# ---------------------------------------------------------------------------

def _child_argv(args, variant_name: str) -> List[str]:
    """Build the argv that runs exactly one variant in a fresh interpreter."""
    argv = [sys.executable, str(Path(__file__).resolve()),
            "--mode", args.mode,
            "--variants", variant_name,
            "--output", args.output,
            "--no-isolate",
            "--child"]
    if args.epochs is not None:
        argv += ["--epochs", str(args.epochs)]
    if args.no_resume:
        argv.append("--no-resume")
    if args.force_resume:
        argv.append("--force-resume")
    if args.no_sync:
        argv.append("--no-sync")
    if args.dry_run_sync:
        argv.append("--dry-run-sync")
    if args.num_workers is not None:
        argv += ["--num-workers", str(args.num_workers)]
    if args.image_size is not None:
        argv += ["--image-size", str(args.image_size)]
    if args.resize_mode:
        argv += ["--resize-mode", args.resize_mode]
    if args.cache_dir:
        argv += ["--cache-dir", args.cache_dir]
    if args.scheduler:
        argv += ["--scheduler", args.scheduler]
    if args.batch_size is not None:
        argv += ["--batch-size", str(args.batch_size)]
    if args.val_fraction is not None:
        argv += ["--val-fraction", str(args.val_fraction)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.with_pretrained_baseline:
        argv.append("--with-pretrained-baseline")
    if args.data_root:
        argv += ["--data-root", args.data_root]
    argv += ["--run-id", args.run_id]
    return argv


def run_isolated(args, variants_to_run: List[dict]) -> None:
    """Run each variant in its own process.

    A CUDA OOM, a host-RAM OOM, or a hard interpreter crash in one variant then
    costs exactly that variant instead of the whole study, and all of its
    memory is returned to the OS before the next one starts. This is the direct
    fix for the run that died in A9 and took A10-A13 with it.
    """
    logger.info(
        f"Isolated mode: running {len(variants_to_run)} variant(s), one subprocess each. "
        f"run_id={args.run_id}"
    )
    output_path = Path(args.output)

    for variant in variants_to_run:
        name = variant["name"]
        logger.info(f"\n{'='*70}\nLaunching subprocess for: {name}\n{'='*70}")
        argv = _child_argv(args, name)
        try:
            proc = subprocess.run(argv, cwd=str(Path(__file__).resolve().parent))
            rc = proc.returncode
        except KeyboardInterrupt:
            logger.warning("Interrupted by user; stopping the study.")
            raise
        except Exception as e:
            rc = -1
            logger.error(f"[{name}] Could not launch subprocess: {e}")

        if rc != 0:
            logger.error(f"[{name}] Subprocess exited with code {rc}.")
            # Record the failure so the variant is visible as failed rather
            # than silently missing from the table.
            _record_failure(output_path, variant,
                            f"subprocess exited with code {rc} (likely OOM or hard crash)",
                            args)

    # Final aggregated view from whatever the children wrote.
    all_results = _load_results(output_path)
    if all_results:
        print_results_table(_sorted_results(all_results))
    logger.info(f"All ablation results are in {args.output}")


def _record_failure(output_path: Path, variant: dict, message: str, args) -> None:
    results = _load_results(output_path)

    # If the child already wrote its own diagnosis before dying, keep it -- a
    # real traceback message is far more useful than "exit code 1".
    existing = next((r for r in results if r.get("variant") == variant["name"]), None)
    if existing and existing.get("error") and existing.get("run_id") == args.run_id:
        logger.info(f"[{variant['name']}] Child reported: {existing['error']}")
        return

    results = [r for r in results if r.get("variant") != variant["name"]]
    results.append({
        "variant": variant["name"],
        "description": variant["description"],
        "metrics": {},
        "error": message,
        "run_id": args.run_id,
    })
    _save_results(output_path, results)

    if not args.no_sync:
        try:
            from upload_to_cloud import MongoDBAtlasSync
            syncer = MongoDBAtlasSync()
            syncer.sync_variant(results[-1], source_file=output_path.name,
                                run_id=args.run_id, dry_run=args.dry_run_sync)
        except Exception as e:
            logger.debug(f"Failure sync notice: {e}")


def _load_results(output_path: Path) -> List[dict]:
    if not output_path.exists():
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"Could not read {output_path}: {e}")
        return []


def _save_results(output_path: Path, results: List[dict]) -> None:
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, output_path)


def _variant_order(name: str) -> Tuple[int, str]:
    head = name.split("_")[0]
    try:
        return (int(head[1:]), name)
    except (ValueError, IndexError):
        return (9999, name)


def _sorted_results(results: List[dict]) -> List[dict]:
    return sorted(results, key=lambda r: _variant_order(r.get("variant", "")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DINOv2-DenseFPN-UNet ablation study runner")
    p.add_argument("--mode", choices=["eval_only", "quick", "full"], default="quick")
    p.add_argument("--variants", nargs="*", default=None, help="Variant names to run (default: all)")
    p.add_argument("--epochs", type=int, default=None, help="Override ablation epoch count")
    p.add_argument("--output", type=str, default="ablation_results.json")
    p.add_argument("--no-resume", action="store_true", help="Disable resuming from checkpoints/previous results")
    p.add_argument("--force-resume", action="store_true",
                   help="Resume even from checkpoints produced under an older protocol (not recommended)")
    p.add_argument("--no-sync", action="store_true", help="Disable continuous MongoDB Atlas sync")
    p.add_argument("--dry-run-sync", action="store_true", help="Preview MongoDB sync without network calls")

    p.add_argument("--no-isolate", action="store_true",
                   help="Run every variant in this process instead of one subprocess each")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--run-id", type=str, default=None,
                   help="Tag written to every result/heartbeat document (default: generated)")

    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers. 0 decodes 1081 JPEGs on the main thread each epoch; "
                        "4-6 is a large speedup on the RTX 3060 box")
    p.add_argument("--image-size", type=int, default=None,
                   help="Input side length; must be a multiple of 14 for ViT-B/14 (default 518)")
    p.add_argument("--resize-mode", choices=["stretch", "letterbox"], default=None,
                   help="'stretch' (default) squashes 1440x1080 to square; 'letterbox' preserves aspect ratio")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Cache resized image/mask arrays here to skip repeated JPEG decoding")
    p.add_argument("--scheduler", choices=["cosine_warmup", "warm_restarts"], default=None,
                   help="'cosine_warmup' (default) or 'warm_restarts' to reproduce earlier runs")
    p.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    p.add_argument("--val-fraction", type=float, default=None,
                   help="Fraction of the training pool held out for checkpoint selection (default 0.1)")
    p.add_argument("--seed", type=int, default=None, help="Global seed (default 42)")
    p.add_argument("--with-pretrained-baseline", action="store_true",
                   help="Also score the deployed 200-epoch checkpoint as A0 (reference row, not protocol-matched)")
    p.add_argument("--data-root", type=str, default=None,
                   help="Dataset root containing train/ and test/ (default ./data/data)")
    p.add_argument("--print-splits", action="store_true", help="Print the split summary and exit")
    return p


def apply_cli_overrides(args) -> None:
    """Fold CLI flags into BASE_CONFIG so every variant inherits them."""
    if args.num_workers is not None:
        BASE_CONFIG["data"]["num_workers"] = args.num_workers
    if args.image_size is not None:
        if args.image_size % 14 != 0:
            raise SystemExit(
                f"--image-size must be a multiple of 14 for ViT-B/14 patches, got {args.image_size}"
            )
        BASE_CONFIG["data"]["image_size"] = args.image_size
    if args.resize_mode is not None:
        BASE_CONFIG["data"]["resize_mode"] = args.resize_mode
    if args.cache_dir is not None:
        BASE_CONFIG["data"]["cache_dir"] = args.cache_dir
    if args.val_fraction is not None:
        BASE_CONFIG["data"]["val_fraction"] = args.val_fraction
    if args.scheduler is not None:
        BASE_CONFIG["training"]["scheduler"] = args.scheduler
    if args.batch_size is not None:
        BASE_CONFIG["training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        BASE_CONFIG["training"]["seed"] = args.seed
    if args.data_root is not None:
        BASE_CONFIG["paths"]["data_root"] = args.data_root

    BASE_CONFIG["_protocol"] = (
        f"val_split_selection|{BASE_CONFIG['training']['scheduler']}|"
        f"img{BASE_CONFIG['data']['image_size']}|{BASE_CONFIG['data']['resize_mode']}|"
        f"metrics_v2"
    )


def main():
    args = build_parser().parse_args()
    if not args.run_id:
        args.run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    apply_cli_overrides(args)
    set_seed(BASE_CONFIG["training"]["seed"])

    variants_to_run = list(ABLATION_VARIANTS)
    if args.with_pretrained_baseline:
        variants_to_run = [PRETRAINED_REFERENCE] + variants_to_run
    if args.variants:
        wanted = set(args.variants)
        variants_to_run = [v for v in variants_to_run if v["name"] in wanted]
        unknown = wanted - {v["name"] for v in variants_to_run}
        if unknown:
            known = ", ".join(v["name"] for v in ABLATION_VARIANTS)
            raise SystemExit(f"Unknown variant(s): {', '.join(sorted(unknown))}\nKnown: {known}")

    # --- device report ---
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(
            f"Using NVIDIA GPU: {props.name} "
            f"({props.total_memory / 1024**3:.1f} GB, CUDA {torch.version.cuda})"
        )
    else:
        logger.warning(
            "\n" + "=" * 70 + "\n"
            "CUDA NOT DETECTED: Running on CPU (will be very slow)!\n"
            "If you have an NVIDIA GPU, install CUDA-enabled PyTorch with:\n"
            "  pip uninstall -y torch torchvision\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            + "=" * 70
        )

    # --- splits (shared by every variant) ---
    data_root = Path(BASE_CONFIG["paths"]["data_root"])
    splits = build_splits(
        data_root,
        val_fraction=BASE_CONFIG["data"]["val_fraction"],
        seed=BASE_CONFIG["training"]["seed"],
    )
    s = splits["summary"]
    logger.info(
        f"Splits | train={s['n_train']} val={s['n_val']} test={s['n_test']} "
        f"(pooled {s['pooled_valid_images']} images from valid/ into the training pool; "
        f"val is stratified, seed={s['split_seed']})"
    )
    logger.info(f"  images containing each class — train: {s['train_images_with_class']}")
    logger.info(f"  images containing each class —   val: {s['val_images_with_class']}")
    logger.info(f"  images containing each class —  test: {s['test_images_with_class']}")

    if args.print_splits:
        print(json.dumps(s, indent=2))
        return

    # --- isolated mode: hand each variant to a fresh interpreter ---
    if not args.no_isolate and not args.child:
        run_isolated(args, variants_to_run)
        return

    # --- in-process execution (one variant per call when isolated) ---
    mongo_syncer = None
    if not args.no_sync:
        try:
            from upload_to_cloud import MongoDBAtlasSync
            mongo_syncer = MongoDBAtlasSync()
            logger.info("Continuous MongoDB Atlas Sync initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize MongoDB Atlas sync: {e}")

    output_path = Path(args.output)
    all_results = [] if args.no_resume else _load_results(output_path)
    completed = {
        r["variant"] for r in all_results
        if r.get("variant") and r.get("metrics")
        and r.get("protocol") == BASE_CONFIG.get("_protocol")
    }

    variant_pbar = tqdm(variants_to_run, desc="Overall Ablation Progress",
                        disable=len(variants_to_run) <= 1)
    for variant in variant_pbar:
        name = variant["name"]
        variant_pbar.set_description(f"Running: {name}")

        epochs = args.epochs or (20 if args.mode == "quick" else 50)

        if not args.no_resume and name in completed:
            save_dir = Path(BASE_CONFIG["paths"]["save_dir"]) / name
            ckpt_to_check = next(
                (c for c in (save_dir / "latest.pth", save_dir / "best.pth") if c.exists()), None
            )
            if ckpt_to_check:
                try:
                    meta = torch.load(ckpt_to_check, map_location="cpu", weights_only=False)
                    if meta.get("epoch", -1) + 1 >= epochs:
                        logger.info(
                            f"Variant '{name}' already complete "
                            f"({meta.get('epoch', -1) + 1}/{epochs} epochs, same protocol). Skipping."
                        )
                        del meta
                        continue
                    del meta
                except Exception:
                    pass

        logger.info(f"\n{'='*60}\nRunning: {name}\n{variant['description']}\n{'='*60}")
        config = build_config(variant)

        try:
            ckpt_path = variant.get("checkpoint")
            if args.mode == "eval_only":
                if not ckpt_path or not Path(ckpt_path).exists():
                    logger.warning(f"No checkpoint for {name}, skipping")
                    continue
                metrics = run_evaluation_only(ckpt_path, config, splits)
            elif ckpt_path and Path(ckpt_path).exists():
                logger.info(f"Pre-trained checkpoint found: {ckpt_path} (reference row)")
                metrics = run_evaluation_only(ckpt_path, config, splits)
                metrics["selected_on"] = "external_pretrained_checkpoint"
            else:
                ckpt_path, best_val_iou = train_variant(
                    config, name, splits, num_epochs=epochs, resume=not args.no_resume,
                    force_resume=args.force_resume,
                    mongo_syncer=mongo_syncer, dry_run_sync=args.dry_run_sync,
                    run_id=args.run_id,
                )
                metrics = run_evaluation_only(ckpt_path, config, splits)

            result = {
                "variant": name,
                "description": variant["description"],
                "metrics": metrics,
                "run_id": args.run_id,
                "protocol": BASE_CONFIG.get("_protocol"),
                "epochs_budget": epochs,
                "split_summary": splits["summary"],
                "config": config,
            }

            all_results = [r for r in _load_results(output_path) if r.get("variant") != name]
            all_results.append(result)
            _save_results(output_path, _sorted_results(all_results))
            logger.info(
                f"[{name}] Saved to {args.output} | test mIoU={metrics.get('mean_iou', 0):.4f} "
                f"| test mIoU(defect)={metrics.get('mean_iou_defect', 0):.4f}"
            )

            if mongo_syncer:
                try:
                    mongo_syncer.sync_variant(result, source_file=args.output,
                                              run_id=args.run_id, dry_run=args.dry_run_sync)
                except Exception as sync_err:
                    logger.warning(f"[{name}] Cloud sync notice: {sync_err}. Saved locally.")

        except Exception as e:
            logger.error(f"[{name}] FAILED: {e}", exc_info=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            err_res = {
                "variant": name,
                "description": variant["description"],
                "metrics": {},
                "error": str(e),
                "run_id": args.run_id,
                "protocol": BASE_CONFIG.get("_protocol"),
            }
            all_results = [r for r in _load_results(output_path) if r.get("variant") != name]
            all_results.append(err_res)
            _save_results(output_path, _sorted_results(all_results))

            if mongo_syncer:
                try:
                    mongo_syncer.sync_variant(err_res, source_file=args.output,
                                              run_id=args.run_id, dry_run=args.dry_run_sync)
                except Exception as sync_err:
                    logger.warning(f"[{name}] Cloud sync notice: {sync_err}. Saved locally.")
            # Re-raise in child mode so the parent sees a non-zero exit code.
            if args.child:
                raise
        finally:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not args.child:
        print_results_table(_sorted_results(_load_results(output_path)))
        logger.info(f"All ablation results updated in {args.output}")

        if mongo_syncer:
            try:
                device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                mongo_syncer.sync_heartbeat(status="COMPLETED", device_info=device_name,
                                            run_id=args.run_id, dry_run=args.dry_run_sync)
            except Exception:
                pass

    if mongo_syncer:
        try:
            mongo_syncer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
