"""
Ablation study runner for DINOv2-DenseFPN-UNet.

Each variant modifies ONE component vs. the full model.
Uses the trained checkpoints if available, else trains from scratch
for ABLATION_EPOCHS (fast-mode) on the local dataset.

Usage:
    python run_ablation.py --mode eval_only   # evaluate existing checkpoints only
    python run_ablation.py --mode full        # train + eval all variants
    python run_ablation.py --mode quick       # 20-epoch training for ablations
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path so local modules (data, models, loss, etc.) can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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
        "num_epochs": 15,           # fast ablation epochs (frozen encoder, ~55 min each)
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_epochs": 3,
        "gradient_clip": 1.0,
        "accumulation_steps": 1,
        "unfreeze_encoder_epoch": 999,  # keep frozen for speed
        "encoder_lr": 1e-5,
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
    },
    "augmentation": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.3,
        "rotation_limit": 15,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "gaussian_blur_p": 0.3,
        "clahe": 0.3,
        "elastic_transform_p": 0.3,
    },
    "paths": {
        "data_root": "./data/data",
        "save_dir": "./checkpoints/ablation",
        "log_dir": "./logs/ablation",
    },
}

ABLATION_VARIANTS = [
    {
        "name": "A1_full_model",
        "description": "Full model — DINOv2-ViT-B/14 + DenseFPN-UNet (3-scale) + combined loss",
        "checkpoint": "checkpoints/black_best_model.pth",  # use pre-trained
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
        "name": "A4_frozen_encoder",
        "description": "Fully frozen encoder throughout (no encoder fine-tuning)",
        "checkpoint": None,
        "config_overrides": {
            "model": {"encoder_frozen": True},
            "training": {"unfreeze_encoder_epoch": 999},
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
        "description": "ViT-B encoder unfrozen from epoch 5 (aggressive fine-tuning)",
        "checkpoint": None,
        "config_overrides": {
            "model": {"encoder_frozen": True},
            "training": {
                "unfreeze_encoder_epoch": 5,
                "encoder_lr": 5e-6,
                "batch_size": 4,
                "accumulation_steps": 2,
            },
        },
    },
    {
        "name": "A10_no_augmentation",
        "description": "No augmentation (geometric/photometric transforms disabled)",
        "checkpoint": None,
        "config_overrides": {
            "augmentation": {
                "horizontal_flip": 0.0,
                "vertical_flip": 0.0,
                "rotation_limit": 0,
                "brightness_limit": 0.0,
                "contrast_limit": 0.0,
                "gaussian_blur_p": 0.0,
                "clahe": 0.0,
                "elastic_transform_p": 0.0,
            }
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


def find_data_dirs(data_root: Path):
    """Find train/test image+label dirs, handling 'valid' vs 'val' naming and nested data dirs."""
    # Automatically resolve nested 'data/data' vs 'data'
    if not (data_root / "train").exists() and (data_root / "data" / "train").exists():
        data_root = data_root / "data"
    elif not (data_root / "train").exists() and (data_root.parent / "train").exists():
        data_root = data_root.parent

    # Train
    for name in ["train"]:
        d = data_root / name / "images"
        if d.exists():
            train_img = str(d)
            train_lbl = str(data_root / name / "labels")
            break
    else:
        raise FileNotFoundError(f"No train/images found under {data_root} or {data_root / 'data'}. Please ensure train/images/ and train/labels/ exist.")

    # Test / val
    for split in ["test", "val", "valid"]:
        d = data_root / split / "images"
        if d.exists():
            test_img = str(d)
            test_lbl = str(data_root / split / "labels")
            break
    else:
        test_img, test_lbl = train_img, train_lbl  # fallback

    return train_img, train_lbl, test_img, test_lbl


@torch.no_grad()
def evaluate_model(model, test_loader, device, num_classes, class_names):
    """Run evaluation, return metrics dict."""
    from utils.metrics import SegmentationMetrics
    import numpy as np

    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, class_names=class_names)
    times = []

    for batch in tqdm(test_loader, desc="Evaluating", leave=False):
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        t0 = time.time()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(imgs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        pred = out.argmax(dim=1)
        metrics.update(pred, masks)

    results = metrics.compute()
    results["avg_inference_ms"] = np.mean(times) * 1000 / test_loader.batch_size
    return results


def train_variant(
    config: dict,
    variant_name: str,
    num_epochs: int = None,
    resume: bool = True,
    mongo_syncer: Optional[Any] = None,
    dry_run_sync: bool = False,
):
    """Train a single ablation variant and return checkpoint path (supports resuming)."""
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    from data.dataset import DefectDataset, get_dataloaders
    from data.augmentation import get_train_transform, get_val_transform
    from models.model import build_model
    from loss.losses import build_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {variant_name} on {device}")

    data_root = Path(config["paths"]["data_root"])
    train_img, train_lbl, test_img, test_lbl = find_data_dirs(data_root)

    train_tf = get_train_transform(config)
    val_tf = get_val_transform(config)

    model_cfg = config["model"]
    num_classes = model_cfg["num_classes"]

    train_ds = DefectDataset(
        image_dir=train_img,
        label_dir=train_lbl,
        image_size=config["data"]["image_size"],
        transform=train_tf,
        num_classes=num_classes - 1,
        is_training=True,
    )
    val_ds = DefectDataset(
        image_dir=test_img,
        label_dir=test_lbl,
        image_size=config["data"]["image_size"],
        transform=val_tf,
        num_classes=num_classes - 1,
        is_training=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config).to(device)
    criterion = build_loss(config)
    t_cfg = config["training"]
    n_epochs = num_epochs or t_cfg["num_epochs"]

    # Param groups: separate encoder LR
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for p in model.parameters() if not any(p is ep for ep in encoder_params)]
    param_groups = [
        {"params": decoder_params, "lr": t_cfg["learning_rate"]},
        {"params": encoder_params, "lr": t_cfg.get("encoder_lr", t_cfg["learning_rate"]) if not model_cfg.get("encoder_frozen", True) else 0.0},
    ]
    optimizer = AdamW(param_groups, weight_decay=t_cfg["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(n_epochs // 3, 5), T_mult=1)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    save_dir = Path(config["paths"]["save_dir"]) / variant_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    best_ckpt = save_dir / "best.pth"
    latest_ckpt = save_dir / "latest.pth"
    class_names = ["Background", "Dust", "RunDown", "Scratch"][:num_classes]

    # Resume support
    start_epoch = 0
    ckpt_to_load = latest_ckpt if latest_ckpt.exists() else (best_ckpt if best_ckpt.exists() else None)

    if resume and ckpt_to_load:
        try:
            logger.info(f"[{variant_name}] Found existing checkpoint: {ckpt_to_load.name}. Loading state...")
            ckpt = torch.load(ckpt_to_load, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            best_iou = ckpt.get("best_iou", 0.0)
            start_epoch = ckpt.get("epoch", -1) + 1
            logger.info(f"[{variant_name}] Resuming from epoch {start_epoch + 1}/{n_epochs} (Best mIoU so far: {best_iou:.4f})")
        except Exception as e:
            logger.warning(f"[{variant_name}] Failed to load checkpoint {ckpt_to_load}: {e}. Starting fresh.")
            start_epoch = 0

    if start_epoch >= n_epochs:
        logger.info(f"[{variant_name}] Training already completed ({start_epoch}/{n_epochs} epochs). Skipping training loop.")
        return str(best_ckpt if best_ckpt.exists() else latest_ckpt), best_iou

    # Freeze/unfreeze encoder (account for resuming after unfreeze_epoch)
    unfreeze_epoch = t_cfg.get("unfreeze_encoder_epoch", 999)
    should_be_unfrozen = (not model_cfg.get("encoder_frozen", True)) or (start_epoch >= unfreeze_epoch)
    for p in model.encoder.parameters():
        p.requires_grad = should_be_unfrozen
    if start_epoch >= unfreeze_epoch:
        param_groups[1]["lr"] = t_cfg.get("encoder_lr", t_cfg["learning_rate"])
        logger.info(f"[{variant_name}] Resuming at epoch {start_epoch} with encoder unfrozen")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for epoch in range(start_epoch, n_epochs):
        # Unfreeze encoder at specified epoch
        if epoch == unfreeze_epoch and start_epoch <= unfreeze_epoch:
            for p in model.encoder.parameters():
                p.requires_grad = True
            param_groups[1]["lr"] = t_cfg.get("encoder_lr", t_cfg["learning_rate"])
            logger.info(f"[{variant_name}] Epoch {epoch}: encoder unfrozen")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"[{variant_name}] Ep {epoch+1}/{n_epochs}", leave=False)
        for step, batch in enumerate(pbar):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with autocast(enabled=torch.cuda.is_available()):
                out = model(imgs)
                loss = criterion(out, masks) / t_cfg["accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % t_cfg["accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), t_cfg["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step_loss = loss.item() * t_cfg["accumulation_steps"]
            total_loss += step_loss
            pbar.set_postfix({"loss": f"{step_loss:.4f}"})

        scheduler.step()

        # Validate
        val_metrics = evaluate_model(model, val_loader, device, num_classes, class_names)
        miou = val_metrics["mean_iou"]
        logger.info(
            f"[{variant_name}] Epoch {epoch+1}/{n_epochs} | "
            f"Loss={total_loss/len(train_loader):.4f} | mIoU={miou:.4f}"
        )

        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_iou": max(best_iou, miou),
            "config": config,
        }

        # Always save latest state for epoch-level resume
        torch.save(ckpt_dict, latest_ckpt)

        if miou > best_iou:
            best_iou = miou
            ckpt_dict["best_iou"] = best_iou
            torch.save(ckpt_dict, best_ckpt)

        if mongo_syncer:
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            mongo_syncer.sync_heartbeat(
                status="RUNNING",
                variant=variant_name,
                current_epoch=epoch + 1,
                total_epochs=n_epochs,
                loss=total_loss / len(train_loader),
                val_iou=miou,
                device_info=device_name,
                dry_run=dry_run_sync,
            )

    logger.info(f"[{variant_name}] Training complete. Best mIoU={best_iou:.4f}")
    return str(best_ckpt if best_ckpt.exists() else latest_ckpt), best_iou


def run_evaluation_only(checkpoint_path: str, config: dict) -> dict:
    """Load checkpoint and evaluate on test set."""
    from data.dataset import DefectDataset
    from data.augmentation import get_val_transform
    from models.model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" in ckpt:
        config = deep_update(config, ckpt["config"])

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data_root = Path(config["paths"]["data_root"])
    # Handle nested data/data structure
    if not (data_root / "train").exists() and (data_root / "data" / "train").exists():
        data_root = data_root / "data"

    _, _, test_img, test_lbl = find_data_dirs(data_root)
    val_tf = get_val_transform(config)
    num_classes = config["model"]["num_classes"]

    test_ds = DefectDataset(
        image_dir=test_img,
        label_dir=test_lbl,
        image_size=config["data"]["image_size"],
        transform=val_tf,
        num_classes=num_classes - 1,
        is_training=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=False,
    )

    class_names = ["Background", "Dust", "RunDown", "Scratch"][:num_classes]
    results = evaluate_model(model, test_loader, device, num_classes, class_names)
    results["checkpoint_epoch"] = ckpt.get("epoch", -1)
    results["num_samples"] = len(test_ds)
    return results


def print_results_table(all_results: list):
    header = f"{'Variant':<30} {'mIoU':>7} {'Dice':>7} {'PixAcc':>7} {'Dust':>7} {'RunDown':>7} {'Scratch':>7} {'ms/img':>7}"
    print("\n" + "=" * len(header))
    print("ABLATION STUDY RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in all_results:
        name = r["variant"][:29]
        m = r["metrics"]
        print(
            f"{name:<30} "
            f"{m.get('mean_iou', 0):>7.4f} "
            f"{m.get('mean_dice', 0):>7.4f} "
            f"{m.get('pixel_accuracy', 0):>7.4f} "
            f"{m.get('iou_Dust', 0):>7.4f} "
            f"{m.get('iou_RunDown', 0):>7.4f} "
            f"{m.get('iou_Scratch', 0):>7.4f} "
            f"{m.get('avg_inference_ms', 0):>7.2f}"
        )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval_only", "quick", "full"], default="quick")
    parser.add_argument("--variants", nargs="*", default=None, help="Variant names to run (default: all)")
    parser.add_argument("--epochs", type=int, default=None, help="Override ablation epoch count")
    parser.add_argument("--output", type=str, default="ablation_results.json")
    parser.add_argument("--no-resume", action="store_true", help="Disable resuming from checkpoints/previous results")
    parser.add_argument("--no-sync", action="store_true", help="Disable continuous MongoDB Atlas sync")
    parser.add_argument("--dry-run-sync", action="store_true", help="Preview MongoDB Atlas document sync without making network calls")
    args = parser.parse_args()

    # Initialize real-time MongoDB Atlas syncer unless disabled
    mongo_syncer = None
    if not args.no_sync:
        try:
            from upload_to_cloud import MongoDBAtlasSync
            mongo_syncer = MongoDBAtlasSync()
            logger.info("Continuous MongoDB Atlas Sync initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize MongoDB Atlas sync: {e}")

    if torch.cuda.is_available():
        logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning(
            "\n" + "="*70 + "\n"
            "CUDA NOT DETECTED: Running on CPU (will be very slow)!\n"
            "If you have an NVIDIA GPU, install CUDA-enabled PyTorch with:\n"
            "  pip uninstall -y torch torchvision\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            + "="*70
        )

    variants_to_run = ABLATION_VARIANTS
    if args.variants:
        variants_to_run = [v for v in ABLATION_VARIANTS if v["name"] in args.variants]

    # Load existing results if available to allow table-level resume
    all_results = []
    completed_variant_names = set()
    output_path = Path(args.output)
    if not args.no_resume and output_path.exists():
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    all_results = existing_data
                    completed_variant_names = {r["variant"] for r in existing_data if "variant" in r and r.get("metrics")}
                    logger.info(f"Loaded {len(all_results)} existing variant results from {args.output}")
        except Exception as e:
            logger.warning(f"Could not load existing results file {args.output}: {e}")

    variant_pbar = tqdm(variants_to_run, desc="Overall Ablation Progress")
    for variant in variant_pbar:
        name = variant["name"]
        variant_pbar.set_description(f"Running: {name}")

        # Check if variant output already exists in JSON and checkpoint is complete
        if not args.no_resume and name in completed_variant_names:
            save_dir = Path(BASE_CONFIG["paths"]["save_dir"]) / name
            latest_ckpt = save_dir / "latest.pth"
            best_ckpt = save_dir / "best.pth"
            epochs = args.epochs or (20 if args.mode == "quick" else 50)
            
            # Check checkpoint epoch count if checkpoint exists
            ckpt_to_check = latest_ckpt if latest_ckpt.exists() else (best_ckpt if best_ckpt.exists() else None)
            if ckpt_to_check:
                try:
                    ckpt = torch.load(ckpt_to_check, map_location="cpu", weights_only=False)
                    if ckpt.get("epoch", -1) + 1 >= epochs:
                        logger.info(f"Variant '{name}' already fully trained ({ckpt.get('epoch', -1)+1}/{epochs} epochs) and recorded in {args.output}. Skipping.")
                        continue
                except Exception:
                    pass

        logger.info(f"\n{'='*60}\nRunning: {name}\n{variant['description']}\n{'='*60}")
        config = build_config(variant)

        try:
            if args.mode == "eval_only":
                ckpt_path = variant.get("checkpoint")
                if not ckpt_path or not Path(ckpt_path).exists():
                    logger.warning(f"No checkpoint for {name}, skipping")
                    continue
                metrics = run_evaluation_only(ckpt_path, config)

            elif args.mode in ("quick", "full"):
                ckpt_path = variant.get("checkpoint")
                if ckpt_path and Path(ckpt_path).exists():
                    # Pre-trained checkpoint available — evaluate directly
                    logger.info(f"Pre-trained checkpoint found: {ckpt_path}")
                    metrics = run_evaluation_only(ckpt_path, config)
                else:
                    # Train from scratch or resume (ablation variant)
                    epochs = args.epochs or (20 if args.mode == "quick" else 50)
                    ckpt_path, best_iou = train_variant(
                        config, name, num_epochs=epochs, resume=not args.no_resume,
                        mongo_syncer=mongo_syncer, dry_run_sync=args.dry_run_sync
                    )
                    metrics = run_evaluation_only(ckpt_path, config)

            result = {"variant": name, "description": variant["description"], "metrics": metrics}
            
            # Update all_results list
            all_results = [r for r in all_results if r.get("variant") != name]
            all_results.append(result)
            completed_variant_names.add(name)

            # Immediately save incremental results locally
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"[{name}] Saved updated results to {args.output} (mIoU={metrics.get('mean_iou', 0):.4f})")

            # Continuously sync variant result to MongoDB Atlas in real time
            if mongo_syncer:
                mongo_syncer.sync_variant(result, source_file=args.output, dry_run=args.dry_run_sync)

        except Exception as e:
            logger.error(f"[{name}] FAILED: {e}", exc_info=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            err_res = {"variant": name, "description": variant["description"], "metrics": {}, "error": str(e)}
            all_results = [r for r in all_results if r.get("variant") != name]
            all_results.append(err_res)
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)

            if mongo_syncer:
                mongo_syncer.sync_variant(err_res, source_file=args.output, dry_run=args.dry_run_sync)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_results_table(all_results)
    logger.info(f"All ablation results updated in {args.output}")

    if mongo_syncer:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        mongo_syncer.sync_heartbeat(status="COMPLETED", device_info=device_name, dry_run=args.dry_run_sync)


if __name__ == "__main__":
    main()
