"""
Follow-up ablation: scale variants (A1_fresh, A2_single, A3_two).
Run AFTER run_ablation.py completes.

Fixed encoder correctly handles skip_layers=[11] and [7,11] by filling
missing keys with zero tensors.

Usage:
    python run_ablation_scale.py --epochs 15 --output ablation_results_scale.json
"""

import json
import sys
from pathlib import Path

# Re-use helpers from run_ablation.py
sys.path.insert(0, str(Path(__file__).parent))
from run_ablation import (
    BASE_CONFIG, deep_update, build_config,
    find_data_dirs, evaluate_model, train_variant,
    run_evaluation_only, print_results_table
)

import argparse
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCALE_VARIANTS = [
    {
        "name": "A1_fresh_local",
        "description": "Full model trained from scratch on local dataset (3-scale baseline)",
        "checkpoint": None,  # Force fresh training
        "config_overrides": {
            "model": {"skip_layers": [3, 7, 11]},
        },
    },
    {
        "name": "A2_single_scale_fixed",
        "description": "Single-scale: only deep layer [11]; shallow+mid filled with zeros",
        "checkpoint": None,
        "config_overrides": {
            "model": {"skip_layers": [11]},
        },
    },
    {
        "name": "A3_two_scale_fixed",
        "description": "Two-scale: mid+deep [7,11]; shallow filled with zeros",
        "checkpoint": None,
        "config_overrides": {
            "model": {"skip_layers": [7, 11]},
        },
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", type=str, default="ablation_results_scale.json")
    parser.add_argument("--no-sync", action="store_true", help="Disable continuous MongoDB Atlas sync")
    parser.add_argument("--dry-run-sync", action="store_true", help="Preview MongoDB Atlas document sync without making network calls")
    args = parser.parse_args()

    mongo_syncer = None
    if not args.no_sync:
        try:
            from upload_to_cloud import MongoDBAtlasSync
            mongo_syncer = MongoDBAtlasSync()
            logger.info("Continuous MongoDB Atlas Sync initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize MongoDB Atlas sync: {e}")

    all_results = []

    for variant in SCALE_VARIANTS:
        name = variant["name"]
        logger.info(f"\n{'='*60}\nRunning: {name}\n{variant['description']}\n{'='*60}")
        config = build_config(variant)

        try:
            ckpt_path, best_iou = train_variant(config, name, num_epochs=args.epochs)
            metrics = run_evaluation_only(ckpt_path, config)
            result = {"variant": name, "description": variant["description"], "metrics": metrics}
            all_results.append(result)
            logger.info(f"[{name}] mIoU={metrics.get('mean_iou', 0):.4f}")

            if mongo_syncer:
                mongo_syncer.sync_variant(result, source_file=args.output, dry_run=args.dry_run_sync)
        except Exception as e:
            logger.error(f"[{name}] FAILED: {e}", exc_info=True)
            err_res = {"variant": name, "description": variant["description"], "metrics": {}, "error": str(e)}
            all_results.append(err_res)
            if mongo_syncer:
                mongo_syncer.sync_variant(err_res, source_file=args.output, dry_run=args.dry_run_sync)

    print_results_table(all_results)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Scale ablation results saved to {args.output}")


if __name__ == "__main__":
    main()
