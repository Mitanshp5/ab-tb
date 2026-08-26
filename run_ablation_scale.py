"""
DEPRECATED — the scale ablation is now part of the main study.

This script existed because `models/encoder.py` could not handle
`skip_layers=[11]` or `[7,11]` (it raised KeyError 'mid' / 'deep'), so A2 and A3
failed in the main run and were re-run separately under the names
`A1_fresh_local`, `A2_single_scale_fixed` and `A3_two_scale_fixed`.

That encoder bug is fixed: missing scales are filled with zero tensors and keys
are assigned from the end of the list. A1/A2/A3 in `run_ablation.py` already
cover exactly the same three configurations:

    A1_full_model     skip_layers=[3, 7, 11]
    A2_single_scale   skip_layers=[11]
    A3_two_scale      skip_layers=[7, 11]

Running this script in parallel with the main study produced duplicate
documents under different variant names. Because documents are upserted by
variant name, those duplicates are never overwritten and sit on the dashboard
as if they were study rows. Clean them up with:

    python upload_to_cloud.py --purge-stale

This file now forwards to the main runner so existing commands and the README
keep working. It will be removed in a future revision.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Old name -> the main-study variant that supersedes it.
SUPERSEDED_BY = {
    "A1_fresh_local": "A1_full_model",
    "A2_single_scale_fixed": "A2_single_scale",
    "A3_two_scale_fixed": "A3_two_scale",
}


def main():
    parser = argparse.ArgumentParser(
        description="DEPRECATED wrapper — forwards the scale ablation to run_ablation.py"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", type=str, default="ablation_results_scale.json")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    args, extra = parser.parse_known_args()

    logger.warning(
        "run_ablation_scale.py is DEPRECATED. The scale variants are now A1/A2/A3 "
        "of the main study; the encoder bug that required a separate run is fixed."
    )
    for old, new in SUPERSEDED_BY.items():
        logger.warning(f"  {old:<24} -> {new}")
    logger.warning(
        "Any %s documents already in MongoDB are stale; remove them with "
        "'python upload_to_cloud.py --purge-stale'.", "/".join(SUPERSEDED_BY)
    )

    argv = [
        sys.executable, str(Path(__file__).resolve().parent / "run_ablation.py"),
        "--mode", "full",
        "--epochs", str(args.epochs),
        "--variants", "A1_full_model", "A2_single_scale", "A3_two_scale",
        "--output", args.output,
    ]
    if args.no_sync:
        argv.append("--no-sync")
    if args.dry_run_sync:
        argv.append("--dry-run-sync")
    if args.num_workers is not None:
        argv += ["--num-workers", str(args.num_workers)]
    if args.data_root:
        argv += ["--data-root", args.data_root]
    argv += extra

    logger.info("Forwarding to: %s", " ".join(argv[1:]))
    raise SystemExit(subprocess.run(argv).returncode)


if __name__ == "__main__":
    main()
