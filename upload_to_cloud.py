#!/usr/bin/env python3
"""
MongoDB Atlas Results Sync for Ablation Study.

Continuously syncs ablation study results (metrics, IoUs, parameters)
to MongoDB Atlas in real-time as each variant completes.

Configuration is loaded from .env:
  MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
  MONGODB_TARGET=ablation_study.results
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mongodb_sync")


def load_env_file(env_path: Optional[Path] = None):
    """Load key-value pairs from .env file into os.environ if present."""
    path = env_path or (Path.cwd() / ".env")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v


class MongoDBAtlasSync:
    """Handles real-time syncing of ablation study results to MongoDB Atlas."""

    def __init__(self, uri: Optional[str] = None, target: Optional[str] = None):
        load_env_file()
        uri = uri or os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
        if not uri:
            user = os.environ.get("MONGODB_USER")
            password = os.environ.get("MONGODB_PASS") or os.environ.get("MONGODB_PASSWORD")
            host = os.environ.get("MONGODB_HOST") or os.environ.get("MONGODB_CLUSTER", "cluster0.mongodb.net")
            if user and password:
                uri = f"mongodb+srv://{user}:{password}@{host}/?retryWrites=true&w=majority"
        self.uri = uri
        self.target = target or os.environ.get("MONGODB_TARGET", "ablation_study.results")

    def parse_target(self) -> tuple[str, str]:
        target_str = self.target or "ablation_study.results"
        parts = target_str.split(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "ablation_study", parts[0] or "results"

    def _get_collection(self):
        if not self.uri:
            raise ValueError("MongoDB Connection URI is missing. Check .env or set MONGODB_URI.")
        try:
            import pymongo
        except ImportError:
            raise ImportError("Package 'pymongo' is not installed. Install via 'pip install pymongo dnspython'.")

        db_name, coll_name = self.parse_target()
        client = pymongo.MongoClient(self.uri)
        return client[db_name][coll_name], db_name, coll_name

    def sync_variant(self, variant_result: Dict[str, Any], source_file: str = "live_sync", dry_run: bool = False) -> bool:
        """Sync a single variant result document to MongoDB Atlas immediately."""
        db_name, coll_name = self.parse_target()
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        doc = {
            "_upload_meta": {
                "source_file": source_file,
                "synced_at": timestamp,
            }
        }
        doc.update(variant_result)

        if dry_run:
            logger.info(f"[DRY-RUN] Would upsert variant '{variant_result.get('variant')}' to MongoDB Atlas ({db_name}.{coll_name})")
            return True

        try:
            coll, db_n, coll_n = self._get_collection()
            variant_name = doc.get("variant")
            if variant_name:
                res = coll.replace_one({"variant": variant_name}, doc, upsert=True)
                action = "Inserted new" if res.upserted_id else "Updated existing"
                logger.info(f"[MongoDB Atlas Sync] {action} record for variant '{variant_name}' in {db_n}.{coll_n}")
            else:
                coll.insert_one(doc)
                logger.info(f"[MongoDB Atlas Sync] Inserted record in {db_n}.{coll_n}")
            return True
        except Exception as e:
            err_msg = str(e)
            if "dns" in err_msg.lower() or "nxdomain" in err_msg.lower() or "does not exist" in err_msg.lower():
                logger.error(
                    f"[MongoDB Atlas Sync Failed] DNS Lookup Error: The cluster domain in your MONGODB_URI is invalid or placeholder.\n"
                    f"-> Please open MongoDB Atlas (cloud.mongodb.net), click 'Connect' -> 'Drivers', copy your full URI (e.g. mongodb+srv://user:pass@cluster0.abcde.mongodb.net), and paste it into your .env file."
                )
            else:
                logger.error(f"[MongoDB Atlas Sync Failed] {e}")
            return False

    def sync_heartbeat(
        self,
        status: str,
        variant: Optional[str] = None,
        current_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        loss: Optional[float] = None,
        val_iou: Optional[float] = None,
        device_info: Optional[str] = None,
        error: Optional[str] = None,
        dry_run: bool = False
    ) -> bool:
        """Sync live heartbeat & current epoch progress to MongoDB Atlas."""
        db_name, coll_name = self.parse_target()
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        progress_pct = 0.0
        if current_epoch and total_epochs and total_epochs > 0:
            progress_pct = round((current_epoch / total_epochs) * 100, 1)

        doc = {
            "_id": "live_status_heartbeat",
            "type": "heartbeat",
            "status": status.upper(),
            "current_variant": variant,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "epoch_progress_pct": progress_pct,
            "latest_loss": round(loss, 4) if loss is not None else None,
            "latest_val_iou": round(val_iou, 4) if val_iou is not None else None,
            "device": device_info,
            "error": error,
            "last_heartbeat": timestamp
        }

        if dry_run:
            logger.info(f"[DRY-RUN] Heartbeat status '{status}':\n{json.dumps(doc, indent=2)}")
            return True

        try:
            coll, db_n, coll_n = self._get_collection()
            coll.replace_one({"_id": "live_status_heartbeat"}, doc, upsert=True)
            return True
        except Exception as e:
            logger.error(f"[Heartbeat Sync Failed] {e}")
            return False

    def get_live_status(self) -> Dict[str, Any]:
        """Fetch latest live status heartbeat document from MongoDB Atlas."""
        try:
            coll, _, _ = self._get_collection()
            doc = coll.find_one({"_id": "live_status_heartbeat"})
            if doc:
                doc.pop("_id", None)
                return doc
        except Exception as e:
            logger.error(f"Failed to fetch live status: {e}")
        return {"status": "UNKNOWN", "last_heartbeat": None}

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Fetch all completed variant documents from MongoDB Atlas."""
        try:
            coll, _, _ = self._get_collection()
            cursor = coll.find({"variant": {"$exists": True}})
            results = []
            for doc in cursor:
                doc.pop("_id", None)
                results.append(doc)
            return results
        except Exception as e:
            logger.error(f"Failed to fetch variant results: {e}")
            return []

    def sync_file(self, json_path: Path, dry_run: bool = False) -> bool:
        """Sync an entire ablation results JSON file to MongoDB Atlas."""
        if not json_path.exists():
            logger.warning(f"File {json_path} does not exist.")
            return False

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                success = True
                for item in data:
                    if not self.sync_variant(item, source_file=json_path.name, dry_run=dry_run):
                        success = False
                return success
            elif isinstance(data, dict):
                return self.sync_variant(data, source_file=json_path.name, dry_run=dry_run)
            return False
        except Exception as e:
            logger.error(f"Failed to sync {json_path}: {e}")
            return False


def sync_to_mongodb(json_files: Optional[List[str]] = None, dry_run: bool = False) -> bool:
    """Sync default or specified ablation JSON result files to MongoDB Atlas."""
    syncer = MongoDBAtlasSync()
    
    if json_files:
        files = [Path(f) for f in json_files]
    else:
        files = list(Path.cwd().glob("ablation_results*.json"))

    if not files:
        logger.warning("No ablation_results*.json files found to sync to MongoDB Atlas.")
        return False

    logger.info(f"Syncing {len(files)} result files to MongoDB Atlas...")
    overall_success = True
    for f in files:
        if not syncer.sync_file(f, dry_run=dry_run):
            overall_success = False
    return overall_success


def main():
    parser = argparse.ArgumentParser(description="Sync ablation study results continuously to MongoDB Atlas.")
    parser.add_argument(
        "--files",
        nargs="+",
        help="JSON result files to sync (default: auto-detect ablation_results*.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview documents to be synced without network operations"
    )

    args = parser.parse_args()
    success = sync_to_mongodb(json_files=args.files, dry_run=args.dry_run)
    if not success and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
