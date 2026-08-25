#!/usr/bin/env python3
"""
MongoDB Atlas Results Sync for Ablation Study.

Continuously syncs ablation study results (metrics, IoUs, parameters) to
MongoDB Atlas in real time as each variant completes, with an offline queue so
a dropped link never costs a result and never needs a manual re-upload.

Configuration is loaded from .env:
  MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
  MONGODB_TARGET=ablation_study.results

CLI:
  python upload_to_cloud.py                    # sync ablation_results*.json
  python upload_to_cloud.py --flush            # push anything stuck in the offline queue
  python upload_to_cloud.py --status           # show connection + queue state
  python upload_to_cloud.py --list             # list variant docs currently in Atlas
  python upload_to_cloud.py --purge-stale      # remove docs whose variant is not in the study
"""

import argparse
import atexit
import datetime
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mongodb_sync")

HEARTBEAT_ID = "live_status_heartbeat"


def load_env_file(env_path: Optional[Path] = None):
    """Load key-value pairs from .env file into os.environ if present."""
    path = env_path or (Path.cwd() / ".env")
    if not path.exists():
        # Also try next to this file, so the sync works regardless of cwd.
        path = Path(__file__).resolve().parent / ".env"
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
    """Real-time sync of ablation results to MongoDB Atlas, with an offline queue.

    One MongoClient is created lazily and reused for the lifetime of the
    instance. The previous implementation built a fresh ``MongoClient`` inside
    every call -- once per epoch for heartbeats, and once per queued item when
    flushing -- and never closed any of them. Each client carries its own
    connection pool and monitor threads, so a 13-variant x 100-epoch run leaked
    well over a thousand of them, which is a large part of why long runs became
    unstable and why syncing ended up being done by hand.
    """

    _CONNECT_KWARGS = dict(
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=20000,
        retryWrites=True,
        maxPoolSize=4,
        appname="ablation-study-sync",
    )

    def __init__(
        self,
        uri: Optional[str] = None,
        target: Optional[str] = None,
        queue_file: str = "offline_sync_queue.json",
    ):
        load_env_file()
        uri = uri or os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
        if not uri:
            # Atlas onboarding writes MONGODB_USERNAME; accept both spellings.
            user = os.environ.get("MONGODB_USER") or os.environ.get("MONGODB_USERNAME")
            password = os.environ.get("MONGODB_PASS") or os.environ.get("MONGODB_PASSWORD")
            host = os.environ.get("MONGODB_HOST") or os.environ.get("MONGODB_CLUSTER", "cluster0.mongodb.net")
            if user and password:
                uri = f"mongodb+srv://{user}:{password}@{host}/?retryWrites=true&w=majority"
        self.uri = uri
        self.target = target or os.environ.get("MONGODB_TARGET", "ablation_study.results")
        self.queue_file = Path(queue_file)

        self._client = None
        self._lock = threading.Lock()
        self._warned_missing_uri = False
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def parse_target(self) -> Tuple[str, str]:
        target_str = self.target or "ablation_study.results"
        parts = target_str.split(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "ablation_study", parts[0] or "results"

    def _get_client(self):
        """Return the shared MongoClient, creating it once on first use."""
        if not self.uri:
            raise ValueError(
                "MongoDB connection URI is missing. Set MONGODB_URI in .env "
                "(copy .env.example and fill in your own credentials)."
            )
        try:
            import pymongo
        except ImportError:
            raise ImportError(
                "Package 'pymongo' is not installed. Install via 'pip install pymongo dnspython'."
            )

        with self._lock:
            if self._client is None:
                self._client = pymongo.MongoClient(self.uri, **self._CONNECT_KWARGS)
            return self._client

    def _get_collection(self):
        db_name, coll_name = self.parse_target()
        client = self._get_client()
        return client[db_name][coll_name], db_name, coll_name

    def _reset_client(self):
        """Drop the cached client so the next call reconnects from scratch."""
        with self._lock:
            client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def close(self):
        """Close the shared client. Safe to call more than once."""
        self._reset_client()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def ping(self) -> Tuple[bool, Optional[str]]:
        """Check connectivity without writing anything."""
        try:
            self._get_client().admin.command("ping")
            return True, None
        except Exception as e:
            return False, str(e)

    # ------------------------------------------------------------------
    # Offline queue
    # ------------------------------------------------------------------

    def _load_queue(self) -> List[Dict[str, Any]]:
        """Load pending offline sync items."""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    def _write_queue(self, queue: List[Dict[str, Any]]) -> None:
        try:
            tmp = self.queue_file.with_suffix(self.queue_file.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=2)
            os.replace(tmp, self.queue_file)
        except Exception as e:
            logger.error(f"[Offline Queue Write Error] {e}")

    def _save_to_queue(self, doc: Dict[str, Any]):
        """Save an un-synced result document to the local offline queue."""
        queue = self._load_queue()
        variant_name = doc.get("variant")
        if variant_name:
            queue = [q for q in queue if q.get("variant") != variant_name]
        queue.append(doc)
        self._write_queue(queue)
        logger.info(
            f"[Offline Queue] Saved '{variant_name}' to {self.queue_file} "
            f"({len(queue)} item(s) pending; they upload automatically once the link is back)"
        )

    def queue_depth(self) -> int:
        return len(self._load_queue())

    def flush_offline_queue(self) -> int:
        """Upload queued documents if the network is available.

        Returns the number of documents successfully flushed. A single
        connection is used for the whole batch, and one failure aborts the
        remainder rather than retrying a dead link once per item.
        """
        queue = self._load_queue()
        if not queue:
            return 0

        try:
            coll, db_n, coll_n = self._get_collection()
        except Exception as e:
            logger.debug(f"[Offline Flush] Still offline: {e}")
            return 0

        remaining: List[Dict[str, Any]] = []
        synced = 0
        for i, item in enumerate(queue):
            try:
                if item.get("variant"):
                    coll.replace_one({"variant": item["variant"]}, item, upsert=True)
                else:
                    coll.insert_one(item)
                synced += 1
            except Exception as e:
                logger.debug(f"[Offline Flush] Stopping at item {i} ({e}); will retry later.")
                remaining.extend(queue[i:])
                self._reset_client()
                break

        if synced:
            logger.info(
                f"[Offline Sync Restored] Flushed {synced} queued result(s) to {db_n}.{coll_n}"
            )
        self._write_queue(remaining)
        return synced

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def sync_variant(
        self,
        variant_result: Dict[str, Any],
        source_file: str = "live_sync",
        run_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> bool:
        """Upsert one variant result. Queues locally if the network is down."""
        db_name, coll_name = self.parse_target()
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        doc: Dict[str, Any] = {
            "_upload_meta": {
                "source_file": source_file,
                "synced_at": timestamp,
            }
        }
        doc.update(variant_result)
        if run_id and "run_id" not in doc:
            doc["run_id"] = run_id

        if dry_run:
            logger.info(
                f"[DRY-RUN] Would upsert variant '{variant_result.get('variant')}' "
                f"to MongoDB Atlas ({db_name}.{coll_name})"
            )
            return True

        # Push anything stranded from an earlier outage first, so ordering on
        # the dashboard matches the order the variants actually finished.
        self.flush_offline_queue()

        try:
            coll, db_n, coll_n = self._get_collection()
            variant_name = doc.get("variant")
            if variant_name:
                res = coll.replace_one({"variant": variant_name}, doc, upsert=True)
                action = "Inserted new" if res.upserted_id else "Updated existing"
                logger.info(
                    f"[MongoDB Atlas Sync] {action} record for variant '{variant_name}' in {db_n}.{coll_n}"
                )
            else:
                coll.insert_one(doc)
                logger.info(f"[MongoDB Atlas Sync] Inserted record in {db_n}.{coll_n}")
            return True
        except Exception as e:
            logger.warning(f"[MongoDB Sync Network Drop] {e}. Storing result in offline queue.")
            self._reset_client()
            self._save_to_queue(doc)
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
        run_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> bool:
        """Publish live progress. Best-effort: never raises, never blocks training."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        progress_pct = 0.0
        if current_epoch and total_epochs and total_epochs > 0:
            progress_pct = round((current_epoch / total_epochs) * 100, 1)

        doc = {
            "_id": HEARTBEAT_ID,
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
            "run_id": run_id,
            "pending_offline_docs": self.queue_depth(),
            "last_heartbeat": timestamp,
        }

        if dry_run:
            logger.info(f"[DRY-RUN] Heartbeat status '{status}':\n{json.dumps(doc, indent=2)}")
            return True

        try:
            self.flush_offline_queue()
        except Exception:
            pass

        try:
            coll, _, _ = self._get_collection()
            coll.replace_one({"_id": HEARTBEAT_ID}, doc, upsert=True)
            return True
        except Exception as e:
            logger.debug(f"[Heartbeat Network Drop] {e}")
            self._reset_client()
            return False

    # ------------------------------------------------------------------
    # Reads / maintenance
    # ------------------------------------------------------------------

    def get_live_status(self) -> Dict[str, Any]:
        """Fetch the latest live status heartbeat document."""
        try:
            coll, _, _ = self._get_collection()
            doc = coll.find_one({"_id": HEARTBEAT_ID})
            if doc:
                doc.pop("_id", None)
                return doc
        except Exception as e:
            logger.error(f"Failed to fetch live status: {e}")
        return {"status": "UNKNOWN", "last_heartbeat": None}

    def get_all_results(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch completed variant documents, optionally filtered to one run."""
        try:
            coll, _, _ = self._get_collection()
            query: Dict[str, Any] = {"variant": {"$exists": True}}
            if run_id:
                query["run_id"] = run_id
            results = []
            for doc in coll.find(query):
                doc.pop("_id", None)
                results.append(doc)
            return results
        except Exception as e:
            logger.error(f"Failed to fetch variant results: {e}")
            return []

    def purge_stale(self, keep_variants: List[str], dry_run: bool = False) -> int:
        """Delete variant docs whose name is not part of the current study.

        The collection accumulated one-off documents (A1_fresh_local,
        A2_single_scale_fixed, A3_two_scale_fixed) from ad-hoc reruns. They are
        upserted by variant name, so they never get overwritten and they show
        up on the dashboard as if they were study rows.
        """
        try:
            coll, db_n, coll_n = self._get_collection()
        except Exception as e:
            logger.error(f"Cannot purge, not connected: {e}")
            return 0

        stale = [
            d.get("variant")
            for d in coll.find({"variant": {"$exists": True}}, {"variant": 1})
            if d.get("variant") not in keep_variants
        ]
        if not stale:
            logger.info("No stale variant documents found.")
            return 0

        logger.info(f"Stale documents in {db_n}.{coll_n}: {', '.join(sorted(stale))}")
        if dry_run:
            logger.info("[DRY-RUN] Nothing deleted.")
            return len(stale)

        res = coll.delete_many({"variant": {"$in": stale}})
        logger.info(f"Deleted {res.deleted_count} stale document(s).")
        return res.deleted_count

    def sync_file(self, json_path: Path, run_id: Optional[str] = None, dry_run: bool = False) -> bool:
        """Sync an entire ablation results JSON file."""
        if not json_path.exists():
            logger.warning(f"File {json_path} does not exist.")
            return False

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                success = True
                for item in data:
                    if not self.sync_variant(item, source_file=json_path.name,
                                             run_id=run_id, dry_run=dry_run):
                        success = False
                return success
            if isinstance(data, dict):
                return self.sync_variant(data, source_file=json_path.name,
                                         run_id=run_id, dry_run=dry_run)
            return False
        except Exception as e:
            logger.error(f"Failed to sync {json_path}: {e}")
            return False


def sync_to_mongodb(json_files: Optional[List[str]] = None, dry_run: bool = False) -> bool:
    """Sync default or specified ablation JSON result files to MongoDB Atlas."""
    with MongoDBAtlasSync() as syncer:
        files = [Path(f) for f in json_files] if json_files else sorted(
            Path.cwd().glob("ablation_results*.json")
        )
        if not files:
            logger.warning("No ablation_results*.json files found to sync.")
            return False

        logger.info(f"Syncing {len(files)} result file(s) to MongoDB Atlas...")
        overall = True
        for f in files:
            if not syncer.sync_file(f, dry_run=dry_run):
                overall = False
        return overall


def main():
    parser = argparse.ArgumentParser(description="Sync ablation study results to MongoDB Atlas.")
    parser.add_argument("--files", nargs="+",
                        help="JSON result files to sync (default: auto-detect ablation_results*.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview documents without network operations")
    parser.add_argument("--flush", action="store_true",
                        help="Only flush the offline queue, then exit")
    parser.add_argument("--status", action="store_true",
                        help="Show connection state, queue depth and last heartbeat")
    parser.add_argument("--list", action="store_true",
                        help="List variant documents currently stored in Atlas")
    parser.add_argument("--purge-stale", action="store_true",
                        help="Delete variant documents that are not part of the current study")
    args = parser.parse_args()

    with MongoDBAtlasSync() as syncer:
        if args.status:
            ok, err = syncer.ping()
            print(f"URI configured : {'yes' if syncer.uri else 'NO — set MONGODB_URI in .env'}")
            print(f"Target         : {syncer.target}")
            print(f"Connection     : {'OK' if ok else 'FAILED — ' + str(err)}")
            print(f"Offline queue  : {syncer.queue_depth()} pending document(s)")
            if ok:
                hb = syncer.get_live_status()
                print(f"Last heartbeat : {hb.get('last_heartbeat')} (status={hb.get('status')})")
            return

        if args.list:
            docs = syncer.get_all_results()
            print(f"{len(docs)} variant document(s):")
            for d in sorted(docs, key=lambda x: x.get("variant", "")):
                m = d.get("metrics") or {}
                state = d.get("error") or f"mIoU={m.get('mean_iou', float('nan')):.4f}"
                print(f"  {d.get('variant', '?'):<28} run_id={d.get('run_id', '-'):<24} {state}")
            return

        if args.purge_stale:
            try:
                from run_ablation import ABLATION_VARIANTS, PRETRAINED_REFERENCE
                keep = [v["name"] for v in ABLATION_VARIANTS] + [PRETRAINED_REFERENCE["name"]]
            except Exception as e:
                logger.error(f"Could not import the variant list from run_ablation.py: {e}")
                sys.exit(1)
            syncer.purge_stale(keep, dry_run=args.dry_run)
            return

        if args.flush:
            n = syncer.flush_offline_queue()
            print(f"Flushed {n} document(s); {syncer.queue_depth()} still pending.")
            return

    success = sync_to_mongodb(json_files=args.files, dry_run=args.dry_run)
    if not success and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
