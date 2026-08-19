#!/usr/bin/env python3
"""
MongoDB Cloud Live Update Monitor & Minimalist Dashboard.

Connects to MongoDB Atlas to track real-time training telemetry updates,
epoch heartbeats, and experiment results, presenting a minimal dashboard
accessible from any phone, tablet, or browser.

Usage:
    python monitor_dashboard.py
    python monitor_dashboard.py --port 8080
    python monitor_dashboard.py --dry-run
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_to_cloud import MongoDBAtlasSync, load_env_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mongo_monitor")

mongo_syncer = None
is_dry_run_mode = False
event_logs: List[Dict[str, Any]] = []
last_seen_heartbeat_time = None


def get_telemetry_payload() -> Dict[str, Any]:
    """Retrieve telemetry status and variant results from MongoDB Cloud."""
    global event_logs, last_seen_heartbeat_time

    if is_dry_run_mode:
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "connected": True,
            "target": "ablation_study.results (Demo)",
            "heartbeat": {
                "status": "RUNNING",
                "current_variant": "A9_vitb_unfrozen",
                "current_epoch": 6,
                "total_epochs": 20,
                "epoch_progress_pct": 30.0,
                "latest_loss": 0.2145,
                "latest_val_iou": 0.4612,
                "device": "NVIDIA RTX 3050 (Demo)",
                "last_heartbeat": now_str,
            },
            "results": [
                {
                    "variant": "A1_full_model",
                    "description": "Full model baseline (3-scale)",
                    "metrics": {"mean_iou": 0.4520, "pixel_accuracy": 0.8210, "iou_Dust": 0.5770, "iou_RunDown": 0.2660, "iou_Scratch": 0.3030, "avg_inference_ms": 48.5}
                },
                {
                    "variant": "A2_single_scale",
                    "description": "Single-scale features (skip_layers=[11])",
                    "metrics": {"mean_iou": 0.4440, "pixel_accuracy": 0.8120, "iou_Dust": 0.5880, "iou_RunDown": 0.1790, "iou_Scratch": 0.3270, "avg_inference_ms": 38.2}
                }
            ],
            "events": [
                {"timestamp": now_str, "message": "Demo: Received epoch 6 heartbeat for A9_vitb_unfrozen"}
            ]
        }

    hb = {}
    results = []
    connected = False
    target = "Unknown"

    if mongo_syncer:
        target = mongo_syncer.target
        try:
            hb = mongo_syncer.get_live_status()
            results = mongo_syncer.get_all_results()
            connected = True
        except Exception as e:
            logger.error(f"Error reading from MongoDB Cloud: {e}")

    # Track new heartbeat events for live activity log
    hb_time = hb.get("last_heartbeat")
    if hb_time and hb_time != last_seen_heartbeat_time:
        last_seen_heartbeat_time = hb_time
        evt_msg = f"Update received: {hb.get('current_variant', 'Unknown')} (Ep {hb.get('current_epoch', 0)}/{hb.get('total_epochs', 0)}, Loss: {hb.get('latest_loss', 'N/A')}, mIoU: {hb.get('latest_val_iou', 'N/A')})"
        event_logs.insert(0, {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S UTC"),
            "message": evt_msg
        })
        event_logs = event_logs[:15]  # Keep last 15 updates

    # Check timeout for running status (>3 mins -> INACTIVE)
    status_str = hb.get("status", "DISCONNECTED" if not connected else "IDLE")
    if status_str == "RUNNING" and hb_time:
        try:
            hb_dt = datetime.datetime.fromisoformat(hb_time.replace("Z", "+00:00"))
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            if (now_dt - hb_dt).total_seconds() > 180:
                status_str = "INACTIVE"
                hb["status"] = status_str
        except Exception:
            pass

    return {
        "connected": connected,
        "target": target,
        "heartbeat": hb,
        "results": results,
        "events": event_logs
    }


MINIMAL_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MongoDB Cloud Live Telemetry</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --surface: #121826;
            --border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-green: #10b981;
            --accent-indigo: #6366f1;
            --accent-red: #f43f5e;
            --accent-amber: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg);
            color: var(--text-primary);
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            padding: 24px;
        }

        .container {
            max-width: 960px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }

        .logo-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .mongo-icon {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 10px var(--accent-green);
        }

        h1 {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .status-pill {
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 6px 14px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .grid-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
        }

        .stat-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            font-weight: 600;
            margin-bottom: 6px;
        }

        .stat-value {
            font-size: 1.6rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        .progress-box {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }

        .progress-info {
            display: flex;
            justify-content: space-between;
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .bar-bg {
            height: 10px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 999px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--accent-indigo), var(--accent-green));
            border-radius: 999px;
            transition: width 0.4s ease;
        }

        .section-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .table-box {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 24px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }

        th {
            text-align: left;
            padding: 12px 16px;
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-transform: uppercase;
            border-bottom: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.01);
        }

        td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            font-family: 'JetBrains Mono', monospace;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .log-box {
            background: #000;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            max-height: 180px;
            overflow-y: auto;
            color: var(--text-secondary);
        }

        .log-item {
            margin-bottom: 6px;
            display: flex;
            gap: 12px;
        }

        .log-ts {
            color: var(--accent-indigo);
            flex-shrink: 0;
        }

        .log-msg {
            color: #e2e8f0;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-group">
                <div class="mongo-icon"></div>
                <h1>MongoDB Cloud Telemetry</h1>
            </div>
            <div class="status-pill" id="statusPill">
                <span id="statusDot" style="width:8px; height:8px; border-radius:50%; background:var(--text-secondary);"></span>
                <span id="statusText">INITIALIZING</span>
            </div>
        </header>

        <div class="progress-box">
            <div class="progress-info">
                <span id="varName">Variant: Idle</span>
                <span id="epText">Epoch 0 / 0</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" id="barFill"></div>
            </div>
        </div>

        <div class="grid-stats">
            <div class="stat-card">
                <div class="stat-title">Validation mIoU</div>
                <div class="stat-value" id="valMiou" style="color:var(--accent-green);">0.00%</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Training Loss</div>
                <div class="stat-value" id="valLoss">0.0000</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Finished Variants</div>
                <div class="stat-value" id="valFinished">0</div>
            </div>
        </div>

        <div class="section-title">Completed Variants</div>
        <div class="table-box">
            <table>
                <thead>
                    <tr>
                        <th>Variant</th>
                        <th>mIoU</th>
                        <th>Dust IoU</th>
                        <th>RunDown IoU</th>
                        <th>Scratch IoU</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <tr><td colspan="5" style="text-align:center; color:var(--text-secondary); padding:20px;">No results recorded yet.</td></tr>
                </tbody>
            </table>
        </div>

        <div class="section-title">MongoDB Cloud Audit Feed</div>
        <div class="log-box" id="logBox">
            <div class="log-item"><span class="log-ts">--:--:--</span><span class="log-msg">Listening for live updates from MongoDB Cloud...</span></div>
        </div>
    </div>

    <script>
        async function updateDashboard() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();

                const hb = data.heartbeat || {};
                const results = data.results || [];
                const events = data.events || [];

                // Status
                const st = (hb.status || (data.connected ? 'IDLE' : 'DISCONNECTED')).toUpperCase();
                document.getElementById('statusText').innerText = st;
                const dot = document.getElementById('statusDot');
                if (st === 'RUNNING') {
                    dot.style.background = 'var(--accent-green)';
                    dot.style.boxShadow = '0 0 10px var(--accent-green)';
                } else if (st === 'COMPLETED') {
                    dot.style.background = 'var(--accent-indigo)';
                    dot.style.boxShadow = '0 0 10px var(--accent-indigo)';
                } else {
                    dot.style.background = 'var(--accent-red)';
                    dot.style.boxShadow = 'none';
                }

                // Progress
                document.getElementById('varName').innerText = 'Variant: ' + (hb.current_variant || 'Idle');
                const curEp = hb.current_epoch || 0;
                const totEp = hb.total_epochs || 0;
                const pct = hb.epoch_progress_pct || 0;
                document.getElementById('epText').innerText = `Epoch ${curEp} / ${totEp} (${pct}%)`;
                document.getElementById('barFill').style.width = pct + '%';

                // Stats
                document.getElementById('valMiou').innerText = hb.latest_val_iou ? (hb.latest_val_iou * 100).toFixed(2) + '%' : '0.00%';
                document.getElementById('valLoss').innerText = hb.latest_loss ? hb.latest_loss.toFixed(4) : '0.0000';
                document.getElementById('valFinished').innerText = results.length;

                // Table
                const tbody = document.getElementById('tableBody');
                if (results.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-secondary); padding:20px;">No results recorded yet.</td></tr>';
                } else {
                    tbody.innerHTML = results.map(r => {
                        const m = r.metrics || {};
                        return `<tr>
                            <td><strong>${r.variant || ''}</strong></td>
                            <td style="color:var(--accent-green);">${m.mean_iou ? (m.mean_iou * 100).toFixed(1) + '%' : 'N/A'}</td>
                            <td>${m.iou_Dust ? (m.iou_Dust * 100).toFixed(1) + '%' : 'N/A'}</td>
                            <td>${m.iou_RunDown ? (m.iou_RunDown * 100).toFixed(1) + '%' : 'N/A'}</td>
                            <td>${m.iou_Scratch ? (m.iou_Scratch * 100).toFixed(1) + '%' : 'N/A'}</td>
                        </tr>`;
                    }).join('');
                }

                // Event Audit Log
                const logBox = document.getElementById('logBox');
                if (events.length > 0) {
                    logBox.innerHTML = events.map(e => `
                        <div class="log-item">
                            <span class="log-ts">[${e.timestamp}]</span>
                            <span class="log-msg">${e.message}</span>
                        </div>
                    `).join('');
                }
            } catch (err) {
                console.error("Telemetry fetch failed:", err);
            }
        }

        setInterval(updateDashboard, 2500);
        updateDashboard();
    </script>
</body>
</html>
"""


class TelemetryRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/telemetry") or self.path.startswith("/api/status"):
            data = get_telemetry_payload()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(MINIMAL_DASHBOARD_HTML.encode("utf-8"))

    def log_message(self, format, *args):
        pass


def main():
    global mongo_syncer, is_dry_run_mode

    parser = argparse.ArgumentParser(description="MongoDB Cloud Minimal Live Monitor")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock telemetry mode")

    args = parser.parse_args()
    is_dry_run_mode = args.dry_run

    if not is_dry_run_mode:
        try:
            mongo_syncer = MongoDBAtlasSync()
            logger.info("Connected to MongoDB Cloud telemetry collection.")
        except Exception as e:
            logger.warning(f"MongoDB Cloud connection failed ({e}). Starting in demo mode.")
            is_dry_run_mode = True

    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, TelemetryRequestHandler)

    logger.info("=" * 60)
    logger.info("  MongoDB Cloud Minimal Live Monitor")
    logger.info(f"  URL: http://localhost:{args.port} (or http://<your-ip>:{args.port})")
    logger.info("=" * 60)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down monitor...")
        httpd.server_close()


if __name__ == "__main__":
    main()
