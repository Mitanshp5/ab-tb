#!/usr/bin/env python3
"""
Remote Monitoring Web Dashboard for Ablation Study.

Fetches live status heartbeats and completed variant results from MongoDB Atlas,
serving a responsive web interface accessible on any device (phone, tablet, PC).

Usage:
    python monitor_server.py --port 8080
    python monitor_server.py --port 8080 --host 0.0.0.0
    python monitor_server.py --dry-run
"""

import argparse
import datetime
import json
import logging
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_to_cloud import MongoDBAtlasSync, load_env_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("monitor_dashboard")

# Global MongoDB Sync instance
mongo_syncer = None
is_dry_run_mode = False


def get_monitoring_data() -> Dict[str, Any]:
    """Retrieve combined live status and variant metrics from MongoDB Atlas or mock."""
    if is_dry_run_mode:
        return {
            "heartbeat": {
                "status": "RUNNING",
                "current_variant": "A9_vitb_unfrozen",
                "current_epoch": 6,
                "total_epochs": 20,
                "epoch_progress_pct": 30.0,
                "latest_loss": 0.2145,
                "latest_val_iou": 0.4612,
                "device": "NVIDIA RTX 3050 (Demo)",
                "last_heartbeat": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
                },
                {
                    "variant": "A3_two_scale",
                    "description": "Two-scale features (skip_layers=[7, 11])",
                    "metrics": {"mean_iou": 0.4600, "pixel_accuracy": 0.8290, "iou_Dust": 0.5780, "iou_RunDown": 0.2680, "iou_Scratch": 0.3270, "avg_inference_ms": 42.1}
                }
            ]
        }

    hb = {}
    results = []
    if mongo_syncer:
        try:
            hb = mongo_syncer.get_live_status()
        except Exception as e:
            logger.error(f"Error fetching live status: {e}")
        try:
            raw_results = mongo_syncer.get_all_results()
            # Filter out failed variants (empty metrics or error field)
            results = [r for r in raw_results if r.get("metrics") and r["metrics"].get("mean_iou") and not r.get("error")]
        except Exception as e:
            logger.error(f"Error fetching results: {e}")

    # Calculate status active/stopped based on last_heartbeat age (> 3 mins -> STOPPED)
    status_str = hb.get("status", "UNKNOWN")
    last_hb_str = hb.get("last_heartbeat")
    if status_str == "RUNNING" and last_hb_str:
        try:
            hb_dt = datetime.datetime.fromisoformat(last_hb_str.replace("Z", "+00:00"))
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            if (now_dt - hb_dt).total_seconds() > 180:
                status_str = "STOPPED / INACTIVE"
                hb["status"] = status_str
        except Exception:
            pass

    return {
        "heartbeat": hb,
        "results": results
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ablation Study — Live Remote Monitor</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --card-bg: rgba(22, 29, 45, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.3);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
            --accent-blue: #3b82f6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px 16px;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 28px;
            flex-wrap: wrap;
            gap: 16px;
        }

        .brand h1 {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 30%, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }

        .brand p {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(12px);
        }

        .pulse-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: var(--accent-green);
            box-shadow: 0 0 12px var(--accent-green);
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        .grid-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .card:hover {
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateY(-2px);
        }

        .card-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 8px;
        }

        .card-val {
            font-size: 1.6rem;
            font-weight: 700;
            color: #fff;
            letter-spacing: -0.02em;
        }

        .card-sub {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .progress-section {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 28px;
            backdrop-filter: blur(16px);
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .progress-track {
            height: 12px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 999px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #6366f1, #10b981);
            border-radius: 999px;
            transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 16px var(--primary-glow);
        }

        .table-container {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            overflow: hidden;
            backdrop-filter: blur(16px);
        }

        .table-title {
            padding: 20px 24px;
            border-bottom: 1px solid var(--card-border);
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.875rem;
        }

        th {
            background: rgba(255, 255, 255, 0.02);
            padding: 14px 20px;
            color: var(--text-muted);
            font-weight: 600;
            border-bottom: 1px solid var(--card-border);
        }

        td {
            padding: 16px 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-main);
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        .miou-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
        }

        footer {
            text-align: center;
            margin-top: 32px;
            font-size: 0.8rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand">
                <h1>Ablation Study Remote Monitor</h1>
                <p>MongoDB Atlas Real-Time Training Telemetry</p>
            </div>
            <div class="status-badge">
                <span class="pulse-dot" id="statusDot"></span>
                <span id="statusText">CONNECTING...</span>
            </div>
        </header>

        <div class="progress-section">
            <div class="progress-header">
                <span id="variantTitle">Current Variant: Idle / Waiting</span>
                <span id="epochLabel">Epoch 0 / 0</span>
            </div>
            <div class="progress-track">
                <div class="progress-bar" id="progressBar"></div>
            </div>
        </div>

        <div class="grid-cards">
            <div class="card">
                <div class="card-label">Validation mIoU</div>
                <div class="card-val" id="valMiou">0.00%</div>
                <div class="card-sub" id="valMiouSub">Latest epoch mean IoU</div>
            </div>
            <div class="card">
                <div class="card-label">Training Loss</div>
                <div class="card-val" id="valLoss">0.0000</div>
                <div class="card-sub">Combined Dice + Focal</div>
            </div>
            <div class="card">
                <div class="card-label">Active Hardware</div>
                <div class="card-val" id="valDevice" style="font-size:1.2rem; padding-top:4px;">N/A</div>
                <div class="card-sub" id="valLastHb">Last Heartbeat: --</div>
            </div>
        </div>

        <div class="table-container">
            <div class="table-title">
                <span>Completed Ablation Variants</span>
                <span id="variantCount" style="font-size:0.85rem; color:var(--text-muted);">0 Finished</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Variant ID</th>
                        <th>mIoU</th>
                        <th>Dust IoU</th>
                        <th>RunDown IoU</th>
                        <th>Scratch IoU</th>
                        <th>Inference</th>
                    </tr>
                </thead>
                <tbody id="resultsTableBody">
                    <tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:32px;">Loading completed variants...</td></tr>
                </tbody>
            </table>
        </div>

        <footer>
            DINOv2 + DenseFPN-UNet Surface Defect Ablation Pipeline &bull; Auto-syncing every 3s
        </footer>
    </div>

    <script>
        async function fetchStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                
                const hb = data.heartbeat || {};
                const results = data.results || [];
                
                // Status Badge
                const statusStr = (hb.status || 'OFFLINE').toUpperCase();
                const statusText = document.getElementById('statusText');
                const statusDot = document.getElementById('statusDot');
                
                statusText.innerText = statusStr;
                if (statusStr === 'RUNNING') {
                    statusDot.style.backgroundColor = 'var(--accent-green)';
                    statusDot.style.boxShadow = '0 0 12px var(--accent-green)';
                } else if (statusStr === 'COMPLETED') {
                    statusDot.style.backgroundColor = 'var(--accent-blue)';
                    statusDot.style.boxShadow = '0 0 12px var(--accent-blue)';
                } else {
                    statusDot.style.backgroundColor = 'var(--accent-red)';
                    statusDot.style.boxShadow = '0 0 12px var(--accent-red)';
                }
                
                // Progress
                const variant = hb.current_variant || 'Idle / Waiting';
                const curEp = hb.current_epoch || 0;
                const totEp = hb.total_epochs || 0;
                const pct = hb.epoch_progress_pct || 0;
                
                document.getElementById('variantTitle').innerText = 'Current Variant: ' + variant;
                document.getElementById('epochLabel').innerText = `Epoch ${curEp} / ${totEp} (${pct}%)`;
                document.getElementById('progressBar').style.width = pct + '%';
                
                // Cards
                const miou = hb.latest_val_iou !== null && hb.latest_val_iou !== undefined ? (hb.latest_val_iou * 100).toFixed(2) + '%' : '0.00%';
                document.getElementById('valMiou').innerText = miou;
                document.getElementById('valLoss').innerText = hb.latest_loss !== null && hb.latest_loss !== undefined ? hb.latest_loss.toFixed(4) : '0.0000';
                document.getElementById('valDevice').innerText = hb.device || 'N/A';
                
                if (hb.last_heartbeat) {
                    const dt = new Date(hb.last_heartbeat);
                    document.getElementById('valLastHb').innerText = 'Last updated: ' + dt.toLocaleTimeString();
                } else {
                    document.getElementById('valLastHb').innerText = 'Last updated: Never';
                }
                
                // Results Table
                document.getElementById('variantCount').innerText = `${results.length} Finished`;
                const tbody = document.getElementById('resultsTableBody');
                if (results.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:32px;">No completed variants yet.</td></tr>';
                } else {
                    let html = '';
                    results.forEach(r => {
                        const m = r.metrics || {};
                        const miouPct = m.mean_iou ? (m.mean_iou * 100).toFixed(1) + '%' : 'N/A';
                        const dust = m.iou_Dust ? (m.iou_Dust * 100).toFixed(1) + '%' : 'N/A';
                        const rundown = m.iou_RunDown ? (m.iou_RunDown * 100).toFixed(1) + '%' : 'N/A';
                        const scratch = m.iou_Scratch ? (m.iou_Scratch * 100).toFixed(1) + '%' : 'N/A';
                        const ms = m.avg_inference_ms ? m.avg_inference_ms.toFixed(1) + ' ms' : 'N/A';
                        
                        html += `<tr>
                            <td><strong>${r.variant || 'Unknown'}</strong><br><small style="color:var(--text-muted);">${r.description || ''}</small></td>
                            <td><span class="miou-badge">${miouPct}</span></td>
                            <td>${dust}</td>
                            <td>${rundown}</td>
                            <td>${scratch}</td>
                            <td>${ms}</td>
                        </tr>`;
                    });
                    tbody.innerHTML = html;
                }
            } catch (err) {
                console.error('Failed to fetch status:', err);
            }
        }

        setInterval(fetchStatus, 3000);
        fetchStatus();
    </script>
</body>
</html>
"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving JSON telemetry API and HTML Dashboard."""

    def do_GET(self):
        if self.path == "/api/status" or self.path.startswith("/api/status"):
            data = get_monitoring_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

    def log_message(self, format, *args):
        # Silence routine GET logging to keep terminal output clean
        pass


def main():
    global mongo_syncer, is_dry_run_mode

    parser = argparse.ArgumentParser(description="Ablation Study Remote Monitoring Web Server.")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)")
    parser.add_argument("--dry-run", action="store_true", help="Run with mock telemetry data without connecting to MongoDB")

    args = parser.parse_args()

    is_dry_run_mode = args.dry_run

    if not is_dry_run_mode:
        try:
            mongo_syncer = MongoDBAtlasSync()
            logger.info("Connected to MongoDB Atlas for live remote monitoring.")
        except Exception as e:
            logger.warning(f"Could not connect to MongoDB Atlas ({e}). Starting in mock/demo mode.")
            is_dry_run_mode = True

    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, DashboardRequestHandler)

    logger.info("=" * 60)
    logger.info("  Ablation Study Remote Monitoring Dashboard")
    logger.info(f"  Local Access:  http://localhost:{args.port}")
    logger.info(f"  Network Access: http://<your-device-ip>:{args.port}")
    logger.info("=" * 60)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down remote monitoring server...")
        httpd.server_close()


if __name__ == "__main__":
    main()
