import datetime
import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, List

try:
    import pymongo
except ImportError:
    pymongo = None


def get_mongo_collection():
    uri = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
    target = os.environ.get("MONGODB_TARGET", "ablation_study.results")
    if not uri:
        return None, "ablation_study", "results", "Missing MONGODB_URI environment variable on Vercel"
    if not pymongo:
        return None, "ablation_study", "results", "pymongo package is not available"

    parts = target.split(".", 1)
    db_name = parts[0] if len(parts) == 2 else "ablation_study"
    coll_name = parts[1] if len(parts) == 2 else parts[0]

    try:
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        return client[db_name][coll_name], db_name, coll_name, None
    except Exception as e:
        return None, db_name, coll_name, str(e)


def get_telemetry_payload() -> Dict[str, Any]:
    coll, db_name, coll_name, err = get_mongo_collection()
    hb = {}
    results = []
    connected = False
    events = []

    if coll is not None:
        try:
            doc = coll.find_one({"_id": "live_status_heartbeat"})
            if doc:
                doc.pop("_id", None)
                hb = doc
            connected = True
        except Exception as e:
            err = str(e)
            hb = {"status": "ERROR", "error": str(e)}

        try:
            cursor = coll.find({"variant": {"$exists": True}})
            for item in cursor:
                item.pop("_id", None)
                results.append(item)
        except Exception:
            pass

    hb_time = hb.get("last_heartbeat")
    if hb_time:
        ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        try:
            hb_dt = datetime.datetime.fromisoformat(hb_time.replace("Z", "+00:00"))
            ist_dt = hb_dt.astimezone(ist_tz)
            ts_str = ist_dt.strftime("%H:%M:%S IST")
        except Exception:
            ts_str = datetime.datetime.now(ist_tz).strftime("%H:%M:%S IST")

        events.append({
            "timestamp": ts_str,
            "message": f"Update received: {hb.get('current_variant', 'Unknown')} (Ep {hb.get('current_epoch', 0)}/{hb.get('total_epochs', 0)}, Loss: {hb.get('latest_loss', 'N/A')}, mIoU: {hb.get('latest_val_iou', 'N/A')})"
        })

    # Status check
    status_str = hb.get("status", "CONNECTED" if connected else "DEMO/OFFLINE")
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
        "target": f"{db_name}.{coll_name}",
        "error": err,
        "heartbeat": hb,
        "results": results,
        "events": events
    }


DASHBOARD_HTML = """<!DOCTYPE html>
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

        <div id="errorBanner" style="display:none; background:rgba(244,63,94,0.12); border:1px solid #f43f5e; color:#fda4af; padding:14px 18px; border-radius:12px; margin-bottom:20px; font-size:0.875rem; line-height:1.5;"></div>

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

        <div class="section-title">All Variants in MongoDB Cloud</div>
        <div class="table-box" style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Variant</th>
                        <th>Status</th>
                        <th>mIoU</th>
                        <th>mDice</th>
                        <th>PixAcc</th>
                        <th>Dust IoU</th>
                        <th>RunDown IoU</th>
                        <th>Scratch IoU</th>
                        <th>Latency</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <tr><td colspan="9" style="text-align:center; color:var(--text-secondary); padding:20px;">No results recorded yet.</td></tr>
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

                // Error Banner Handling
                const errBanner = document.getElementById('errorBanner');
                if (!data.connected && data.error) {
                    errBanner.style.display = 'block';
                    errBanner.innerHTML = `<strong>MongoDB Cloud Connection Alert:</strong> ${data.error}<br><small style="opacity:0.85;">Please verify that <strong>MONGODB_URI</strong> is set in Vercel Settings &rarr; Environment Variables, and that your MongoDB Atlas cluster allows Network Access from anywhere (0.0.0.0/0).</small>`;
                } else {
                    errBanner.style.display = 'none';
                }

                // Status
                const st = (hb.status || (data.connected ? 'IDLE' : 'OFFLINE')).toUpperCase();
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
                    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; color:var(--text-secondary); padding:20px;">No results recorded yet.</td></tr>';
                } else {
                    const fmtPct = (val) => (val !== undefined && val !== null && !isNaN(val)) ? (val * 100).toFixed(1) + '%' : 'N/A';
                    const fmtNum = (val, decimals=1) => (val !== undefined && val !== null && !isNaN(val)) ? val.toFixed(decimals) : 'N/A';

                    tbody.innerHTML = results.map(r => {
                        const m = r.metrics || {};
                        const hasMetrics = m.mean_iou !== undefined && m.mean_iou !== null;
                        
                        let statusHtml = '';
                        if (r.error) {
                            statusHtml = `<span style="color:var(--accent-red); font-size:0.75rem; font-weight:600;">FAILED (${r.error})</span>`;
                        } else if (hasMetrics) {
                            const ep = m.checkpoint_epoch !== undefined ? ` (Ep ${m.checkpoint_epoch})` : '';
                            statusHtml = `<span style="color:var(--accent-green); font-size:0.75rem; font-weight:600;">COMPLETED${ep}</span>`;
                        } else {
                            statusHtml = `<span style="color:var(--text-secondary); font-size:0.75rem;">NO DATA</span>`;
                        }

                        return `<tr>
                            <td>
                                <strong>${r.variant || ''}</strong>
                                ${r.description ? `<br><small style="color:var(--text-secondary); font-size:0.72rem;">${r.description}</small>` : ''}
                            </td>
                            <td>${statusHtml}</td>
                            <td style="color:${hasMetrics ? 'var(--accent-green)' : 'inherit'}; font-weight:600;">${fmtPct(m.mean_iou)}</td>
                            <td>${fmtPct(m.mean_dice)}</td>
                            <td>${fmtPct(m.pixel_accuracy)}</td>
                            <td>${fmtPct(m.iou_Dust)}</td>
                            <td>${fmtPct(m.iou_RunDown)}</td>
                            <td>${fmtPct(m.iou_Scratch)}</td>
                            <td>${m.avg_inference_ms ? fmtNum(m.avg_inference_ms, 1) + ' ms' : 'N/A'}</td>
                        </tr>`;
                    }).join('');
                }

                // Event Audit Log (IST)
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


class handler(BaseHTTPRequestHandler):
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
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

    def log_message(self, format, *args):
        pass
