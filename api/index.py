"""
api/index.py
------------
Vercel Serverless Function entry point for SENTINEL — Fraud Intelligence Platform.
Mounts the FastAPI application and serves both the REST API and the interactive
SENTINEL web command portal.
"""

import sys
import os
from pathlib import Path

# Add project root and src directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

for path in [str(BASE_DIR), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from fastapi.responses import HTMLResponse
from starlette.requests import Request
from src.api import app

@app.middleware("http")
async def normalize_vercel_paths(request: Request, call_next):
    """Normalize Vercel rewrite prefixes (/api/index.py or /api) to standard route paths."""
    path = request.scope.get("path", "")
    for prefix in ["/api/index.py", "/api/index", "/api"]:
        if path == prefix:
            request.scope["path"] = "/"
            break
        elif path.startswith(prefix + "/"):
            request.scope["path"] = path[len(prefix):]
            break
    return await call_next(request)

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENTINEL — Fraud Intelligence Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['"Plus Jakarta Sans"', 'sans-serif'],
                        mono: ['"JetBrains Mono"', 'monospace']
                    },
                    colors: {
                        brand: { 50: '#eff6ff', 500: '#3b82f6', 600: '#2563eb', 900: '#1e3a8a' },
                        slate: { 850: '#111827', 900: '#0f172a', 950: '#020617' }
                    }
                }
            }
        }
    </script>
    <style>
        body { background-color: #0b0f19; color: #f8fafc; }
        .glass-card { background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
        .glass-card:hover { border-color: rgba(59, 130, 246, 0.3); }
    </style>
</head>
<body class="font-sans antialiased min-h-screen flex flex-col">

    <!-- Top Navigation Header -->
    <header class="border-b border-slate-800 bg-slate-950/80 sticky top-0 z-50 backdrop-blur-md">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="text-2xl">🛡️</span>
                <div>
                    <h1 class="text-lg font-bold tracking-tight bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">SENTINEL</h1>
                    <p class="text-xs text-slate-400">Fraud Intelligence Platform</p>
                </div>
            </div>
            <div class="flex items-center space-x-4">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <span class="w-1.5 h-1.5 mr-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    Model: Tuned Random Forest (SMOTE)
                </span>
                <a href="/docs" target="_blank" class="text-xs text-blue-400 hover:text-blue-300 transition-colors font-mono">API Docs &rarr;</a>
            </div>
        </div>
    </header>

    <!-- Main Command Center -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

        <!-- Telemetry KPIs -->
        <section class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4" id="kpi-grid">
            <div class="glass-card p-4 rounded-xl">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Total Evaluated</div>
                <div class="text-2xl font-bold mt-1 text-slate-100" id="kpi-total">--</div>
                <div class="text-[11px] text-slate-500">Live & preset records</div>
            </div>
            <div class="glass-card p-4 rounded-xl">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Fraud Incidents</div>
                <div class="text-2xl font-bold mt-1 text-rose-400" id="kpi-fraud">--</div>
                <div class="text-[11px] text-slate-500">Flagged by model</div>
            </div>
            <div class="glass-card p-4 rounded-xl">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">High Risk Count</div>
                <div class="text-2xl font-bold mt-1 text-orange-400" id="kpi-highrisk">--</div>
                <div class="text-[11px] text-slate-500">Score &ge; 61</div>
            </div>
            <div class="glass-card p-4 rounded-xl">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Fraud Rate</div>
                <div class="text-2xl font-bold mt-1 text-slate-100" id="kpi-rate">--%</div>
                <div class="text-[11px] text-slate-500">Of processed volume</div>
            </div>
            <div class="glass-card p-4 rounded-xl">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Avg Amount</div>
                <div class="text-2xl font-bold mt-1 text-slate-100" id="kpi-amount">$0.00</div>
                <div class="text-[11px] text-slate-500">Per transaction</div>
            </div>
            <div class="glass-card p-4 rounded-xl">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Incidents Cleared</div>
                <div class="text-2xl font-bold mt-1 text-sky-400" id="kpi-reviewed">--</div>
                <div class="text-[11px] text-slate-500">Analyst reviews</div>
            </div>
        </section>

        <!-- Two Column Operational Layout -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">

            <!-- Left Panel: Transaction Analyzer -->
            <section class="lg:col-span-7 space-y-6">
                <div class="glass-card p-6 rounded-2xl">
                    <div class="flex items-center justify-between pb-4 border-b border-slate-800">
                        <div>
                            <h2 class="text-lg font-bold text-slate-100">Live Risk Scoring Terminal</h2>
                            <p class="text-xs text-slate-400">Score transactions using trained Random Forest & SMOTE pipeline</p>
                        </div>
                        <button onclick="triggerSimulation()" class="px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 rounded-lg text-xs font-semibold transition-all">
                            ⚡ Stream Simulation
                        </button>
                    </div>

                    <!-- Preset Picker -->
                    <div class="mt-4">
                        <label class="block text-xs font-semibold text-slate-400 uppercase mb-2">Load Benchmark Sample (Real Kaggle Dataset):</label>
                        <select id="preset-selector" onchange="loadSelectedPreset()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500">
                            <option value="legit">Legitimate Retail Transaction ($42.50) [Class 0]</option>
                            <option value="suspicious">Borderline Suspicious Transaction ($580.00) [Class 0 Borderline]</option>
                            <option value="fraud" selected>High-Risk Fraud Incident ($1,250.00) [Class 1 Real Fraud]</option>
                        </select>
                    </div>

                    <!-- Input Fields -->
                    <div class="grid grid-cols-2 gap-4 mt-4">
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 mb-1">Amount ($)</label>
                            <input type="number" id="input-amount" value="1250.00" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-slate-200">
                        </div>
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 mb-1">Time Elapsed (sec)</label>
                            <input type="number" id="input-time" value="406.0" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-slate-200">
                        </div>
                    </div>

                    <!-- Action Button -->
                    <button onclick="scoreTransaction()" class="mt-6 w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold rounded-xl text-sm transition-all shadow-lg shadow-blue-500/25">
                        ⚡ Evaluate Fraud Risk Score
                    </button>

                    <!-- Score Result Card -->
                    <div id="result-box" class="mt-6 p-4 rounded-xl border border-slate-700 bg-slate-900/60 hidden">
                        <div class="flex items-center justify-between">
                            <div>
                                <span id="res-badge" class="px-2.5 py-0.5 rounded-full text-xs font-semibold">CRITICAL RISK</span>
                                <div class="text-3xl font-extrabold mt-1 text-slate-100" id="res-score">92 <span class="text-sm font-normal text-slate-400">/ 100</span></div>
                                <div class="text-xs text-slate-400 mt-1" id="res-prob">Fraud Confidence: 92.0%</div>
                            </div>
                            <div class="text-right">
                                <div class="text-xs text-slate-500">DECISION</div>
                                <div class="text-sm font-bold text-rose-400" id="res-decision">🚨 FLAGGED FOR REVIEW</div>
                            </div>
                        </div>
                        <!-- Explanation -->
                        <div class="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-300" id="res-narrative">
                            Analyzing feature contributions...
                        </div>
                    </div>
                </div>
            </section>

            <!-- Right Panel: Live Feed & Investigation -->
            <section class="lg:col-span-5 space-y-6">
                <div class="glass-card p-6 rounded-2xl">
                    <div class="flex items-center justify-between pb-4 border-b border-slate-800">
                        <div>
                            <h2 class="text-lg font-bold text-slate-100">Live Incident Feed</h2>
                            <p class="text-xs text-slate-400">Real-time alerts and analyst audit trail</p>
                        </div>
                        <button onclick="fetchKPIsAndFeed()" class="text-xs text-slate-400 hover:text-slate-200">↻ Refresh</button>
                    </div>

                    <div id="feed-container" class="mt-4 space-y-3 max-h-[480px] overflow-y-auto pr-1">
                        <div class="text-xs text-slate-500 text-center py-8">Loading live alerts...</div>
                    </div>
                </div>
            </section>
        </div>

        <!-- Academic & Technical Integrity Note -->
        <footer class="pt-6 border-t border-slate-800/80 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div>
                <b>Academic Project:</b> B.Tech AIML / ECE Final Mini-Project &middot; Technical Transparency &amp; Data Honesty Compliant.
            </div>
            <div>
                PCA Components V1–V28 &middot; Zero Data Leakage &middot; Imbalanced-Learn SMOTE
            </div>
        </footer>
    </main>

    <script>
        const PRESET_VECTORS = {
            legit: {
                Amount: 42.50, Time: 3600.0,
                V1: -0.92, V2: 0.18, V3: 1.54, V4: 0.22, V5: 0.38, V6: -0.15, V7: 0.44, V8: 0.05,
                V9: 0.12, V10: -0.08, V11: -0.32, V12: 0.11, V13: 0.45, V14: 0.05, V15: 0.21,
                V16: 0.14, V17: -0.10, V18: 0.08, V19: -0.12, V20: -0.05, V21: -0.08, V22: -0.15,
                V23: 0.05, V24: 0.12, V25: -0.08, V26: 0.04, V27: 0.02, V28: 0.01
            },
            suspicious: {
                Amount: 580.00, Time: 12000.0,
                V1: -2.31, V2: 1.85, V3: -1.20, V4: 1.45, V5: -1.10, V6: -0.45, V7: -1.25, V8: 0.85,
                V9: -0.95, V10: -1.50, V11: 1.15, V12: -1.80, V13: -0.10, V14: -2.40, V15: -0.20,
                V16: -1.10, V17: -1.65, V18: -0.80, V19: 0.45, V20: 0.35, V21: 0.32, V22: -0.10,
                V23: -0.15, V24: -0.25, V25: 0.15, V26: 0.10, V27: 0.20, V28: 0.08
            },
            fraud: {
                Amount: 1250.00, Time: 406.0,
                V1: -2.31, V2: 1.95, V3: -1.60, V4: 3.99, V5: -0.52, V6: -1.42, V7: -2.53, V8: 1.39,
                V9: -2.77, V10: -2.77, V11: 3.20, V12: -2.89, V13: -0.59, V14: -4.28, V15: 0.38,
                V16: -1.14, V17: -2.83, V18: -0.01, V19: 0.41, V20: 0.12, V21: 0.51, V22: -0.03,
                V23: -0.46, V24: 0.32, V25: 0.04, V26: 0.17, V27: 0.26, V28: -0.14
            }
        };

        let currentVector = { ...PRESET_VECTORS.fraud };

        function loadSelectedPreset() {
            const key = document.getElementById('preset-selector').value;
            currentVector = { ...PRESET_VECTORS[key] };
            document.getElementById('input-amount').value = currentVector.Amount;
            document.getElementById('input-time').value = currentVector.Time;
        }

        async function fetchKPIsAndFeed() {
            try {
                const [kpiRes, txnRes] = await Promise.all([
                    fetch('/kpis'),
                    fetch('/transactions?limit=8')
                ]);
                const kpis = await kpiRes.json();
                const txns = await txnRes.json();

                document.getElementById('kpi-total').textContent = (kpis.total_transactions || 0).toLocaleString();
                document.getElementById('kpi-fraud').textContent = (kpis.fraud_flags || 0).toLocaleString();
                document.getElementById('kpi-highrisk').textContent = (kpis.high_risk_transactions || 0).toLocaleString();
                document.getElementById('kpi-rate').textContent = (kpis.fraud_rate_pct || 0).toFixed(1) + '%';
                document.getElementById('kpi-amount').textContent = '$' + (kpis.avg_amount || 0).toFixed(2);
                document.getElementById('kpi-reviewed').textContent = (kpis.transactions_reviewed || 0).toLocaleString();

                renderFeed(txns);
            } catch (e) {
                console.error("Telemetry update error:", e);
            }
        }

        function renderFeed(txns) {
            const container = document.getElementById('feed-container');
            if (!txns || txns.length === 0) {
                container.innerHTML = '<div class="text-xs text-slate-500 text-center py-6">No transactions in vault yet.</div>';
                return;
            }

            container.innerHTML = txns.map(t => {
                const isFraud = t.is_flagged === 1;
                const badgeColor = t.risk_level === 'CRITICAL' ? 'text-rose-400 bg-rose-500/10 border-rose-500/30' :
                                   t.risk_level === 'HIGH' ? 'text-orange-400 bg-orange-500/10 border-orange-500/30' :
                                   t.risk_level === 'MEDIUM' ? 'text-amber-400 bg-amber-500/10 border-amber-500/30' :
                                   'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';

                return `
                <div class="p-3 rounded-xl border border-slate-800 bg-slate-900/40 flex items-center justify-between text-xs hover:border-slate-700 transition-colors">
                    <div>
                        <div class="font-mono font-semibold text-slate-200">${t.id}</div>
                        <div class="text-[11px] text-slate-400 mt-0.5">$${parseFloat(t.amount).toFixed(2)} &middot; ${t.channel || 'POS'} &middot; <span class="capitalize">${t.status}</span></div>
                    </div>
                    <div class="text-right">
                        <span class="px-2 py-0.5 rounded-full font-semibold border ${badgeColor}">${t.risk_level} (${t.risk_score})</span>
                        <div class="mt-1 ${isFraud ? 'text-rose-400 font-bold' : 'text-emerald-400'}">${isFraud ? '🚨 FLAGGED' : '✅ APPROVED'}</div>
                    </div>
                </div>
                `;
            }).join('');
        }

        async function scoreTransaction() {
            currentVector.Amount = parseFloat(document.getElementById('input-amount').value) || 0.0;
            currentVector.Time = parseFloat(document.getElementById('input-time').value) || 0.0;
            currentVector.save_to_db = true;

            const resBox = document.getElementById('result-box');
            resBox.classList.remove('hidden');
            document.getElementById('res-narrative').textContent = "Running ML model inference...";

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentVector)
                });
                const data = await res.json();

                const isFraud = data.is_flagged;
                document.getElementById('res-score').innerHTML = `${data.risk_score} <span class="text-sm font-normal text-slate-400">/ 100</span>`;
                document.getElementById('res-prob').textContent = `Fraud Confidence: ${(data.fraud_probability * 100).toFixed(1)}%`;
                
                const badge = document.getElementById('res-badge');
                badge.textContent = `${data.risk_level} RISK LEVEL`;
                badge.className = `px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                    data.risk_level === 'CRITICAL' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' :
                    data.risk_level === 'HIGH' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                    data.risk_level === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                    'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                }`;

                document.getElementById('res-decision').textContent = isFraud ? '🚨 FLAGGED FOR REVIEW' : '✅ CLEARED LEGITIMATE';
                document.getElementById('res-decision').className = `text-sm font-bold ${isFraud ? 'text-rose-400' : 'text-emerald-400'}`;

                // Fetch XAI explanation
                const xaiRes = await fetch('/explain', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentVector)
                });
                const xaiData = await xaiRes.json();
                document.getElementById('res-narrative').innerHTML = `<b>Attribution Analysis:</b> ${xaiData.narrative}`;

                fetchKPIsAndFeed();
            } catch (err) {
                document.getElementById('res-narrative').textContent = "Error: " + err.message;
            }
        }

        async function triggerSimulation() {
            try {
                await fetch('/simulate?count=3&fraud_bias=0.3', { method: 'POST' });
                fetchKPIsAndFeed();
            } catch (e) {
                console.error(e);
            }
        }

        // Initialize on load
        fetchKPIsAndFeed();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["Web Portal"])
@app.get("/api", response_class=HTMLResponse, tags=["Web Portal"])
@app.get("/api/", response_class=HTMLResponse, tags=["Web Portal"])
def get_sentinel_dashboard():
    """Serve the complete responsive SENTINEL web portal for Vercel deployment."""
    return HTMLResponse(content=HTML_DASHBOARD)
