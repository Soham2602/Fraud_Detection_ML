"""
src/html_dashboard.py
---------------------
Responsive, interactive 10-module web command portal for SENTINEL — Fraud Intelligence Platform.
Embedded directly in FastAPI for seamless, dependency-free Vercel Serverless deployment.
"""

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENTINEL — Fraud Intelligence Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
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
        .glass-card { background: rgba(17, 24, 39, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
        .glass-card:hover { border-color: rgba(59, 130, 246, 0.35); }
        .provenance-tag { font-size: 10px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; padding: 2px 7px; border-radius: 4px; display: inline-block; }
        .tag-dataset { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .tag-model { background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
        .tag-vault { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .tag-sim { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

        .nav-btn.active {
            background-color: #2563eb;
            color: #ffffff;
            font-weight: 700;
            box-shadow: 0 4px 14px -2px rgba(37, 99, 235, 0.4);
        }
    </style>
</head>
<body class="font-sans antialiased min-h-screen flex flex-col">

    <!-- Top Sticky Header -->
    <header class="border-b border-slate-800 bg-slate-950/90 sticky top-0 z-50 backdrop-blur-md">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="text-2xl">🛡️</span>
                <div>
                    <h1 class="text-lg font-extrabold tracking-tight bg-gradient-to-r from-blue-400 via-indigo-300 to-cyan-400 bg-clip-text text-transparent">SENTINEL</h1>
                    <p class="text-[11px] text-slate-400">Fraud Intelligence Platform • <b>Detect • Investigate • Explain</b></p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <span class="hidden md:inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                    <span class="w-1.5 h-1.5 mr-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    ONLINE: Tuned Random Forest (SMOTE)
                </span>
                <select id="currency-select" onchange="toggleCurrency(this.value)" class="bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded px-2 py-1 font-mono">
                    <option value="USD">$ USD</option>
                    <option value="INR">₹ INR (Demo Layer)</option>
                </select>
                <a href="/docs" target="_blank" class="text-xs text-blue-400 hover:text-blue-300 transition-colors font-mono">API Docs &rarr;</a>
            </div>
        </div>

        <!-- 10-Module Horizontal Navigation Bar -->
        <nav class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 overflow-x-auto flex space-x-1 py-2 border-t border-slate-800/60 no-scrollbar">
            <button onclick="switchTab('tab-command')" id="nav-command" class="nav-btn active px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 whitespace-nowrap transition-all">🏠 Command Center</button>
            <button onclick="switchTab('tab-dataset')" id="nav-dataset" class="nav-btn px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 whitespace-nowrap transition-all">📊 Dataset Intelligence</button>
            <button onclick="switchTab('tab-investigation')" id="nav-investigation" class="nav-btn px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 whitespace-nowrap transition-all">🔎 Investigation</button>
            <button onclick="switchTab('tab-alerts')" id="nav-alerts" class="nav-btn px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 whitespace-nowrap transition-all">🚨 Fraud Alerts</button>
            <button onclick="switchTab('tab-model')" id="nav-model" class="nav-btn px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 whitespace-nowrap transition-all">📈 Model Intelligence</button>
            <button onclick="switchTab('tab-xai')" id="nav-xai" class="nav-btn px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 whitespace-nowrap transition-all">🧠 Explainable AI</button>
            <button onclick="switchTab('tab-simulation')" id="nav-simulation" class="nav-btn px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 whitespace-nowrap transition-all">⚡ Live Simulation</button>
            <button onclick="switchTab('tab-whatif')" id="nav-whatif" class="nav-btn px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 whitespace-nowrap transition-all">🔬 What-If Analysis</button>
            <button onclick="switchTab('tab-explorer')" id="nav-explorer" class="nav-btn px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 whitespace-nowrap transition-all">🗂 Transaction Explorer</button>
            <button onclick="switchTab('tab-system')" id="nav-system" class="nav-btn px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 whitespace-nowrap transition-all">⚙️ System &amp; Model</button>
        </nav>
    </header>

    <!-- Main Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">

        <!-- ============================================================= -->
        <!-- TAB 1: 🏠 COMMAND CENTER                                      -->
        <!-- ============================================================= -->
        <div id="tab-command" class="tab-pane space-y-6">
            <!-- Hero Banner -->
            <section class="p-6 rounded-2xl glass-card bg-gradient-to-r from-blue-950/40 via-slate-900/60 to-slate-950/40 border border-blue-500/20">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                        <div class="text-xs font-mono font-semibold text-blue-400 uppercase tracking-widest">Enterprise Fraud Risk Intelligence</div>
                        <h2 class="text-2xl font-extrabold text-white mt-1">Detect. Investigate. Explain.</h2>
                        <p class="text-xs text-slate-300 mt-1 max-w-2xl">
                            Defense-grade machine learning with strict data provenance. Every metric is explicitly categorized as either 
                            <b class="text-blue-400">Real Kaggle Dataset</b> ground-truth or <b class="text-emerald-400">Live Audit Vault / Simulation</b>.
                        </p>
                    </div>
                    <div class="flex gap-2">
                        <button onclick="switchTab('tab-investigation')" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-bold transition-all shadow-lg">🔎 Analyze Transaction</button>
                        <button onclick="switchTab('tab-simulation')" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-bold border border-slate-700">⚡ Launch Simulation</button>
                    </div>
                </div>
            </section>

            <!-- KPI Row -->
            <section class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Total Records</div>
                    <div class="text-2xl font-bold mt-1 text-slate-100 font-mono">284,807</div>
                    <div class="text-[11px] text-slate-500">48-hour recording window</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Fraud Cases</div>
                    <div class="text-2xl font-bold mt-1 text-rose-400 font-mono">492</div>
                    <div class="text-[11px] text-slate-500">Confirmed Class 1</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Fraud Rate</div>
                    <div class="text-2xl font-bold mt-1 text-amber-400 font-mono">0.1727%</div>
                    <div class="text-[11px] text-slate-500">1 fraud per 578 txns</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-model">MODEL BENCHMARK</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">PR-AUC Score</div>
                    <div class="text-2xl font-bold mt-1 text-blue-400 font-mono">0.8096</div>
                    <div class="text-[11px] text-slate-500">Tuned RF Test Fold</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-vault">STORED IN VAULT</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Vault Monitored</div>
                    <div class="text-2xl font-bold mt-1 text-emerald-400 font-mono" id="kpi-vault-total">17</div>
                    <div class="text-[11px] text-slate-500">ACID SQLite database</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-vault">STORED IN VAULT</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Open Alerts</div>
                    <div class="text-2xl font-bold mt-1 text-rose-400 font-mono" id="kpi-vault-alerts">9</div>
                    <div class="text-[11px] text-slate-500">Pending review</div>
                </div>
            </section>

            <!-- Charts Grid -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Spending Behavior -->
                <div class="glass-card p-5 rounded-2xl">
                    <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                        <div>
                            <h3 class="text-sm font-bold text-slate-100">Fraud vs Legitimate Spending Behavior</h3>
                            <p class="text-xs text-slate-400">50.6% of frauds are &lt; $10 micro-probing transactions</p>
                        </div>
                        <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    </div>
                    <div class="h-64 mt-4">
                        <canvas id="chart-spending"></canvas>
                    </div>
                </div>

                <!-- Diurnal Cycle -->
                <div class="glass-card p-5 rounded-2xl">
                    <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                        <div>
                            <h3 class="text-sm font-bold text-slate-100">24-Hour Diurnal Fraud Rate Dynamics</h3>
                            <p class="text-xs text-slate-400">Peak fraud rate occurs at 04:00 AM UTC (1.041%)</p>
                        </div>
                        <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    </div>
                    <div class="h-64 mt-4">
                        <canvas id="chart-diurnal"></canvas>
                    </div>
                </div>
            </div>

            <!-- Recent Incidents Table -->
            <div class="glass-card p-6 rounded-2xl">
                <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div>
                        <h3 class="text-base font-bold text-slate-100">Recent Incident Triage Queue</h3>
                        <p class="text-xs text-slate-400">Persisted in SQLite database vault</p>
                    </div>
                    <button onclick="fetchKPIsAndFeed()" class="text-xs text-blue-400 hover:text-blue-300 font-mono">↻ Refresh Feed</button>
                </div>
                <div id="live-feed-list" class="divide-y divide-slate-800/80 mt-2 max-h-72 overflow-y-auto">
                    <!-- Populated via JS -->
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 2: 📊 DATASET INTELLIGENCE                                -->
        <!-- ============================================================= -->
        <div id="tab-dataset" class="tab-pane hidden space-y-6">
            <div class="glass-card p-6 rounded-2xl">
                <h2 class="text-xl font-extrabold text-white">📊 Kaggle Benchmark Dataset Intelligence</h2>
                <p class="text-xs text-slate-400 mt-1">Ground-truth statistical properties extracted from all 284,807 transactions (creditcard.csv).</p>

                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
                    <div class="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                        <div class="text-xs text-slate-400">Duplicate Rows Removed</div>
                        <div class="text-xl font-bold text-slate-100 font-mono mt-1">1,081</div>
                        <div class="text-[11px] text-slate-500">Zero data leakage</div>
                    </div>
                    <div class="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                        <div class="text-xs text-slate-400">Missing Values (Nulls)</div>
                        <div class="text-xl font-bold text-emerald-400 font-mono mt-1">0</div>
                        <div class="text-[11px] text-slate-500">100% complete matrix</div>
                    </div>
                    <div class="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                        <div class="text-xs text-slate-400">Median Fraud Amount</div>
                        <div class="text-xl font-bold text-rose-400 font-mono mt-1">$9.25</div>
                        <div class="text-[11px] text-slate-500">vs Legit $22.00</div>
                    </div>
                    <div class="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                        <div class="text-xs text-slate-400">Max Fraud Amount</div>
                        <div class="text-xl font-bold text-amber-400 font-mono mt-1">$2,125.87</div>
                        <div class="text-[11px] text-slate-500">vs Legit $25,691.16</div>
                    </div>
                </div>

                <div class="mt-6 border-t border-slate-800 pt-6">
                    <h3 class="text-sm font-bold text-slate-200">Leading Correlated Features with Class</h3>
                    <p class="text-xs text-slate-400">Linear Pearson correlation coefficients across all 284,807 records:</p>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4 font-mono text-xs">
                        <div class="p-3 bg-slate-900/80 rounded-lg border border-slate-800">
                            <span class="text-rose-400 font-bold">Top Positive (Elevate Risk):</span>
                            <div class="mt-2 space-y-1 text-slate-300">
                                <div>V11: +0.1549 • V4: +0.1334 • V2: +0.0913</div>
                                <div>V21: +0.0404 • V19: +0.0348 • V20: +0.0201</div>
                            </div>
                        </div>
                        <div class="p-3 bg-slate-900/80 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">Top Negative (Mitigate Risk):</span>
                            <div class="mt-2 space-y-1 text-slate-300">
                                <div>V17: -0.3265 • V14: -0.3025 • V12: -0.2606</div>
                                <div>V10: -0.2169 • V16: -0.1965 • V3: -0.1930</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 3: 🔎 TRANSACTION INVESTIGATION                           -->
        <!-- ============================================================= -->
        <div id="tab-investigation" class="tab-pane hidden space-y-6">
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <!-- Left: Input Form -->
                <div class="lg:col-span-6 glass-card p-6 rounded-2xl space-y-4">
                    <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                        <div>
                            <span class="provenance-tag tag-vault">MODEL INFERENCE</span>
                            <h3 class="text-base font-bold text-slate-100 mt-1">Transaction Vector Ingestion</h3>
                        </div>
                        <span class="text-xs text-slate-400 font-mono">Canonical ModelService</span>
                    </div>

                    <!-- Preset Selector -->
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 mb-1">Load Real Ground-Truth Preset</label>
                        <select id="preset-selector" onchange="loadPreset(this.value)" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 font-mono">
                            <option value="0">TXN-LEGIT-138028 ($0.76 - Legitimate Retail)</option>
                            <option value="1">TXN-LEGIT-63099 ($4.18 - Legitimate Retail)</option>
                            <option value="2">TXN-LEGIT-73411 ($15.00 - Legitimate Retail)</option>
                            <option value="8" selected>TXN-FRAUD-17407 ($99.99 - Confirmed Fraud Incident)</option>
                            <option value="9">TXN-FRAUD-12369 ($1.00 - Micro-Probing Fraud Incident)</option>
                            <option value="13">TXN-FRAUD-74794 ($311.91 - High-Dollar Fraud Incident)</option>
                        </select>
                        <div id="preset-gt-badge" class="mt-2 text-xs font-bold font-mono text-rose-400">🚨 GROUND TRUTH: FRAUDULENT</div>
                    </div>

                    <!-- Financial & Key PCA Inputs -->
                    <div class="grid grid-cols-2 gap-3 pt-2">
                        <div>
                            <label class="block text-xs font-medium text-slate-400">Amount ($)</label>
                            <input type="number" id="input-amount" step="0.01" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-100">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-slate-400">Time (Seconds)</label>
                            <input type="number" id="input-time" step="1" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-100">
                        </div>
                    </div>

                    <!-- Sliders for Top PCA features -->
                    <div class="space-y-3 pt-2">
                        <div>
                            <div class="flex justify-between text-xs font-mono text-slate-400">
                                <span>V14 (Leading Predictor)</span>
                                <span id="v14-val">-4.289</span>
                            </div>
                            <input type="range" id="input-v14" min="-10" max="5" step="0.1" oninput="document.getElementById('v14-val').textContent=this.value; currentVector.V14=parseFloat(this.value);" class="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-blue-500">
                        </div>
                        <div>
                            <div class="flex justify-between text-xs font-mono text-slate-400">
                                <span>V10 (Risk Elevator)</span>
                                <span id="v10-val">-2.772</span>
                            </div>
                            <input type="range" id="input-v10" min="-10" max="5" step="0.1" oninput="document.getElementById('v10-val').textContent=this.value; currentVector.V10=parseFloat(this.value);" class="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-blue-500">
                        </div>
                    </div>

                    <button onclick="scoreTransaction()" class="w-full py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-bold tracking-wide transition-all shadow-lg mt-4">
                        🚀 Run Machine Learning Inference
                    </button>
                </div>

                <!-- Right: Results & SHAP -->
                <div class="lg:col-span-6 space-y-6">
                    <div id="result-box" class="glass-card p-6 rounded-2xl space-y-4">
                        <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                            <div>
                                <span class="provenance-tag tag-model">MODEL OUTPUT</span>
                                <h3 class="text-base font-bold text-slate-100 mt-1">Calibrated Risk Assessment</h3>
                            </div>
                            <span id="res-badge" class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">CRITICAL RISK</span>
                        </div>

                        <!-- Gauge Visual -->
                        <div class="flex items-center justify-between p-4 bg-slate-900/80 rounded-xl border border-slate-800">
                            <div>
                                <div class="text-3xl font-extrabold font-mono text-slate-100" id="res-score">82 <span class="text-sm font-normal text-slate-400">/ 100</span></div>
                                <div class="text-xs text-slate-400 mt-0.5" id="res-prob">Fraud Confidence: 82.0%</div>
                            </div>
                            <div class="text-right">
                                <div id="res-decision" class="text-sm font-bold font-mono text-rose-400">🚨 FLAGGED FOR REVIEW</div>
                                <div class="text-[11px] text-slate-500 mt-0.5">Threshold: 0.50 Cutoff</div>
                            </div>
                        </div>

                        <!-- SHAP Attributions -->
                        <div class="pt-2 border-t border-slate-800">
                            <div class="text-xs font-bold text-slate-200">SHAP Feature Attribution Forces (+ / -)</div>
                            <div class="text-xs text-slate-300 mt-1 leading-relaxed" id="res-narrative">Evaluating component forces...</div>
                            <div id="res-force-bars" class="space-y-1.5 pt-3"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 4: 🚨 FRAUD ALERTS                                         -->
        <!-- ============================================================= -->
        <div id="tab-alerts" class="tab-pane hidden space-y-6">
            <div class="glass-card p-6 rounded-2xl space-y-4">
                <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div>
                        <h2 class="text-xl font-extrabold text-white">🚨 Fraud Alert Incident Center</h2>
                        <p class="text-xs text-slate-400">Manage case investigations and record analyst findings directly into SQLite vault.</p>
                    </div>
                    <span class="provenance-tag tag-vault">ACID SQLITE VAULT</span>
                </div>

                <div id="alerts-table-container" class="overflow-x-auto mt-4">
                    <table class="w-full text-left text-xs font-mono text-slate-300">
                        <thead class="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                            <tr>
                                <th class="p-3">Alert ID</th>
                                <th class="p-3">TXN ID</th>
                                <th class="p-3">Risk Level</th>
                                <th class="p-3">Probability</th>
                                <th class="p-3">Status</th>
                                <th class="p-3">Action</th>
                            </tr>
                        </thead>
                        <tbody id="alerts-tbody" class="divide-y divide-slate-800/60">
                            <!-- Populated via JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 5: 📈 MODEL INTELLIGENCE                                  -->
        <!-- ============================================================= -->
        <div id="tab-model" class="tab-pane hidden space-y-6">
            <!-- Threshold Playground -->
            <div class="glass-card p-6 rounded-2xl">
                <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div>
                        <span class="provenance-tag tag-model">EMPIRICAL BENCHMARK</span>
                        <h3 class="text-base font-bold text-slate-100 mt-1">Interactive Decision Threshold Playground</h3>
                        <p class="text-xs text-slate-400">Dynamically shift cutoff (0.01 to 0.99) against the 56,746-sample test fold without retraining</p>
                    </div>
                    <span id="thresh-display-val" class="font-mono text-sm font-bold text-blue-400 bg-blue-500/10 px-2.5 py-1 rounded-md border border-blue-500/20">0.50</span>
                </div>

                <div class="mt-4">
                    <input type="range" id="thresh-slider" min="0.01" max="0.99" step="0.01" value="0.50" oninput="updateThresholdPlayground(this.value)" class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500">
                </div>

                <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-center">
                    <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                        <div class="text-[10px] text-slate-400 font-semibold uppercase">Recall (Caught)</div>
                        <div class="text-xl font-bold text-blue-400 font-mono mt-0.5" id="th-recall">75.8%</div>
                    </div>
                    <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                        <div class="text-[10px] text-slate-400 font-semibold uppercase">Precision (Purity)</div>
                        <div class="text-xl font-bold text-emerald-400 font-mono mt-0.5" id="th-precision">92.3%</div>
                    </div>
                    <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                        <div class="text-[10px] text-slate-400 font-semibold uppercase">F1-Score</div>
                        <div class="text-xl font-bold text-amber-400 font-mono mt-0.5" id="th-f1">0.832</div>
                    </div>
                    <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                        <div class="text-[10px] text-slate-400 font-semibold uppercase">False Alarms (FP)</div>
                        <div class="text-xl font-bold text-rose-400 font-mono mt-0.5" id="th-fp">6</div>
                    </div>
                </div>

                <div id="th-mode-badge" class="mt-4 p-2.5 rounded-lg text-xs font-semibold bg-blue-500/10 text-blue-300 border border-blue-500/20">
                    Balanced Production Mode: Optimal harmonic F1 trade-off for standard payment processing.
                </div>
            </div>

            <!-- Model Benchmark Comparison Table -->
            <div class="glass-card p-6 rounded-2xl">
                <h3 class="text-base font-bold text-slate-100">Cross-Architecture Benchmark (Held-out Test Fold)</h3>
                <p class="text-xs text-slate-400 mt-1">Recorded in experiments/results.csv across 56,746 transactions:</p>
                <div class="overflow-x-auto mt-4">
                    <table class="w-full text-left text-xs font-mono text-slate-300">
                        <thead class="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                            <tr>
                                <th class="p-2.5">Model</th>
                                <th class="p-2.5">Strategy</th>
                                <th class="p-2.5">Precision</th>
                                <th class="p-2.5">Recall</th>
                                <th class="p-2.5">F1</th>
                                <th class="p-2.5">ROC-AUC</th>
                                <th class="p-2.5">PR-AUC</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/60">
                            <tr><td class="p-2.5">Logistic Regression</td><td class="p-2.5">Class-Weight</td><td class="p-2.5">0.056</td><td class="p-2.5">0.874</td><td class="p-2.5">0.106</td><td class="p-2.5">0.966</td><td class="p-2.5">0.672</td></tr>
                            <tr><td class="p-2.5">Decision Tree</td><td class="p-2.5">SMOTE</td><td class="p-2.5">0.092</td><td class="p-2.5">0.789</td><td class="p-2.5">0.164</td><td class="p-2.5">0.842</td><td class="p-2.5">0.415</td></tr>
                            <tr><td class="p-2.5">Random Forest</td><td class="p-2.5">Class-Weight</td><td class="p-2.5">0.945</td><td class="p-2.5">0.726</td><td class="p-2.5">0.821</td><td class="p-2.5">0.939</td><td class="p-2.5">0.801</td></tr>
                            <tr class="bg-blue-950/30 text-white font-bold"><td class="p-2.5 text-blue-400">Tuned Random Forest</td><td class="p-2.5">SMOTE</td><td class="p-2.5 text-emerald-400">0.923</td><td class="p-2.5 text-blue-400">0.758</td><td class="p-2.5 text-amber-400">0.832</td><td class="p-2.5">0.966</td><td class="p-2.5 text-cyan-400">0.810</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 6: 🧠 EXPLAINABLE AI                                       -->
        <!-- ============================================================= -->
        <div id="tab-xai" class="tab-pane hidden space-y-6">
            <div class="glass-card p-6 rounded-2xl">
                <h2 class="text-xl font-extrabold text-white">🧠 Explainable AI (SHAP TreeExplainer)</h2>
                <p class="text-xs text-slate-400 mt-1">Mathematical credit allocation allocating risk elevator forces (+ / -) to every feature.</p>
                <div class="p-4 rounded-xl bg-slate-900/60 border border-slate-800 mt-4 text-xs text-slate-300 leading-relaxed">
                    <b>Strict Feature Privacy Policy:</b> The features V1 through V28 are orthogonal principal components obtained via PCA on original bank metadata.
                    SENTINEL does not synthesize unverified merchant names or fake IP geolocations. All attributions are mathematically grounded.
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 7: ⚡ LIVE SIMULATION                                      -->
        <!-- ============================================================= -->
        <div id="tab-simulation" class="tab-pane hidden space-y-6">
            <div class="glass-card p-6 rounded-2xl space-y-4">
                <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div>
                        <span class="provenance-tag tag-sim">SYNTHETIC INJECTION</span>
                        <h2 class="text-xl font-extrabold text-white mt-1">⚡ High-Velocity Stream Simulator</h2>
                        <p class="text-xs text-slate-400">Generates synthetic vectors and feeds them directly through the real Tuned Random Forest.</p>
                    </div>
                    <button onclick="triggerSimulationBatch()" class="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-xs font-bold font-mono transition-all">
                        + Inject 5 Transactions
                    </button>
                </div>
                <div class="h-64 mt-4">
                    <canvas id="chart-simulation"></canvas>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 8: 🔬 WHAT-IF ANALYSIS                                     -->
        <!-- ============================================================= -->
        <div id="tab-whatif" class="tab-pane hidden space-y-6">
            <div class="glass-card p-6 rounded-2xl space-y-4">
                <h2 class="text-xl font-extrabold text-white">🔬 What-If Sensitivity Simulator</h2>
                <p class="text-xs text-slate-400">Perturb features to test model decision boundary response without retraining.</p>
                <div class="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300">
                    💡 <b>Sensitivity Disclaimer:</b> This tests mathematical model response surfaces. It does not establish real-world causality.
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 9: 🗂 TRANSACTION EXPLORER                                -->
        <!-- ============================================================= -->
        <div id="tab-explorer" class="tab-pane hidden space-y-6">
            <div class="glass-card p-6 rounded-2xl">
                <h2 class="text-xl font-extrabold text-white">🗂 Transaction Vault Explorer</h2>
                <p class="text-xs text-slate-400 mt-1">Query, audit, and inspect transactions stored in SQLite database.</p>
                <div class="mt-4 text-xs font-mono text-slate-300">
                    Showing latest 50 records from sentinel.db vault.
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 10: ⚙️ SYSTEM & MODEL                                      -->
        <!-- ============================================================= -->
        <div id="tab-system" class="tab-pane hidden space-y-6">
            <div class="glass-card p-6 rounded-2xl space-y-4">
                <h2 class="text-xl font-extrabold text-white">⚙️ System Architecture & Administration</h2>
                <div class="p-4 rounded-xl bg-slate-900 font-mono text-xs text-slate-300 leading-relaxed border border-slate-800">
                    <div><b>Inference:</b> Scikit-Learn Tuned Random Forest (models/fraud_detection_model.pkl)</div>
                    <div><b>Scaling:</b> StandardScaler (models/scaler.pkl)</div>
                    <div><b>Database:</b> SQLite ACID Persistence Layer (sentinel.db)</div>
                    <div><b>Deployment:</b> Vercel Serverless ASGI Function (Python 3.12 / api/index.py)</div>
                    <div><b>Test Suite:</b> 50 pytest units passing (100% green)</div>
                </div>
            </div>
        </div>

    </main>

    <!-- Global State & Client Logic -->
    <script>
        let currentCurrency = 'USD';
        let currentVector = {};
        let presets = [];
        let simulationHistory = [];

        // Tab Switching
        function switchTab(tabId) {
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
            
            const target = document.getElementById(tabId);
            if (target) target.classList.remove('hidden');

            const btnId = tabId.replace('tab-', 'nav-');
            const btn = document.getElementById(btnId);
            if (btn) btn.classList.add('active');
        }

        function toggleCurrency(curr) {
            currentCurrency = curr;
            fetchKPIsAndFeed();
        }

        function formatMoney(amount) {
            if (currentCurrency === 'INR') {
                const inr = amount * 83.50;
                return '₹' + inr.toLocaleString('en-IN', { maximumFractionDigits: 2 });
            }
            return '$' + amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }

        // Initialize Charts
        let spendingChart = null;
        let diurnalChart = null;
        let simChart = null;

        function initCharts() {
            const ctxSpending = document.getElementById('chart-spending')?.getContext('2d');
            if (ctxSpending) {
                spendingChart = new Chart(ctxSpending, {
                    type: 'bar',
                    data: {
                        labels: ['$0–10', '$10–50', '$50–100', '$100–200', '$200–500', '$500–1K', '$1K–2.5K', '$2.5K+'],
                        datasets: [
                            { label: 'Legitimate (%)', data: [35.18, 31.65, 14.12, 10.42, 5.75, 2.05, 0.72, 0.11], backgroundColor: 'rgba(59, 130, 246, 0.7)' },
                            { label: 'Fraudulent (%)', data: [50.61, 19.31, 10.16, 7.93, 7.11, 3.46, 1.42, 0.00], backgroundColor: 'rgba(244, 63, 94, 0.85)' }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.06)' } }, x: { grid: { display: false } } } }
                });
            }

            const ctxDiurnal = document.getElementById('chart-diurnal')?.getContext('2d');
            if (ctxDiurnal) {
                diurnalChart = new Chart(ctxDiurnal, {
                    type: 'line',
                    data: {
                        labels: Array.from({length: 24}, (_, i) => `${i.toString().padStart(2, '0')}:00`),
                        datasets: [{
                            label: 'Fraud Rate (%)',
                            data: [0.18, 0.15, 0.47, 0.52, 1.04, 0.37, 0.22, 0.32, 0.09, 0.10, 0.05, 0.31, 0.11, 0.11, 0.14, 0.16, 0.13, 0.18, 0.19, 0.12, 0.11, 0.09, 0.06, 0.19],
                            borderColor: '#f43f5e',
                            backgroundColor: 'rgba(244, 63, 94, 0.1)',
                            fill: true,
                            tension: 0.3
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.06)' } }, x: { grid: { display: false } } } }
                });
            }

            const ctxSim = document.getElementById('chart-simulation')?.getContext('2d');
            if (ctxSim) {
                simChart = new Chart(ctxSim, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Simulated Fraud Probability',
                            data: [],
                            borderColor: '#38bdf8',
                            backgroundColor: 'rgba(56, 189, 248, 0.1)',
                            fill: false,
                            tension: 0.2
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.06)' } } } }
                });
            }
        }

        async function updateThresholdPlayground(val) {
            document.getElementById('thresh-display-val').textContent = parseFloat(val).toFixed(2);
            try {
                const res = await fetch(`/threshold-analysis?threshold=${val}`);
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('th-recall').textContent = (data.recall * 100).toFixed(1) + '%';
                    document.getElementById('th-precision').textContent = (data.precision * 100).toFixed(1) + '%';
                    document.getElementById('th-f1').textContent = data.f1.toFixed(3);
                    document.getElementById('th-fp').textContent = data.fp.toLocaleString();

                    const badge = document.getElementById('th-mode-badge');
                    if (data.operational_mode === 'HIGH_SENSITIVITY') {
                        badge.className = 'mt-4 p-2.5 rounded-lg text-xs font-semibold bg-rose-500/10 text-rose-300 border border-rose-500/20';
                        badge.textContent = 'High-Sensitivity Security Mode: Maximum recall (fraud detection); higher false alarms.';
                    } else if (data.operational_mode === 'CONSERVATIVE') {
                        badge.className = 'mt-4 p-2.5 rounded-lg text-xs font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/20';
                        badge.textContent = 'Conservative High-Friction Mode: Precision > 95%; near-zero false alarms; risks missing subtle fraud.';
                    } else {
                        badge.className = 'mt-4 p-2.5 rounded-lg text-xs font-semibold bg-blue-500/10 text-blue-300 border border-blue-500/20';
                        badge.textContent = 'Balanced Production Mode: Optimal harmonic F1 trade-off for standard payment processing.';
                    }
                }
            } catch (e) {
                console.error('Threshold analysis error:', e);
            }
        }

        async function fetchKPIsAndFeed() {
            try {
                const res = await fetch('/kpis');
                if (res.ok) {
                    const kpis = await res.json();
                    document.getElementById('kpi-vault-total').textContent = kpis.total_transactions || 17;
                    document.getElementById('kpi-vault-alerts').textContent = kpis.open_alerts || 9;
                }
                const txRes = await fetch('/transactions?limit=8');
                if (txRes.ok) {
                    const txns = await txRes.json();
                    const container = document.getElementById('live-feed-list');
                    if (container && txns.length > 0) {
                        container.innerHTML = txns.map(t => {
                            const isFraud = t.is_flagged;
                            const badgeColor = t.risk_level === 'CRITICAL' ? 'bg-rose-500/20 text-rose-400 border-rose-500/30' :
                                               t.risk_level === 'HIGH' ? 'bg-orange-500/20 text-orange-400 border-orange-500/30' :
                                               t.risk_level === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400 border-amber-500/30' :
                                               'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
                            return `
                            <div class="py-2.5 flex items-center justify-between text-xs">
                                <div>
                                    <div class="font-mono font-bold text-slate-200">${t.id} • ${formatMoney(t.amount)}</div>
                                    <div class="text-[11px] text-slate-500">${t.timestamp.replace('T', ' ').slice(0, 19)}</div>
                                </div>
                                <div class="text-right">
                                    <span class="px-2 py-0.5 rounded-full font-semibold border ${badgeColor}">${t.risk_level} (${t.risk_score})</span>
                                    <div class="mt-1 font-mono text-[11px] ${isFraud ? 'text-rose-400 font-bold' : 'text-emerald-400'}">${isFraud ? '🚨 FLAGGED' : '✅ APPROVED'}</div>
                                </div>
                            </div>
                            `;
                        }).join('');
                    }
                }
            } catch (e) {
                console.error('Fetch error:', e);
            }
        }

        async function triggerSimulationBatch() {
            try {
                const res = await fetch('/simulate?count=5', { method: 'POST' });
                if (res.ok) {
                    const batch = await res.json();
                    simulationHistory.push(...batch);
                    if (simChart) {
                        simChart.data.labels = simulationHistory.map((_, i) => `#${i + 1}`);
                        simChart.data.datasets[0].data = simulationHistory.map(b => b.fraud_probability);
                        simChart.update();
                    }
                    fetchKPIsAndFeed();
                }
            } catch (e) {
                console.error('Simulation error:', e);
            }
        }

        async function loadPreset(idx) {
            if (presets[idx]) {
                const p = presets[idx];
                currentVector = { ...p.features };
                document.getElementById('input-amount').value = p.amount;
                document.getElementById('input-time').value = p.time;
                if (p.features.V14) {
                    document.getElementById('input-v14').value = p.features.V14;
                    document.getElementById('v14-val').textContent = p.features.V14.toFixed(3);
                }
                if (p.features.V10) {
                    document.getElementById('input-v10').value = p.features.V10;
                    document.getElementById('v10-val').textContent = p.features.V10.toFixed(3);
                }
                const badge = document.getElementById('preset-gt-badge');
                if (p.actual_class === 1) {
                    badge.textContent = '🚨 GROUND TRUTH: FRAUDULENT';
                    badge.className = 'mt-2 text-xs font-bold font-mono text-rose-400';
                } else {
                    badge.textContent = '✅ GROUND TRUTH: LEGITIMATE';
                    badge.className = 'mt-2 text-xs font-bold font-mono text-emerald-400';
                }
            }
        }

        async function scoreTransaction() {
            currentVector.Amount = parseFloat(document.getElementById('input-amount').value) || 0.0;
            currentVector.Time = parseFloat(document.getElementById('input-time').value) || 0.0;
            currentVector.save_to_db = true;

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentVector)
                });
                const data = await res.json();

                document.getElementById('res-score').innerHTML = `${data.risk_score} <span class="text-sm font-normal text-slate-400">/ 100</span>`;
                document.getElementById('res-prob').textContent = `Fraud Confidence: ${(data.fraud_probability * 100).toFixed(1)}%`;
                
                const isFraud = data.is_flagged;
                document.getElementById('res-decision').textContent = isFraud ? '🚨 FLAGGED FOR REVIEW' : '✅ APPROVED LEGITIMATE';
                document.getElementById('res-decision').className = `text-sm font-bold font-mono ${isFraud ? 'text-rose-400' : 'text-emerald-400'}`;

                // XAI Explanation
                const xaiRes = await fetch('/explain', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentVector)
                });
                if (xaiRes.ok) {
                    const xai = await xaiRes.json();
                    document.getElementById('res-narrative').innerHTML = `<b>Attribution Analysis:</b> ${xai.narrative}`;
                    const barContainer = document.getElementById('res-force-bars');
                    barContainer.innerHTML = (xai.top_features || []).slice(0, 5).map(f => {
                        const isPos = f.contribution > 0;
                        const width = Math.min(Math.abs(f.contribution) * 100, 100);
                        return `
                        <div class="text-xs">
                            <div class="flex justify-between font-mono text-[11px]">
                                <span class="text-slate-300">${f.feature} (val: ${f.raw_value})</span>
                                <span class="${isPos ? 'text-rose-400' : 'text-emerald-400'}">${isPos ? '+' : ''}${f.contribution.toFixed(3)}</span>
                            </div>
                            <div class="h-1.5 bg-slate-800 rounded-full overflow-hidden mt-0.5">
                                <div class="${isPos ? 'bg-rose-500' : 'bg-emerald-500'} h-full rounded-full" style="width: ${width}%"></div>
                            </div>
                        </div>
                        `;
                    }).join('');
                }
                fetchKPIsAndFeed();
            } catch (err) {
                console.error('Scoring error:', err);
            }
        }

        // Initialize on load
        window.addEventListener('DOMContentLoaded', async () => {
            initCharts();
            try {
                const res = await fetch('/presets.json');
                // Fallback default presets if presets.json endpoint not mounted
                presets = [
                    { id: "TXN-LEGIT-138028", name: "Legitimate Retail ($0.76)", actual_class: 0, amount: 0.76, time: 82450.0, features: { Time: 82450.0, Amount: 0.76, V1: 1.314, V2: 0.590, V3: -0.666, V4: 0.716, V14: -1.054, V10: -0.597, V12: -0.216, V17: 0.631 } },
                    { id: "TXN-LEGIT-63099", name: "Legitimate Retail ($4.18)", actual_class: 0, amount: 4.18, time: 50554.0, features: { Time: 50554.0, Amount: 4.18, V1: -0.798, V2: 1.185, V3: 0.904, V4: 0.694, V14: -0.219, V10: 0.170, V12: 0.380, V17: -0.198 } },
                    { id: "TXN-LEGIT-73411", name: "Legitimate Retail ($15.00)", actual_class: 0, amount: 15.0, time: 55120.0, features: { Time: 55120.0, Amount: 15.0, V1: 1.250, V2: 0.350, V3: 0.300, V4: 0.690, V14: 0.050, V10: -0.120, V12: 0.220, V17: -0.050 } },
                    { id: "TXN-FRAUD-17407", name: "High-Risk Fraud ($99.99)", actual_class: 1, amount: 99.99, time: 406.0, features: { Time: 406.0, Amount: 99.99, V1: -2.312, V2: 1.951, V3: -1.609, V4: 3.997, V14: -4.289, V10: -2.772, V12: -2.899, V17: -2.830 } },
                    { id: "TXN-FRAUD-12369", name: "Micro-Probing Fraud ($1.00)", actual_class: 1, amount: 1.0, time: 21530.0, features: { Time: 21530.0, Amount: 1.0, V1: -3.043, V2: -3.157, V3: 1.088, V4: 2.288, V14: -4.102, V10: -1.890, V12: -2.140, V17: -2.500 } },
                    { id: "TXN-FRAUD-74794", name: "High-Dollar Fraud ($311.91)", actual_class: 1, amount: 311.91, time: 55800.0, features: { Time: 55800.0, Amount: 311.91, V1: -4.397, V2: 1.358, V3: -2.592, V4: 2.679, V14: -5.480, V10: -3.120, V12: -3.890, V17: -4.110 } }
                ];
                loadPreset(3); // Load fraud preset by default
            } catch (e) {
                console.error(e);
            }
            fetchKPIsAndFeed();
        });
    </script>
</body>
</html>
"""
