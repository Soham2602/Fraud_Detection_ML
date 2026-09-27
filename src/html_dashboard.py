"""
src/html_dashboard.py
---------------------
Responsive, interactive web command portal for SENTINEL — Fraud Intelligence Platform.
Implements:
- Dual Modes (Mode 1: Dataset Analytics, Mode 2: Live Operations & Simulation)
- Strict Data Provenance labeling (REAL DATASET vs LIVE VAULT / SIMULATION)
- Calibrated Deterministic Risk Gauge (0-100)
- Interactive Decision Threshold Playground (0.01 to 0.99)
- SHAP Feature Attribution Force Bar Chart
- Live Incident Feed and Stream Simulator
"""

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENTINEL — Fraud Intelligence Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
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
    </style>
</head>
<body class="font-sans antialiased min-h-screen flex flex-col">

    <!-- Top Navigation Header -->
    <header class="border-b border-slate-800 bg-slate-950/80 sticky top-0 z-50 backdrop-blur-md">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="text-2xl">🛡️</span>
                <div>
                    <h1 class="text-lg font-extrabold tracking-tight bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">SENTINEL</h1>
                    <p class="text-xs text-slate-400">Fraud Intelligence Platform · Data-Driven &amp; Explainable</p>
                </div>
            </div>
            <div class="flex items-center space-x-4">
                <span class="hidden sm:inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <span class="w-1.5 h-1.5 mr-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    Model: Tuned Random Forest (SMOTE)
                </span>
                <a href="/docs" target="_blank" class="text-xs text-blue-400 hover:text-blue-300 transition-colors font-mono">Swagger API &rarr;</a>
            </div>
        </div>
    </header>

    <!-- Main Command Center -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

        <!-- Hero Sub-Header -->
        <section class="p-6 rounded-2xl glass-card bg-gradient-to-r from-blue-950/40 via-slate-900/60 to-slate-950/40 border border-blue-500/20">
            <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <div class="text-xs font-mono font-semibold text-blue-400 uppercase tracking-widest">Enterprise Fraud Risk Intelligence</div>
                    <h2 class="text-2xl font-extrabold text-white mt-1">Detect. Investigate. Understand.</h2>
                    <p class="text-xs text-slate-300 mt-1 max-w-2xl">
                        Defense-grade machine learning with strict data provenance. Every metric is explicitly categorized as either 
                        <b class="text-blue-400">Real Kaggle Dataset</b> ground-truth or <b class="text-emerald-400">Live Audit Vault / Simulation</b>.
                    </p>
                </div>
                <!-- Dual Mode Selector -->
                <div class="flex bg-slate-900/90 border border-slate-700/80 rounded-xl p-1 self-start md:self-auto">
                    <button id="btn-mode-dataset" onclick="switchMode('dataset')" class="px-4 py-2 rounded-lg text-xs font-bold transition-all bg-blue-600 text-white shadow-md">
                        📊 Mode 1: Dataset Analytics
                    </button>
                    <button id="btn-mode-live" onclick="switchMode('live')" class="px-4 py-2 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-slate-200">
                        ⚡ Mode 2: Live Operations
                    </button>
                </div>
            </div>
        </section>

        <!-- ============================================================= -->
        <!-- MODE 1 CONTAINER: DATASET ANALYTICS (ACTUAL KAGGLE DATASET)   -->
        <!-- ============================================================= -->
        <div id="container-mode-dataset" class="space-y-6">
            <!-- Dataset Ground Truth KPIs -->
            <section class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Total Records</div>
                    <div class="text-2xl font-bold mt-1 text-slate-100 font-mono">284,807</div>
                    <div class="text-[11px] text-slate-500">Kaggle benchmark data</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Fraud Cases</div>
                    <div class="text-2xl font-bold mt-1 text-rose-400 font-mono">492</div>
                    <div class="text-[11px] text-slate-500">Confirmed Class 1</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Legitimate</div>
                    <div class="text-2xl font-bold mt-1 text-emerald-400 font-mono">284,315</div>
                    <div class="text-[11px] text-slate-500">Confirmed Class 0</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Class Imbalance</div>
                    <div class="text-2xl font-bold mt-1 text-amber-400 font-mono">0.17%</div>
                    <div class="text-[11px] text-slate-500">1 fraud per 578 txns</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Avg Fraud Amount</div>
                    <div class="text-2xl font-bold mt-1 text-slate-100 font-mono">$122.21</div>
                    <div class="text-[11px] text-slate-500">vs Legit $88.29</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-dataset">REAL DATASET</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Median Fraud</div>
                    <div class="text-2xl font-bold mt-1 text-orange-400 font-mono">$9.25</div>
                    <div class="text-[11px] text-slate-500">vs Legit $22.00 (Probing)</div>
                </div>
            </section>

            <!-- Real Dataset Visualizations -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Amount Distribution Comparison -->
                <div class="glass-card p-5 rounded-2xl">
                    <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                        <div>
                            <h3 class="text-sm font-bold text-slate-100">Fraud vs Legitimate Spending Distribution</h3>
                            <p class="text-xs text-slate-400">Proves fraudsters execute micro-probing transactions (&lt; $10) at double the legit rate</p>
                        </div>
                        <span class="provenance-tag tag-dataset">ACTUAL DATA</span>
                    </div>
                    <div class="mt-4 space-y-3" id="amount-binned-container">
                        <div class="text-xs text-slate-400 text-center py-4">Loading distribution metrics...</div>
                    </div>
                </div>

                <!-- 24-Hour Cyclic Fraud Activity -->
                <div class="glass-card p-5 rounded-2xl">
                    <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                        <div>
                            <h3 class="text-sm font-bold text-slate-100">Cyclic 24-Hour Fraud Frequency</h3>
                            <p class="text-xs text-slate-400">Reveals nocturnal fraud spikes during early morning hours (2 AM – 5 AM)</p>
                        </div>
                        <span class="provenance-tag tag-dataset">ACTUAL DATA</span>
                    </div>
                    <div class="mt-4 space-y-2.5 max-h-[290px] overflow-y-auto pr-1" id="time-cyclic-container">
                        <div class="text-xs text-slate-400 text-center py-4">Loading temporal dynamics...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- MODE 2 CONTAINER: LIVE OPERATIONS & INVESTIGATION VAULT       -->
        <!-- ============================================================= -->
        <div id="container-mode-live" class="space-y-6 hidden">
            <!-- Telemetry KPIs -->
            <section class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-vault">STORED IN VAULT</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Evaluated Total</div>
                    <div class="text-2xl font-bold mt-1 text-slate-100 font-mono" id="kpi-total">--</div>
                    <div class="text-[11px] text-slate-500">Live vault records</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-model">MODEL OUTPUT</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Fraud Flags</div>
                    <div class="text-2xl font-bold mt-1 text-rose-400 font-mono" id="kpi-fraud">--</div>
                    <div class="text-[11px] text-slate-500">Cutoff &ge; 0.50</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-model">MODEL OUTPUT</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">High / Critical</div>
                    <div class="text-2xl font-bold mt-1 text-orange-400 font-mono" id="kpi-highrisk">--</div>
                    <div class="text-[11px] text-slate-500">Score &ge; 61</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-vault">STORED IN VAULT</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Flag Rate</div>
                    <div class="text-2xl font-bold mt-1 text-slate-100 font-mono" id="kpi-rate">--%</div>
                    <div class="text-[11px] text-slate-500">Of processed volume</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-vault">STORED IN VAULT</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Avg Amount</div>
                    <div class="text-2xl font-bold mt-1 text-slate-100 font-mono" id="kpi-amount">$0.00</div>
                    <div class="text-[11px] text-slate-500">Per transaction</div>
                </div>
                <div class="glass-card p-4 rounded-xl">
                    <span class="provenance-tag tag-vault">AUDIT TRAIL</span>
                    <div class="text-xs font-semibold text-slate-400 uppercase mt-1">Analyst Reviewed</div>
                    <div class="text-2xl font-bold mt-1 text-sky-400 font-mono" id="kpi-reviewed">--</div>
                    <div class="text-[11px] text-slate-500">Cleared in audit log</div>
                </div>
            </section>
        </div>

        <!-- Two Column Operational Layout (Universal) -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- Left Panel: Transaction Analyzer & Threshold Playground -->
            <section class="lg:col-span-7 space-y-6">

                <!-- Risk Scoring Terminal -->
                <div class="glass-card p-6 rounded-2xl">
                    <div class="flex items-center justify-between pb-4 border-b border-slate-800">
                        <div>
                            <span class="provenance-tag tag-model">MODEL INFERENCE</span>
                            <h2 class="text-lg font-bold text-slate-100 mt-1">Real-Time Risk Scoring Terminal</h2>
                            <p class="text-xs text-slate-400">Score transactions using trained Random Forest &amp; SMOTE pipeline</p>
                        </div>
                        <button onclick="triggerSimulation()" class="px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 rounded-lg text-xs font-bold transition-all">
                            ⚡ Stream Simulation
                        </button>
                    </div>

                    <!-- Preset Picker -->
                    <div class="mt-4">
                        <label class="block text-xs font-semibold text-slate-400 uppercase mb-2">Load Benchmark Sample (Real Kaggle Dataset):</label>
                        <select id="preset-selector" onchange="loadSelectedPreset()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500 font-mono">
                            <option value="legit">Legitimate Retail Transaction ($42.50) [Class 0]</option>
                            <option value="suspicious">Borderline Suspicious Transaction ($580.00) [Class 0 Borderline]</option>
                            <option value="fraud" selected>High-Risk Fraud Incident ($1,250.00) [Class 1 Real Fraud]</option>
                        </select>
                    </div>

                    <!-- Input Fields -->
                    <div class="grid grid-cols-2 gap-4 mt-4">
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 mb-1">Amount ($)</label>
                            <input type="number" id="input-amount" value="1250.00" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-slate-200 focus:outline-none focus:border-blue-500">
                        </div>
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 mb-1">Time Elapsed (sec)</label>
                            <input type="number" id="input-time" value="406.0" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-slate-200 focus:outline-none focus:border-blue-500">
                        </div>
                    </div>

                    <!-- Action Button -->
                    <button onclick="scoreTransaction()" class="mt-5 w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold rounded-xl text-sm transition-all shadow-lg shadow-blue-500/25">
                        ⚡ Run Model Risk Evaluation
                    </button>

                    <!-- Score Result Card with Risk Gauge -->
                    <div id="result-box" class="mt-6 p-4 rounded-xl border border-slate-700 bg-slate-900/80 hidden space-y-4">
                        <div class="flex items-center justify-between">
                            <div>
                                <span id="res-badge" class="px-2.5 py-0.5 rounded-full text-xs font-bold">CRITICAL RISK</span>
                                <div class="text-3xl font-extrabold mt-1 text-slate-100 font-mono" id="res-score">92 <span class="text-sm font-normal text-slate-400">/ 100</span></div>
                                <div class="text-xs text-slate-400 mt-0.5" id="res-prob">Fraud Confidence: 92.0%</div>
                            </div>
                            <div class="text-right">
                                <div class="text-[10px] text-slate-500 font-semibold tracking-wider uppercase">MODEL DECISION</div>
                                <div class="text-sm font-bold text-rose-400 font-mono mt-0.5" id="res-decision">🚨 FLAGGED FOR REVIEW</div>
                            </div>
                        </div>

                        <!-- Visual Segmented Risk Gauge -->
                        <div class="space-y-1">
                            <div class="flex justify-between text-[10px] font-mono text-slate-400 font-semibold">
                                <span>LOW (0–30)</span>
                                <span>MEDIUM (31–60)</span>
                                <span>HIGH (61–80)</span>
                                <span>CRITICAL (81–100)</span>
                            </div>
                            <div class="flex h-2.5 rounded-full overflow-hidden bg-slate-800">
                                <div class="w-[30%] bg-emerald-500"></div>
                                <div class="w-[30%] bg-amber-500"></div>
                                <div class="w-[20%] bg-orange-500"></div>
                                <div class="w-[20%] bg-rose-500"></div>
                            </div>
                        </div>

                        <!-- Horizontal Force Attribution (Graph 10) -->
                        <div class="pt-3 border-t border-slate-800 space-y-2">
                            <div class="text-xs font-bold text-slate-200">Why Was This Transaction Flagged? (SHAP Attributions)</div>
                            <div class="text-xs text-slate-300 leading-relaxed" id="res-narrative">Analyzing feature forces...</div>
                            <div id="res-force-bars" class="space-y-1.5 pt-2"></div>
                        </div>
                    </div>
                </div>

                <!-- Interactive Decision Threshold Playground -->
                <div class="glass-card p-6 rounded-2xl">
                    <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                        <div>
                            <span class="provenance-tag tag-model">MODEL PLAYGROUND</span>
                            <h3 class="text-base font-bold text-slate-100 mt-1">Decision Threshold Playground</h3>
                            <p class="text-xs text-slate-400">Dynamically shift cutoff to inspect Precision, Recall, and False Alarms</p>
                        </div>
                        <span id="thresh-display-val" class="font-mono text-sm font-bold text-blue-400 bg-blue-500/10 px-2.5 py-1 rounded-md border border-blue-500/20">0.50</span>
                    </div>

                    <div class="mt-4">
                        <input type="range" id="thresh-slider" min="0.01" max="0.99" step="0.01" value="0.50" oninput="updateThresholdPlayground(this.value)" class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500">
                    </div>

                    <!-- Dynamic Metrics from 99-step test set sweep -->
                    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-center">
                        <div class="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                            <div class="text-[10px] text-slate-400 font-semibold uppercase">Recall (Caught)</div>
                            <div class="text-lg font-bold text-blue-400 font-mono mt-0.5" id="th-recall">75.8%</div>
                        </div>
                        <div class="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                            <div class="text-[10px] text-slate-400 font-semibold uppercase">Precision (Accuracy)</div>
                            <div class="text-lg font-bold text-emerald-400 font-mono mt-0.5" id="th-precision">92.3%</div>
                        </div>
                        <div class="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                            <div class="text-[10px] text-slate-400 font-semibold uppercase">F1-Score</div>
                            <div class="text-lg font-bold text-amber-400 font-mono mt-0.5" id="th-f1">0.832</div>
                        </div>
                        <div class="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                            <div class="text-[10px] text-slate-400 font-semibold uppercase">False Alarms (FP)</div>
                            <div class="text-lg font-bold text-rose-400 font-mono mt-0.5" id="th-fp">6</div>
                        </div>
                    </div>

                    <!-- Trade-off explanation badge -->
                    <div id="th-mode-badge" class="mt-3 p-2.5 rounded-lg text-xs font-semibold bg-blue-500/10 text-blue-300 border border-blue-500/20">
                        Balanced Production Mode: Optimal harmonic F1 trade-off for standard production fraud queuing.
                    </div>
                </div>
            </section>

            <!-- Right Panel: Live Incident Feed & Audit Vault -->
            <section class="lg:col-span-5 space-y-6">
                <div class="glass-card p-6 rounded-2xl">
                    <div class="flex items-center justify-between pb-4 border-b border-slate-800">
                        <div>
                            <span class="provenance-tag tag-vault">AUDIT TRAIL</span>
                            <h2 class="text-lg font-bold text-slate-100 mt-1">Live Incident Vault</h2>
                            <p class="text-xs text-slate-400">Stored evaluation records and analyst audit trail</p>
                        </div>
                        <button onclick="fetchKPIsAndFeed()" class="text-xs text-slate-400 hover:text-slate-200">↻ Refresh</button>
                    </div>

                    <div id="feed-container" class="mt-4 space-y-3 max-h-[580px] overflow-y-auto pr-1">
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
                PCA Components V1–V28 &middot; Zero Data Leakage &middot; Imbalanced-Learn SMOTE Pipeline
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

        function switchMode(mode) {
            const btnDataset = document.getElementById('btn-mode-dataset');
            const btnLive = document.getElementById('btn-mode-live');
            const cDataset = document.getElementById('container-mode-dataset');
            const cLive = document.getElementById('container-mode-live');

            if (mode === 'dataset') {
                btnDataset.className = "px-4 py-2 rounded-lg text-xs font-bold transition-all bg-blue-600 text-white shadow-md";
                btnLive.className = "px-4 py-2 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-slate-200";
                cDataset.classList.remove('hidden');
                cLive.classList.add('hidden');
            } else {
                btnLive.className = "px-4 py-2 rounded-lg text-xs font-bold transition-all bg-emerald-600 text-white shadow-md";
                btnDataset.className = "px-4 py-2 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-slate-200";
                cLive.classList.remove('hidden');
                cDataset.classList.add('hidden');
            }
        }

        function loadSelectedPreset() {
            const key = document.getElementById('preset-selector').value;
            currentVector = { ...PRESET_VECTORS[key] };
            document.getElementById('input-amount').value = currentVector.Amount;
            document.getElementById('input-time').value = currentVector.Time;
        }

        async function fetchDatasetIntelligence() {
            try {
                const res = await fetch('/dataset-intelligence');
                if (res.ok) {
                    const data = await res.json();
                    renderAmountBars(data.amount_distribution || []);
                    renderTimeBars(data.time_cyclic_24h || []);
                }
            } catch (e) {
                console.error("Dataset intelligence fetch error:", e);
            }
        }

        function renderAmountBars(dist) {
            const container = document.getElementById('amount-binned-container');
            if (!dist || dist.length === 0) return;

            container.innerHTML = dist.map(d => `
                <div class="space-y-1 text-xs">
                    <div class="flex justify-between font-mono">
                        <span class="text-slate-300 font-semibold">${d.bin}</span>
                        <span class="text-slate-400">Legit: <b class="text-emerald-400">${d.legit_pct}%</b> | Fraud: <b class="text-rose-400">${d.fraud_pct}%</b></span>
                    </div>
                    <div class="flex h-2 rounded-full overflow-hidden bg-slate-800 gap-0.5">
                        <div style="width: ${d.legit_pct}%" class="bg-emerald-500 rounded-l-full"></div>
                        <div style="width: ${d.fraud_pct}%" class="bg-rose-500 rounded-r-full"></div>
                    </div>
                </div>
            `).join('');
        }

        function renderTimeBars(timeData) {
            const container = document.getElementById('time-cyclic-container');
            if (!timeData || timeData.length === 0) return;

            container.innerHTML = timeData.map(d => {
                const isNocturnalPeak = d.hour_of_day >= 2 && d.hour_of_day <= 5;
                const barColor = isNocturnalPeak ? 'bg-rose-500' : 'bg-blue-500';
                return `
                <div class="flex items-center text-xs gap-3">
                    <span class="font-mono text-slate-400 w-12">${String(d.hour_of_day).padStart(2, '0')}:00</span>
                    <div class="flex-1 bg-slate-800 h-2 rounded-full overflow-hidden">
                        <div style="width: ${Math.min(d.fraud_count * 3, 100)}%" class="${barColor} h-full rounded-full"></div>
                    </div>
                    <span class="font-mono w-16 text-right ${isNocturnalPeak ? 'text-rose-400 font-bold' : 'text-slate-400'}">${d.fraud_count} frauds</span>
                </div>
                `;
            }).join('');
        }

        async function fetchKPIsAndFeed() {
            try {
                const [kpiRes, txnRes] = await Promise.all([
                    fetch('/kpis'),
                    fetch('/transactions?limit=8')
                ]);
                if (kpiRes.ok) {
                    const kpis = await kpiRes.json();
                    document.getElementById('kpi-total').textContent = (kpis.total_transactions || 0).toLocaleString();
                    document.getElementById('kpi-fraud').textContent = (kpis.fraud_flags || 0).toLocaleString();
                    document.getElementById('kpi-highrisk').textContent = (kpis.high_risk_transactions || 0).toLocaleString();
                    document.getElementById('kpi-rate').textContent = (kpis.fraud_rate_pct || 0).toFixed(1) + '%';
                    document.getElementById('kpi-amount').textContent = '$' + (kpis.avg_amount || 0).toFixed(2);
                    document.getElementById('kpi-reviewed').textContent = (kpis.transactions_reviewed || 0).toLocaleString();
                }
                if (txnRes.ok) {
                    const txns = await txnRes.json();
                    renderFeed(txns);
                }
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
                        <div class="mt-1 font-mono text-[11px] ${isFraud ? 'text-rose-400 font-bold' : 'text-emerald-400'}">${isFraud ? '🚨 FLAGGED' : '✅ APPROVED'}</div>
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
                badge.className = `px-2.5 py-0.5 rounded-full text-xs font-bold ${
                    data.risk_level === 'CRITICAL' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' :
                    data.risk_level === 'HIGH' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                    data.risk_level === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                    'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                }`;

                document.getElementById('res-decision').textContent = isFraud ? '🚨 FLAGGED FOR REVIEW' : '✅ APPROVED LEGITIMATE';
                document.getElementById('res-decision').className = `text-sm font-bold font-mono ${isFraud ? 'text-rose-400' : 'text-emerald-400'}`;

                // Fetch XAI explanation
                try {
                    const xaiRes = await fetch('/explain', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(currentVector)
                    });
                    if (xaiRes.ok) {
                        const xaiData = await xaiRes.json();
                        document.getElementById('res-narrative').innerHTML = `<b>Attribution Analysis:</b> ${xaiData.narrative}`;
                        
                        // Render force contribution bars (Graph 10)
                        const barContainer = document.getElementById('res-force-bars');
                        const topFeats = xaiData.top_features || [];
                        barContainer.innerHTML = topFeats.slice(0, 5).map(f => {
                            const isPos = f.contribution > 0;
                            const barColor = isPos ? 'bg-rose-500' : 'bg-emerald-500';
                            const width = Math.min(Math.abs(f.contribution) * 100, 100);
                            return `
                            <div class="text-xs">
                                <div class="flex justify-between font-mono text-[11px]">
                                    <span class="text-slate-300">${f.feature} (val: ${f.raw_value})</span>
                                    <span class="${isPos ? 'text-rose-400' : 'text-emerald-400'}">${isPos ? '+' : ''}${f.contribution.toFixed(3)}</span>
                                </div>
                                <div class="h-1.5 bg-slate-800 rounded-full overflow-hidden mt-0.5">
                                    <div class="${barColor} h-full rounded-full" style="width: ${width}%"></div>
                                </div>
                            </div>
                            `;
                        }).join('');
                    }
                } catch (e) {
                    console.error("XAI error:", e);
                }

                fetchKPIsAndFeed();
            } catch (err) {
                document.getElementById('res-narrative').textContent = "Error: " + err.message;
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
                        badge.className = "mt-3 p-2.5 rounded-lg text-xs font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/20";
                        badge.textContent = "⚠️ High Sensitivity Mode: Maximum fraud recall. Captures near-all fraud but increases customer verification load.";
                    } else if (data.operational_mode === 'CONSERVATIVE') {
                        badge.className = "mt-3 p-2.5 rounded-lg text-xs font-semibold bg-orange-500/10 text-orange-300 border border-orange-500/20";
                        badge.textContent = "⚠️ Conservative Mode: Minimum customer friction. Reduces false alarms, but stealthy fraud may escape detection.";
                    } else {
                        badge.className = "mt-3 p-2.5 rounded-lg text-xs font-semibold bg-blue-500/10 text-blue-300 border border-blue-500/20";
                        badge.textContent = "✅ Balanced Production Mode: Optimal harmonic F1 trade-off for standard production fraud queuing.";
                    }
                }
            } catch (e) {
                console.error("Threshold query error:", e);
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
        fetchDatasetIntelligence();
        fetchKPIsAndFeed();
        updateThresholdPlayground(0.50);
    </script>
</body>
</html>
"""
