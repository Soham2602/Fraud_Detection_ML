"""
database.py
-----------
SQLite persistence layer for SENTINEL — Fraud Intelligence Platform.
Stores transactions, alerts, analyst reviews, and audit trails.
Safely falls back to /tmp for Vercel / serverless deployments.
"""

import sys
import sqlite3
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import DB_PATH


def get_db_connection() -> sqlite3.Connection:
    """Create and return a thread-safe connection to the SQLite database."""
    try:
        if DB_PATH.parent.exists():
            pass
        else:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    except Exception:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize SQLite database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS transactions (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        amount REAL NOT NULL,
        time_elapsed REAL NOT NULL,
        prediction INTEGER NOT NULL,
        is_flagged INTEGER NOT NULL,
        fraud_probability REAL NOT NULL,
        risk_score INTEGER NOT NULL,
        risk_level TEXT NOT NULL,
        threshold_used REAL NOT NULL,
        features_json TEXT NOT NULL,
        channel TEXT DEFAULT 'POS Terminal',
        category TEXT DEFAULT 'Retail',
        location TEXT DEFAULT 'Global',
        status TEXT DEFAULT 'Pending',
        is_simulated INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        risk_score INTEGER NOT NULL,
        risk_level TEXT NOT NULL,
        fraud_probability REAL NOT NULL,
        status TEXT DEFAULT 'Open',
        assigned_to TEXT DEFAULT 'Unassigned',
        resolved_at TEXT,
        FOREIGN KEY (transaction_id) REFERENCES transactions (id)
    );

    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        reviewed_at TEXT NOT NULL,
        reviewer TEXT NOT NULL,
        action TEXT NOT NULL,
        notes TEXT,
        FOREIGN KEY (transaction_id) REFERENCES transactions (id)
    );

    CREATE TABLE IF NOT EXISTS model_metadata_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()


def save_transaction(record: Dict[str, Any], is_simulated: bool = False) -> str:
    """Save an evaluated transaction and create an alert if flagged."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    txn_id = record.get("transaction_id") or f"TXN-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"
    now = record.get("timestamp") or datetime.datetime.now().isoformat()
    features_json = json.dumps(record.get("features_raw", {}))

    cursor.execute("""
    INSERT OR REPLACE INTO transactions (
        id, timestamp, amount, time_elapsed, prediction, is_flagged,
        fraud_probability, risk_score, risk_level, threshold_used,
        features_json, channel, category, location, status, is_simulated
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        txn_id,
        now,
        float(record.get("amount", 0.0)),
        float(record.get("time", 0.0)),
        int(record.get("prediction", 0)),
        1 if record.get("is_flagged") else 0,
        float(record.get("fraud_probability", 0.0)),
        int(record.get("risk_score", 0)),
        str(record.get("risk_level", "LOW")),
        float(record.get("threshold_used", 0.50)),
        features_json,
        str(record.get("demo_channel", "POS Terminal")),
        str(record.get("demo_category", "Retail")),
        str(record.get("demo_location", "Global")),
        "Pending" if record.get("is_flagged") else "Approved",
        1 if is_simulated else 0,
    ))

    # Auto-create alert if flagged
    if record.get("is_flagged"):
        cursor.execute("""
        INSERT INTO alerts (transaction_id, created_at, risk_score, risk_level, fraud_probability, status)
        VALUES (?, ?, ?, ?, ?, 'Open')
        """, (
            txn_id,
            now,
            int(record.get("risk_score", 0)),
            str(record.get("risk_level", "HIGH")),
            float(record.get("fraud_probability", 0.0)),
        ))

    conn.commit()
    conn.close()
    return txn_id


def get_recent_transactions(
    limit: int = 50,
    risk_level: Optional[str] = None,
    status: Optional[str] = None,
    only_flagged: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieve recent transactions with optional filters."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM transactions WHERE 1=1"
    params: List[Any] = []

    if risk_level:
        query += " AND risk_level = ?"
        params.append(risk_level)

    if status:
        query += " AND status = ?"
        params.append(status)

    if only_flagged:
        query += " AND is_flagged = 1"

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_transaction_by_id(txn_id: str) -> Optional[Dict[str, Any]]:
    """Fetch full transaction details including parsed features and review history."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM transactions WHERE id = ?", (txn_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    data = dict(row)
    try:
        data["features"] = json.loads(data["features_json"])
    except Exception:
        data["features"] = {}

    # Fetch reviews
    cursor.execute("SELECT * FROM reviews WHERE transaction_id = ? ORDER BY reviewed_at DESC", (txn_id,))
    data["reviews"] = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return data


def add_review_action(
    txn_id: str, action: str, reviewer: str = "Lead Fraud Analyst", notes: str = ""
) -> bool:
    """Record an analyst action (Mark Reviewed, Mark Legitimate, Escalate) and update status."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    now = datetime.datetime.now().isoformat()

    # Determine new transaction status
    status_map = {
        "Mark as reviewed": "Reviewed",
        "Mark legitimate": "Cleared Legitimate",
        "Escalate": "Escalated for Investigation",
        "Add note": None,
    }
    new_status = status_map.get(action)

    if new_status:
        cursor.execute("UPDATE transactions SET status = ? WHERE id = ?", (new_status, txn_id))
        alert_status = "Resolved" if action in ["Mark as reviewed", "Mark legitimate"] else "Escalated"
        cursor.execute("UPDATE alerts SET status = ?, resolved_at = ? WHERE transaction_id = ?", (alert_status, now, txn_id))

    cursor.execute("""
    INSERT INTO reviews (transaction_id, reviewed_at, reviewer, action, notes)
    VALUES (?, ?, ?, ?, ?)
    """, (txn_id, now, reviewer, action, notes))

    conn.commit()
    conn.close()
    return True


def get_alerts(status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch active alerts joined with transaction metadata."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    SELECT a.*, t.amount, t.category, t.channel, t.location, t.status as txn_status
    FROM alerts a
    JOIN transactions t ON a.transaction_id = t.id
    """
    params: List[Any] = []

    if status:
        query += " WHERE a.status = ?"
        params.append(status)

    query += " ORDER BY a.created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_dashboard_kpis() -> Dict[str, Any]:
    """Compute aggregate KPIs for the Overview Dashboard."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total, SUM(is_flagged) as flagged, AVG(amount) as avg_amt FROM transactions")
    base = cursor.fetchone()
    total_txns = int(base["total"] or 0)
    flagged_txns = int(base["flagged"] or 0)
    avg_amount = float(base["avg_amt"] or 0.0)

    cursor.execute("SELECT COUNT(*) as high_risk FROM transactions WHERE risk_score >= 61")
    high_risk_txns = int(cursor.fetchone()["high_risk"] or 0)

    cursor.execute("SELECT COUNT(DISTINCT transaction_id) as reviewed FROM reviews")
    reviewed_count = int(cursor.fetchone()["reviewed"] or 0)

    cursor.execute("SELECT COUNT(*) as open_alerts FROM alerts WHERE status = 'Open'")
    open_alerts = int(cursor.fetchone()["open_alerts"] or 0)

    fraud_rate = (flagged_txns / total_txns * 100) if total_txns > 0 else 0.0

    conn.close()
    return {
        "total_transactions": total_txns,
        "fraud_flags": flagged_txns,
        "high_risk_transactions": high_risk_txns,
        "fraud_rate_pct": round(fraud_rate, 2),
        "avg_amount": round(avg_amount, 2),
        "transactions_reviewed": reviewed_count,
        "open_alerts": open_alerts,
    }


def seed_demo_database_if_empty() -> int:
    """Pre-populate the database with presets if completely empty, ensuring a populated dashboard on first launch."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM transactions")
    count = cursor.fetchone()["count"]
    conn.close()

    if count > 0:
        return count

    # Seed using sample_presets.json
    from config import SAMPLE_PRESETS_JSON, DEMO_CHANNELS, DEMO_CATEGORIES, DEMO_LOCATIONS
    from model_service import ModelService

    if not SAMPLE_PRESETS_JSON.exists():
        return 0

    with open(SAMPLE_PRESETS_JSON, "r") as f:
        presets = json.load(f)

    service = ModelService.get_instance()
    service.ensure_loaded()

    seeded = 0
    import random
    rng = random.Random(42)

    for i, p in enumerate(presets):
        res = service.predict_single(p["features"], transaction_id=p["id"])
        res["demo_channel"] = rng.choice(DEMO_CHANNELS)
        res["demo_category"] = rng.choice(DEMO_CATEGORIES)
        res["demo_location"] = rng.choice(DEMO_LOCATIONS)
        save_transaction(res, is_simulated=False)
        seeded += 1

    return seeded
