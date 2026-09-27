"""
api.py
------
FastAPI REST backend for SENTINEL — Fraud Intelligence Platform.
Provides endpoints for single inference, batch scoring, alerts,
investigation workflows, simulation, and model health.
Compatible with Vercel Serverless Functions via ASGI adapter.
"""

import sys
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI, HTTPException, Query, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from config import DEFAULT_THRESHOLD, FEATURE_COLUMNS, get_risk_level
from model_service import ModelService, ValidationError
from database import (
    init_db,
    save_transaction,
    get_recent_transactions,
    get_transaction_by_id,
    add_review_action,
    get_alerts,
    get_dashboard_kpis,
    seed_demo_database_if_empty,
)
from explainability import ExplainabilityEngine
from simulation import SimulationEngine

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database and demo presets are initialized on launch."""
    init_db()
    seed_demo_database_if_empty()
    yield

app = FastAPI(
    title="SENTINEL — Fraud Intelligence API",
    description="REST API for ML-powered fraud risk scoring, batch analytics, and alert management.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request Models
class TransactionInput(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since reference transaction")
    Amount: float = Field(..., ge=0.0, description="Transaction amount")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    threshold: Optional[float] = Field(default=DEFAULT_THRESHOLD, ge=0.01, le=0.99)
    save_to_db: Optional[bool] = False
    demo_channel: Optional[str] = "POS Terminal"
    demo_category: Optional[str] = "Retail"
    demo_location: Optional[str] = "Global"


class BatchInput(BaseModel):
    transactions: List[Dict[str, Any]]
    threshold: Optional[float] = Field(default=DEFAULT_THRESHOLD, ge=0.01, le=0.99)


class ReviewInput(BaseModel):
    transaction_id: str
    action: str = Field(..., description="Mark as reviewed, Mark legitimate, Escalate, or Add note")
    reviewer: str = "Lead Fraud Analyst"
    notes: Optional[str] = ""


class WhatIfInput(BaseModel):
    base_features: Dict[str, Any]
    modifications: Dict[str, Any]
    threshold: Optional[float] = DEFAULT_THRESHOLD




@app.get("/", response_class=HTMLResponse, tags=["Web Portal"])
def root(request: Request):
    """
    Serve the interactive SENTINEL web portal by default.
    If the client explicitly requests application/json without text/html, return API metadata.
    """
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(content={
            "platform": "SENTINEL — Fraud Intelligence Platform",
            "version": "2.0.0",
            "portal": "/",
            "documentation": "/docs",
            "health": "/health",
            "disclaimer": "Educational AI fraud intelligence platform. Do not enter real credit card numbers.",
        })
    try:
        from src.html_dashboard import HTML_DASHBOARD
    except ImportError:
        from html_dashboard import HTML_DASHBOARD
    return HTMLResponse(content=HTML_DASHBOARD)


@app.get("/portal", response_class=HTMLResponse, tags=["Web Portal"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Web Portal"])
def portal_view():
    from html_dashboard import HTML_DASHBOARD
    return HTMLResponse(content=HTML_DASHBOARD)


@app.get("/debug", tags=["Monitoring"])
def debug_info(request: Request):
    return {
        "headers": dict(request.headers),
        "scope_path": request.scope.get("path"),
        "url_path": request.url.path,
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    service = ModelService.get_instance()
    return {
        "status": "healthy" if service.is_loaded else "degraded",
        "model_loaded": service.is_loaded,
        "feature_count": len(service.feature_columns),
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.get("/model-info", tags=["Monitoring"])
def get_model_information():
    service = ModelService.get_instance()
    return service.get_model_info()


@app.get("/dataset-intelligence", tags=["Dataset Analytics"])
def get_dataset_intelligence():
    """Return precomputed Kaggle credit card dataset ground-truth statistics."""
    service = ModelService.get_instance()
    summary = service.get_dataset_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="Dataset summary artifact not found.")
    return summary


@app.get("/evaluation-bundle", tags=["Model Analytics"])
def get_evaluation_bundle():
    """Return precomputed test-fold evaluation curves and metrics."""
    service = ModelService.get_instance()
    bundle = service.get_evaluation_bundle()
    if not bundle:
        raise HTTPException(status_code=404, detail="Evaluation bundle artifact not found.")
    return bundle


@app.get("/threshold-analysis", tags=["Model Analytics"])
def analyze_threshold(threshold: float = Query(DEFAULT_THRESHOLD, ge=0.01, le=0.99)):
    """Evaluate empirical test-fold precision, recall, and error counts at any decision threshold."""
    service = ModelService.get_instance()
    return service.get_threshold_metrics(threshold=threshold)


@app.post("/predict", tags=["Inference"])
def predict_single_transaction(payload: TransactionInput):
    service = ModelService.get_instance()
    try:
        raw_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        threshold = raw_dict.pop("threshold", DEFAULT_THRESHOLD)
        save_db = raw_dict.pop("save_to_db", False)
        demo_channel = raw_dict.pop("demo_channel", "POS Terminal")
        demo_category = raw_dict.pop("demo_category", "Retail")
        demo_location = raw_dict.pop("demo_location", "Global")

        result = service.predict_single(raw_dict, threshold=threshold)
        result["demo_channel"] = demo_channel
        result["demo_category"] = demo_category
        result["demo_location"] = demo_location

        if save_db:
            save_transaction(result, is_simulated=False)

        return result
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail={"message": ve.message, "errors": ve.errors})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Inference"])
def predict_batch_transactions(payload: BatchInput):
    import pandas as pd
    service = ModelService.get_instance()
    try:
        df = pd.DataFrame(payload.transactions)
        enriched, summary = service.predict_batch(df, threshold=payload.threshold)
        return {
            "summary": summary,
            "results": enriched.to_dict(orient="records"),
        }
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail={"message": ve.message, "errors": ve.errors})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", tags=["Explainability"])
def explain_transaction(payload: Dict[str, Any]):
    engine = ExplainabilityEngine.get_instance()
    try:
        # Accept both flat feature dicts and nested {"features": {...}} payloads
        features_dict = payload.get("features", payload) if isinstance(payload.get("features"), dict) else payload
        top_k = int(payload.get("top_k", 8)) if "top_k" in payload else 8
        res = engine.explain_transaction(features_dict, top_k=top_k)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts", tags=["Investigation"])
def list_alerts(status_filter: Optional[str] = Query(None, alias="status"), limit: int = 50):
    return get_alerts(status=status_filter, limit=limit)


@app.get("/transactions", tags=["Investigation"])
def list_transactions(
    risk_level: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = 50,
):
    return get_recent_transactions(limit=limit, risk_level=risk_level, status=status_filter)


@app.get("/transactions/{txn_id}", tags=["Investigation"])
def get_transaction_details(txn_id: str):
    data = get_transaction_by_id(txn_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found.")
    return data


@app.post("/reviews", tags=["Investigation"])
def submit_review(payload: ReviewInput):
    success = add_review_action(
        payload.transaction_id,
        action=payload.action,
        reviewer=payload.reviewer,
        notes=payload.notes,
    )
    return {"success": success, "transaction_id": payload.transaction_id, "action": payload.action}


@app.get("/kpis", tags=["Monitoring"])
def get_kpis():
    return get_dashboard_kpis()


@app.post("/simulate", tags=["Simulation"])
def simulate_transactions(count: int = Query(1, ge=1, le=50), fraud_bias: float = Query(0.15, ge=0.0, le=1.0)):
    sim = SimulationEngine()
    if count == 1:
        return sim.generate_single_simulated_transaction(fraud_bias=fraud_bias, persist=True)
    return sim.generate_batch(count=count, fraud_bias=fraud_bias)


@app.post("/what-if", tags=["Simulation"])
def simulate_what_if(payload: WhatIfInput):
    sim = SimulationEngine()
    return sim.run_what_if_analysis(payload.base_features, payload.modifications, threshold=payload.threshold)
