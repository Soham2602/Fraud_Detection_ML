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

from fastapi.responses import HTMLResponse, JSONResponse
from starlette.requests import Request
from src.api import app
from src.html_dashboard import HTML_DASHBOARD


@app.middleware("http")
async def normalize_vercel_paths(request: Request, call_next):
    """
    Normalize Vercel rewrite parameters and entrypoint path prefixes so FastAPI
    routes match the client's originally intended resource (/health, /docs, /predict).
    """
    v_path = request.query_params.get("__vercel_path")
    if v_path:
        path = "/" + v_path.lstrip("/")
    else:
        matched = (
            request.headers.get("x-matched-path")
            or request.headers.get("x-invoke-path")
            or request.headers.get("x-rewrite-url")
            or request.headers.get("x-original-url")
            or request.headers.get("x-forwarded-uri")
        )
        raw = matched if matched else request.scope.get("path", "")
        path = raw.split("?")[0] if raw else "/"

        for prefix in ["/api/index.py", "/api/index", "/api"]:
            if path == prefix:
                path = "/"
                break
            elif path.startswith(prefix + "/"):
                path = path[len(prefix):]
                break

    request.scope["path"] = path or "/"
    return await call_next(request)


@app.get("/api/index.py", response_class=HTMLResponse, tags=["Web Portal"])
@app.get("/api/index", response_class=HTMLResponse, tags=["Web Portal"])
@app.get("/api", response_class=HTMLResponse, tags=["Web Portal"])
@app.get("/api/", response_class=HTMLResponse, tags=["Web Portal"])
def get_sentinel_dashboard():
    """Serve the complete responsive SENTINEL web portal for Vercel deployment."""
    return HTMLResponse(content=HTML_DASHBOARD)
