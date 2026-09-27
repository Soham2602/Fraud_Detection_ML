"""
tests/test_visualizations.py
----------------------------
Unit tests verifying that all Plotly figures in src/visualizations.py
construct valid, non-empty figures with expected data traces.
"""

import pytest
import plotly.graph_objects as go
from src.visualizations import (
    plot_class_imbalance,
    plot_amount_distribution,
    plot_amount_by_class_boxplot,
    plot_fraud_over_time,
    plot_cyclic_fraud_dynamics,
    plot_risk_distribution,
    plot_threshold_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importances,
    plot_shap_force_bars,
    plot_risk_gauge,
    plot_correlation_heatmap,
    plot_3d_feature_space,
    plot_what_if_comparison,
    plot_simulation_stream,
)


def test_plot_class_imbalance():
    fig = plot_class_imbalance(284315, 492)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "pie"


def test_plot_amount_distribution():
    binned = [
        {"bin": "$0–10", "legit_count": 100015, "fraud_count": 249, "legit_pct": 35.18, "fraud_pct": 50.61},
        {"bin": "$10–50", "legit_count": 90000, "fraud_count": 150, "legit_pct": 31.65, "fraud_pct": 30.49},
    ]
    fig = plot_amount_distribution(binned, use_log_scale=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_amount_by_class_boxplot():
    stats = {
        "legitimate": {"p25": 5.65, "median": 22.0, "p75": 77.05, "mean": 88.29, "min": 0.0, "p99": 1016.97},
        "fraudulent": {"p25": 1.0, "median": 9.25, "p75": 105.89, "mean": 122.21, "min": 0.0, "p99": 1357.43},
    }
    fig = plot_amount_by_class_boxplot(stats)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].type == "box"


def test_plot_fraud_over_time():
    hourly = [
        {"hour": 0, "fraud_count": 5, "legit_count": 5000, "fraud_rate_pct": 0.1},
        {"hour": 1, "fraud_count": 8, "legit_count": 4000, "fraud_rate_pct": 0.2},
    ]
    fig_count = plot_fraud_over_time(hourly, show_rate=False)
    assert isinstance(fig_count, go.Figure)
    assert len(fig_count.data) == 2

    fig_rate = plot_fraud_over_time(hourly, show_rate=True)
    assert isinstance(fig_rate, go.Figure)
    assert len(fig_rate.data) == 1


def test_plot_cyclic_fraud_dynamics():
    cyclic = [{"hour_of_day": h, "fraud_count": 2, "legit_count": 1000, "fraud_rate_pct": 0.2} for h in range(24)]
    fig = plot_cyclic_fraud_dynamics(cyclic)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_plot_risk_distribution():
    risk_dist = [
        {"bin_label": "0.0–0.1", "legit_count": 56000, "fraud_count": 2},
        {"bin_label": "0.9–1.0", "legit_count": 5, "fraud_count": 70},
    ]
    fig = plot_risk_distribution(risk_dist, threshold=0.5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_threshold_curves():
    sweep = [
        {"threshold": 0.1, "precision": 0.35, "recall": 0.85, "f1": 0.50},
        {"threshold": 0.5, "precision": 0.92, "recall": 0.75, "f1": 0.83},
        {"threshold": 0.9, "precision": 0.98, "recall": 0.60, "f1": 0.75},
    ]
    fig = plot_threshold_curves(sweep, current_threshold=0.5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_plot_confusion_matrix():
    cm = {"tn": 56645, "fp": 6, "fn": 23, "tp": 72}
    fig = plot_confusion_matrix(cm, threshold=0.5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_plot_roc_curve():
    roc_pts = [
        {"fpr": 0.0, "tpr": 0.0, "threshold": 1.0},
        {"fpr": 0.0001, "tpr": 0.75, "threshold": 0.5},
        {"fpr": 1.0, "tpr": 1.0, "threshold": 0.0},
    ]
    fig = plot_roc_curve(roc_pts, roc_auc=0.9664)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_precision_recall_curve():
    pr_pts = [
        {"recall": 0.0, "precision": 1.0, "threshold": 1.0},
        {"recall": 0.75, "precision": 0.92, "threshold": 0.5},
        {"recall": 1.0, "precision": 0.0017, "threshold": 0.0},
    ]
    fig = plot_precision_recall_curve(pr_pts, pr_auc=0.8096)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_plot_feature_importances():
    importances = [
        {"feature": "V14", "importance": 0.25},
        {"feature": "V10", "importance": 0.18},
        {"feature": "V12", "importance": 0.12},
    ]
    fig = plot_feature_importances(importances, top_n=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_plot_shap_force_bars():
    top_feats = [
        {"feature": "V14", "contribution": 0.35, "raw_value": -4.28},
        {"feature": "V16", "contribution": -0.12, "raw_value": 0.85},
    ]
    fig = plot_shap_force_bars(top_feats)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_plot_risk_gauge():
    fig = plot_risk_gauge(82, threshold=0.5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "indicator"


def test_plot_3d_feature_space():
    sample_data = {
        "index": [1, 2],
        "Class": [0, 1],
        "Amount": [50.0, 120.0],
        "Time": [100.0, 200.0],
        "V14": [0.5, -4.2],
        "V10": [-0.1, -2.5],
        "V12": [0.2, -3.1],
    }
    fig = plot_3d_feature_space(sample_data, x_feature="V14", y_feature="V10", z_feature="V12")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter3d"


def test_plot_what_if_comparison():
    fig = plot_what_if_comparison(0.25, 0.85, threshold=0.5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_plot_simulation_stream():
    records = [
        {"fraud_probability": 0.1, "amount": 25.0, "is_flagged": False},
        {"fraud_probability": 0.9, "amount": 150.0, "is_flagged": True},
    ]
    fig = plot_simulation_stream(records, threshold=0.5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
