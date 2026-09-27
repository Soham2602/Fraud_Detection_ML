"""
src/visualizations.py
---------------------
Reusable, interactive Plotly visualization module for SENTINEL.
Provides grounded, data-driven charts answering specific fraud detection
and machine learning questions with zero decorative slop.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# High-End Dark Cyber/Fintech Color Palette
THEME = {
    "bg_color": "rgba(11, 15, 25, 0.75)",
    "paper_color": "rgba(0, 0, 0, 0)",
    "grid_color": "rgba(255, 255, 255, 0.07)",
    "text_color": "#e2e8f0",
    "text_muted": "#94a3b8",
    "font_family": "Plus Jakarta Sans, -apple-system, sans-serif",
    "mono_family": "JetBrains Mono, monospace",
    "emerald": "#10b981",
    "blue": "#3b82f6",
    "indigo": "#6366f1",
    "amber": "#f59e0b",
    "orange": "#f97316",
    "rose": "#ef4444",
    "crimson": "#dc2626",
    "cyan": "#06b6d4",
}


def _apply_theme(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply consistent high-end dark styling to any Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family=THEME["font_family"], size=15, color=THEME["text_color"]),
            x=0.02,
            y=0.96,
        ),
        paper_bgcolor=THEME["paper_color"],
        plot_bgcolor=THEME["paper_color"],
        font=dict(family=THEME["font_family"], color=THEME["text_muted"], size=12),
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            font=dict(color=THEME["text_color"], size=11),
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor=THEME["grid_color"],
            zerolinecolor=THEME["grid_color"],
            tickfont=dict(color=THEME["text_muted"]),
        ),
        yaxis=dict(
            gridcolor=THEME["grid_color"],
            zerolinecolor=THEME["grid_color"],
            tickfont=dict(color=THEME["text_muted"]),
        ),
    )
    return fig


# -----------------------------------------------------------------------------
# 1. Class Imbalance Donut (Visualization 1)
# -----------------------------------------------------------------------------
def plot_class_imbalance(legit_count: int, fraud_count: int) -> go.Figure:
    """Interactive Donut chart demonstrating extreme 99.83% vs 0.17% class skew."""
    total = legit_count + fraud_count
    labels = ["Legitimate (Class 0)", "Fraudulent (Class 1)"]
    values = [legit_count, fraud_count]
    colors = [THEME["blue"], THEME["rose"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
                textinfo="percent",
                hoverinfo="label+value+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Proportion: %{percent}<extra></extra>",
            )
        ]
    )

    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:10px;color:#94a3b8'>Total TXNs</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(family=THEME["font_family"], size=16, color=THEME["text_color"]),
    )

    return _apply_theme(fig, "Transaction Classification Skew", height=320)


# -----------------------------------------------------------------------------
# 2. Transaction Amount Distribution (Visualization 2)
# -----------------------------------------------------------------------------
def plot_amount_distribution(binned_dist: List[Dict[str, Any]], use_log_scale: bool = False) -> go.Figure:
    """Grouped bar chart comparing spending behaviors of Legitimate vs Fraudulent transactions."""
    bins = [item["bin"] for item in binned_dist]
    legit_pcts = [item["legit_pct"] for item in binned_dist]
    fraud_pcts = [item["fraud_pct"] for item in binned_dist]
    legit_counts = [item["legit_count"] for item in binned_dist]
    fraud_counts = [item["fraud_count"] for item in binned_dist]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Legitimate (%)",
            x=bins,
            y=legit_pcts,
            marker_color=THEME["blue"],
            customdata=legit_counts,
            hovertemplate="<b>%{x} (Legit)</b><br>Share: %{y:.2f}%<br>Count: %{customdata:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Fraudulent (%)",
            x=bins,
            y=fraud_pcts,
            marker_color=THEME["rose"],
            customdata=fraud_counts,
            hovertemplate="<b>%{x} (Fraud)</b><br>Share: %{y:.2f}%<br>Count: %{customdata:,}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        xaxis_title="Amount Range",
        yaxis_title="Proportion of Class (%)",
        yaxis_type="log" if use_log_scale else "linear",
    )
    return _apply_theme(fig, "Transaction Amount Distribution by Class", height=350)


# -----------------------------------------------------------------------------
# 3. Fraud Amount vs Legitimate Amount Box Plot (Visualization 3)
# -----------------------------------------------------------------------------
def plot_amount_by_class_boxplot(stats_dict: Dict[str, Any], chart_type: str = "Box Plot") -> go.Figure:
    """Box plot comparison of amounts between legitimate and fraudulent classes using quantiles."""
    legit = stats_dict.get("legitimate", {})
    fraud = stats_dict.get("fraudulent", {})

    fig = go.Figure()

    # Create synthesized box traces matching exact quantiles
    fig.add_trace(
        go.Box(
            name="Legitimate",
            q1=[legit.get("p25", 5.65)],
            median=[legit.get("median", 22.0)],
            q3=[legit.get("p75", 77.05)],
            mean=[legit.get("mean", 88.29)],
            lowerfence=[legit.get("min", 0.0)],
            upperfence=[legit.get("p99", 1016.97)],
            marker_color=THEME["blue"],
            boxmean=True,
            hovertemplate="<b>Legitimate Amounts</b><br>Median: $%{median}<br>Mean: $%{mean}<br>P25: $%{q1}<br>P75: $%{q3}<br>P99 Cap: $%{upperfence}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Box(
            name="Fraudulent",
            q1=[fraud.get("p25", 1.0)],
            median=[fraud.get("median", 9.25)],
            q3=[fraud.get("p75", 105.89)],
            mean=[fraud.get("mean", 122.21)],
            lowerfence=[fraud.get("min", 0.0)],
            upperfence=[fraud.get("p99", 1357.43)],
            marker_color=THEME["rose"],
            boxmean=True,
            hovertemplate="<b>Fraudulent Amounts</b><br>Median: $%{median}<br>Mean: $%{mean}<br>P25: $%{q1}<br>P75: $%{q3}<br>P99 Cap: $%{upperfence}<extra></extra>",
        )
    )

    fig.update_layout(
        yaxis_title="Amount (Log Scale)",
        yaxis_type="log",
        showlegend=True,
    )
    return _apply_theme(fig, f"Amount Distribution Comparison ({chart_type})", height=350)


# -----------------------------------------------------------------------------
# 4. Fraud Activity Over Time (Visualization 4 & 5)
# -----------------------------------------------------------------------------
def plot_fraud_over_time(hourly_data: List[Dict[str, Any]], show_rate: bool = False) -> go.Figure:
    """Timeline chart across the 48-hour recording window."""
    hours = [item["hour"] for item in hourly_data]
    fraud_counts = [item["fraud_count"] for item in hourly_data]
    fraud_rates = [item["fraud_rate_pct"] for item in hourly_data]
    legit_counts = [item["legit_count"] for item in hourly_data]

    fig = go.Figure()

    if show_rate:
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=fraud_rates,
                mode="lines+markers",
                name="Fraud Rate (%)",
                line=dict(color=THEME["rose"], width=2.5),
                marker=dict(size=6, color=THEME["amber"]),
                hovertemplate="<b>Hour %{x}:00</b><br>Fraud Rate: %{y:.3f}%<extra></extra>",
            )
        )
        fig.update_layout(yaxis_title="Fraud Rate (%)", xaxis_title="Elapsed Time (Hours)")
    else:
        fig.add_trace(
            go.Bar(
                x=hours,
                y=legit_counts,
                name="Legitimate Volume",
                marker_color="rgba(59, 130, 246, 0.35)",
                yaxis="y",
                hovertemplate="<b>Hour %{x}:00</b><br>Legit Volume: %{y:,}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=fraud_counts,
                name="Fraud Incidents",
                mode="lines+markers",
                line=dict(color=THEME["rose"], width=2.5),
                marker=dict(size=6, color=THEME["rose"]),
                yaxis="y2",
                hovertemplate="<b>Hour %{x}:00</b><br>Fraud Count: %{y}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title="Elapsed Time (Hours 0–47)",
            yaxis=dict(title="Legitimate Transactions", gridcolor=THEME["grid_color"]),
            yaxis2=dict(
                title="Fraud Incidents",
                overlaying="y",
                side="right",
                showgrid=False,
                tickfont=dict(color=THEME["rose"]),
            ),
        )

    return _apply_theme(fig, "Temporal Transaction & Fraud Activity (48 Hours)", height=350)


# -----------------------------------------------------------------------------
# 5. 24-Hour Cyclic Diurnal Fraud Dynamics (Visualization 5)
# -----------------------------------------------------------------------------
def plot_cyclic_fraud_dynamics(cyclic_data: List[Dict[str, Any]]) -> go.Figure:
    """Diurnal hourly plot showing early morning fraud spikes (02:00 - 05:00 UTC)."""
    hours = [f"{item['hour_of_day']:02d}:00" for item in cyclic_data]
    rates = [item["fraud_rate_pct"] for item in cyclic_data]
    counts = [item["fraud_count"] for item in cyclic_data]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hours,
            y=rates,
            marker_color=[THEME["rose"] if r > 0.3 else THEME["blue"] for r in rates],
            customdata=counts,
            hovertemplate="<b>Time: %{x}</b><br>Fraud Rate: %{y:.3f}%<br>Incidents: %{customdata}<extra></extra>",
        )
    )

    # Reference threshold line for mean fraud rate
    fig.add_hline(
        y=0.1727,
        line_dash="dash",
        line_color=THEME["amber"],
        annotation_text="Global Baseline (0.173%)",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Hour of Day (UTC)",
        yaxis_title="Fraud Rate (%)",
    )
    return _apply_theme(fig, "24-Hour Cyclic Diurnal Fraud Dynamics", height=340)


# -----------------------------------------------------------------------------
# 6. Model Risk Score Distribution (Visualization 6)
# -----------------------------------------------------------------------------
def plot_risk_distribution(risk_dist: List[Dict[str, Any]], threshold: float = 0.5) -> go.Figure:
    """Empirical bimodal distribution of model probabilities across the test set."""
    labels = [item["bin_label"] for item in risk_dist]
    legit_counts = [item["legit_count"] for item in risk_dist]
    fraud_counts = [item["fraud_count"] for item in risk_dist]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Legitimate",
            x=labels,
            y=legit_counts,
            marker_color=THEME["blue"],
            hovertemplate="<b>Risk %{x}</b><br>Legit TXNs: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Fraudulent",
            x=labels,
            y=fraud_counts,
            marker_color=THEME["rose"],
            hovertemplate="<b>Risk %{x}</b><br>Fraud TXNs: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="stack",
        yaxis_type="log",
        xaxis_title="Model Fraud Probability Bin",
        yaxis_title="Transaction Count (Log Scale)",
    )

    # Add vertical line for threshold
    thresh_bin_idx = int(threshold * len(labels))
    if 0 <= thresh_bin_idx < len(labels):
        fig.add_vline(
            x=thresh_bin_idx,
            line_dash="dot",
            line_color=THEME["amber"],
            line_width=2.5,
            annotation_text=f"Threshold ({threshold:.2f})",
            annotation_position="top",
        )

    return _apply_theme(fig, "Empirical Model Risk Distribution (Test Fold)", height=350)


# -----------------------------------------------------------------------------
# 7. Interactive Decision Threshold Curves (Visualization 7)
# -----------------------------------------------------------------------------
def plot_threshold_curves(sweep_list: List[Dict[str, Any]], current_threshold: float = 0.5) -> go.Figure:
    """Dynamic Precision, Recall, and F1 trade-off curves across decision thresholds."""
    thresholds = [item["threshold"] for item in sweep_list]
    precisions = [item["precision"] for item in sweep_list]
    recalls = [item["recall"] for item in sweep_list]
    f1s = [item["f1"] for item in sweep_list]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=recalls,
            mode="lines",
            name="Recall (Frauds Caught)",
            line=dict(color=THEME["blue"], width=2.5),
            hovertemplate="<b>Threshold: %{x:.2f}</b><br>Recall: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=precisions,
            mode="lines",
            name="Precision (Alert Accuracy)",
            line=dict(color=THEME["emerald"], width=2.5),
            hovertemplate="<b>Threshold: %{x:.2f}</b><br>Precision: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=f1s,
            mode="lines",
            name="F1-Score (Harmonic Mean)",
            line=dict(color=THEME["amber"], width=2, dash="dash"),
            hovertemplate="<b>Threshold: %{x:.2f}</b><br>F1-Score: %{y:.3f}<extra></extra>",
        )
    )

    # Vertical line at active threshold
    fig.add_vline(
        x=current_threshold,
        line_dash="solid",
        line_color=THEME["rose"],
        line_width=2,
        annotation_text=f"Active: {current_threshold:.2f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        xaxis_title="Decision Threshold (Cutoff)",
        yaxis_title="Metric Value (0.0 to 1.0)",
        xaxis=dict(range=[0.01, 0.99]),
        yaxis=dict(range=[0.0, 1.05]),
    )
    return _apply_theme(fig, "Precision vs. Recall vs. Threshold Playground", height=360)


# -----------------------------------------------------------------------------
# 8. Interactive Confusion Matrix Heatmap (Visualization 8)
# -----------------------------------------------------------------------------
def plot_confusion_matrix(cm_data: Dict[str, Any], threshold: float = 0.5) -> go.Figure:
    """Interactive 2x2 confusion matrix heatmap with counts and percentages."""
    tn = cm_data.get("tn", 56645)
    fp = cm_data.get("fp", 6)
    fn = cm_data.get("fn", 23)
    tp = cm_data.get("tp", 72)
    total = tn + fp + fn + tp

    z = [[tn, fp], [fn, tp]]
    text = [
        [
            f"<b>True Negative (TN)</b><br>{tn:,}<br>({(tn/total)*100:.2f}%)",
            f"<b>False Alarm (FP)</b><br>{fp:,}<br>({(fp/total)*100:.2f}%)",
        ],
        [
            f"<b>Missed Fraud (FN)</b><br>{fn:,}<br>({(fn/total)*100:.2f}%)",
            f"<b>Caught Fraud (TP)</b><br>{tp:,}<br>({(tp/total)*100:.2f}%)",
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Legitimate", "Predicted Fraudulent"],
            y=["Actual Legitimate", "Actual Fraudulent"],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#0f172a"], [0.5, "#1e3a8a"], [1, "#3b82f6"]],
            showscale=False,
            hoverinfo="none",
        )
    )

    fig.update_layout(
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    return _apply_theme(fig, f"Model Confusion Matrix (Threshold = {threshold:.2f})", height=350)


# -----------------------------------------------------------------------------
# 9. ROC Curve (Visualization 9)
# -----------------------------------------------------------------------------
def plot_roc_curve(roc_points: List[Dict[str, Any]], roc_auc: float = 0.9664) -> go.Figure:
    """Interactive ROC Curve with actual FPR and TPR."""
    fprs = [p["fpr"] for p in roc_points]
    tprs = [p["tpr"] for p in roc_points]
    threshs = [p.get("threshold", 0.0) for p in roc_points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fprs,
            y=tprs,
            mode="lines",
            name=f"Tuned Random Forest (AUC = {roc_auc:.4f})",
            line=dict(color=THEME["emerald"], width=2.5),
            customdata=threshs,
            hovertemplate="<b>ROC Point</b><br>FPR: %{x:.4f}<br>TPR: %{y:.4f}<br>Threshold: %{customdata:.3f}<extra></extra>",
        )
    )
    # 45-degree diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier (AUC = 0.50)",
            line=dict(color=THEME["text_muted"], dash="dash", width=1.5),
            hoverinfo="none",
        )
    )

    fig.update_layout(
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall / Sensitivity)",
        xaxis=dict(range=[-0.01, 1.01]),
        yaxis=dict(range=[-0.01, 1.01]),
    )
    return _apply_theme(fig, f"Receiver Operating Characteristic (ROC-AUC = {roc_auc:.4f})", height=360)


# -----------------------------------------------------------------------------
# 10. Precision-Recall Curve (Visualization 10)
# -----------------------------------------------------------------------------
def plot_precision_recall_curve(pr_points: List[Dict[str, Any]], pr_auc: float = 0.8096) -> go.Figure:
    """Interactive PR Curve essential for extreme class imbalance evaluation."""
    recalls = [p["recall"] for p in pr_points]
    precisions = [p["precision"] for p in pr_points]
    threshs = [p.get("threshold", 0.0) for p in pr_points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recalls,
            y=precisions,
            mode="lines",
            name=f"Tuned Random Forest (PR-AUC = {pr_auc:.4f})",
            line=dict(color=THEME["cyan"], width=2.5),
            customdata=threshs,
            hovertemplate="<b>PR Point</b><br>Recall: %{x:.4f}<br>Precision: %{y:.4f}<br>Threshold: %{customdata:.3f}<extra></extra>",
        )
    )
    # Imbalance baseline
    fig.add_hline(
        y=0.001727,
        line_dash="dot",
        line_color=THEME["text_muted"],
        annotation_text="No-Skill Baseline (0.17%)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        xaxis_title="Recall (Frauds Detected)",
        yaxis_title="Precision (Alert Purity)",
        xaxis=dict(range=[-0.01, 1.01]),
        yaxis=dict(range=[-0.01, 1.05]),
    )
    return _apply_theme(fig, f"Precision-Recall Curve (PR-AUC = {pr_auc:.4f})", height=360)


# -----------------------------------------------------------------------------
# 11. Feature Importance (Visualization 11)
# -----------------------------------------------------------------------------
def plot_feature_importances(importances: List[Dict[str, Any]], top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of top Random Forest Gini feature importances."""
    top_items = importances[:top_n][::-1]  # reverse for top at the top
    features = [item["feature"] for item in top_items]
    scores = [item["importance"] for item in top_items]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=features,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale=[[0, "#1e3a8a"], [1, "#06b6d4"]],
            ),
            hovertemplate="<b>%{y}</b><br>Gini Importance: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Mean Decrease in Impurity (Gini Importance)",
        yaxis_title="Feature Component",
    )
    return _apply_theme(fig, f"Top {top_n} Fraud Driving Features (Model Global)", height=380)


# -----------------------------------------------------------------------------
# 12. SHAP Force Contribution Bar (Local Explainability)
# -----------------------------------------------------------------------------
def plot_shap_force_bars(top_features: List[Dict[str, Any]]) -> go.Figure:
    """Horizontal diverged bar chart showing directional attribution forces on a single transaction."""
    sorted_feats = sorted(top_features[:8], key=lambda x: abs(x["contribution"]), reverse=False)
    features = [f["feature"] for f in sorted_feats]
    contribs = [f["contribution"] for f in sorted_feats]
    raw_vals = [f["raw_value"] for f in sorted_feats]
    colors = [THEME["rose"] if c > 0 else THEME["emerald"] for c in contribs]

    fig = go.Figure(
        go.Bar(
            x=contribs,
            y=features,
            orientation="h",
            marker_color=colors,
            customdata=raw_vals,
            hovertemplate="<b>%{y}</b> (Val: %{customdata})<br>Contribution: %{x:+.4f}<extra></extra>",
        )
    )

    fig.add_vline(x=0, line_width=1.5, line_color=THEME["text_muted"])

    fig.update_layout(
        xaxis_title="SHAP Attribution Force (← Legitimate | Fraud →)",
        yaxis_title="Component",
    )
    return _apply_theme(fig, "Local Feature Attribution Forces", height=320)


# -----------------------------------------------------------------------------
# 13. Calibrated Segmented Risk Gauge
# -----------------------------------------------------------------------------
def plot_risk_gauge(risk_score: int, threshold: float = 0.5) -> go.Figure:
    """Plotly Indicator Gauge showing deterministic risk score and color-coded risk bands."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number=dict(suffix=" / 100", font=dict(family=THEME["mono_family"], size=28, color=THEME["text_color"])),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=THEME["text_muted"], tickfont=dict(size=10)),
                bar=dict(color=THEME["text_color"], thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=1,
                bordercolor="rgba(255,255,255,0.1)",
                steps=[
                    dict(range=[0, 30], color="rgba(16, 185, 129, 0.35)"),   # Low (Emerald)
                    dict(range=[30, 60], color="rgba(245, 158, 11, 0.35)"),  # Medium (Amber)
                    dict(range=[60, 80], color="rgba(249, 115, 22, 0.45)"),  # High (Orange)
                    dict(range=[80, 100], color="rgba(239, 68, 68, 0.55)"),  # Critical (Rose)
                ],
                threshold=dict(
                    line=dict(color="#ffffff", width=3),
                    thickness=0.8,
                    value=int(threshold * 100),
                ),
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor=THEME["paper_color"],
        font=dict(family=THEME["font_family"]),
        height=220,
        margin=dict(l=25, r=25, t=25, b=20),
    )
    return fig


# -----------------------------------------------------------------------------
# 14. Correlation Heatmap
# -----------------------------------------------------------------------------
def plot_correlation_heatmap(heatmap_data: Dict[str, Any]) -> go.Figure:
    """Interactive heatmap showing pairwise correlations between top fraud predictors."""
    features = heatmap_data.get("features", [])
    z = heatmap_data.get("z", [])

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=features,
            y=features,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
    )
    return _apply_theme(fig, "Top Feature Correlation Matrix", height=480)


# -----------------------------------------------------------------------------
# 15. 3D Feature Space Explorer (Visualization 66 & 67)
# -----------------------------------------------------------------------------
def plot_3d_feature_space(
    sample_data: Dict[str, Any],
    x_feature: str = "V14",
    y_feature: str = "V10",
    z_feature: str = "V12",
    color_by: str = "Class",
) -> go.Figure:
    """Interactive 3D scatter plot of 1,500 real dataset transactions (492 frauds + 1008 legits)."""
    xs = sample_data.get(x_feature, [])
    ys = sample_data.get(y_feature, [])
    zs = sample_data.get(z_feature, [])
    classes = sample_data.get("Class", [])
    amounts = sample_data.get("Amount", [])
    indices = sample_data.get("index", [])

    if color_by == "Class":
        colors = [THEME["rose"] if c == 1 else THEME["blue"] for c in classes]
        names = ["Fraud" if c == 1 else "Legitimate" for c in classes]
    else:
        colors = amounts
        names = [f"${a:.2f}" for a in amounts]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=4,
                    color=colors,
                    opacity=0.8,
                    line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
                ),
                customdata=np.stack((indices, amounts, classes), axis=-1),
                hovertemplate="<b>Index #%{customdata[0]}</b><br>Class: %{customdata[2]}<br>Amount: $%{customdata[1]:.2f}<br>"
                + f"{x_feature}: %{{x:.3f}}<br>{y_feature}: %{{y:.3f}}<br>{z_feature}: %{{z:.3f}}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title=x_feature, backgroundcolor="rgba(15,23,42,0.4)", gridcolor=THEME["grid_color"]),
            yaxis=dict(title=y_feature, backgroundcolor="rgba(15,23,42,0.4)", gridcolor=THEME["grid_color"]),
            zaxis=dict(title=z_feature, backgroundcolor="rgba(15,23,42,0.4)", gridcolor=THEME["grid_color"]),
        ),
        paper_bgcolor=THEME["paper_color"],
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# -----------------------------------------------------------------------------
# 16. What-If Comparison Bar
# -----------------------------------------------------------------------------
def plot_what_if_comparison(orig_prob: float, mod_prob: float, threshold: float = 0.5) -> go.Figure:
    """Side-by-side comparison of original vs modified transaction probability."""
    delta = mod_prob - orig_prob
    categories = ["Original Input", "Modified Input"]
    probs = [orig_prob, mod_prob]
    colors = [
        THEME["rose"] if orig_prob >= threshold else THEME["blue"],
        THEME["rose"] if mod_prob >= threshold else THEME["blue"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=probs,
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Fraud Probability: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=THEME["amber"],
        annotation_text=f"Cutoff Threshold ({threshold:.2f})",
        annotation_position="top right",
    )

    fig.update_layout(
        yaxis_title="Model Fraud Probability",
        yaxis=dict(range=[0, 1.05]),
    )
    return _apply_theme(fig, f"Sensitivity Outcome (Δ = {delta:+.4f})", height=320)


# -----------------------------------------------------------------------------
# 17. Simulation Risk Stream
# -----------------------------------------------------------------------------
def plot_simulation_stream(records: List[Dict[str, Any]], threshold: float = 0.5) -> go.Figure:
    """Real-time scatter/line plot of synthetic simulation sequence through the model."""
    if not records:
        fig = go.Figure()
        return _apply_theme(fig, "No Simulation Records Yet", height=300)

    seq = list(range(1, len(records) + 1))
    probs = [r.get("fraud_probability", 0.0) for r in records]
    amounts = [r.get("amount", 0.0) for r in records]
    flags = [r.get("is_flagged", False) for r in records]
    colors = [THEME["rose"] if f else THEME["emerald"] for f in flags]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=seq,
            y=probs,
            mode="lines+markers",
            name="Stream Probabilities",
            line=dict(color="rgba(148, 163, 184, 0.4)", width=1.5),
            marker=dict(size=7, color=colors),
            customdata=amounts,
            hovertemplate="<b>TXN #%{x}</b><br>Fraud Prob: %{y:.3f}<br>Amount: $%{customdata:.2f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=THEME["amber"],
        annotation_text=f"Decision Threshold ({threshold:.2f})",
    )

    fig.update_layout(
        xaxis_title="Transaction Sequence Number",
        yaxis_title="Model Fraud Probability",
        yaxis=dict(range=[0, 1.05]),
    )
    return _apply_theme(fig, "Live Simulation Risk Stream", height=340)
