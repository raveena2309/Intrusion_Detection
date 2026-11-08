"""
dashboard/components/charts.py
================================
Reusable Plotly chart functions for the NIDS dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional


DARK_BG = "#0e1117"
CARD_BG = "#1e2130"
ACCENT = "#00d4aa"
DANGER = "#ff4b4b"
WARNING = "#ffa500"
INFO = "#4b9fff"
TEXT = "#fafafa"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor="#2d3250"),
        yaxis=dict(gridcolor="#2d3250"),
    )
)

CLASS_COLORS = {
    "Normal": "#2ecc71",
    "Generic": "#e74c3c",
    "Exploits": "#e67e22",
    "Fuzzers": "#f39c12",
    "DoS": "#c0392b",
    "Reconnaissance": "#9b59b6",
    "Analysis": "#1abc9c",
    "Backdoor": "#e91e63",
    "Shellcode": "#ff5722",
    "Worms": "#795548",
    "unknown_suspicious": "#ff4b4b",
    "uncertain_review": "#ffa500",
    "normal": "#2ecc71",
}


def kpi_card(label: str, value, delta=None, color=None):
    """Return a dict for st.metric (used as KPI card)."""
    return {"label": label, "value": value, "delta": delta}


def attack_distribution_chart(counts: dict, title: str = "Attack Distribution") -> go.Figure:
    """Pie/donut chart of attack category distribution."""
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [CLASS_COLORS.get(l, INFO) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color=DARK_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=TEXT),
        showlegend=True,
        legend=dict(bgcolor=CARD_BG),
        margin=dict(t=50, b=20, l=20, r=20),
        height=380,
    )
    return fig


def class_bar_chart(counts: dict, title: str = "Class Distribution") -> go.Figure:
    """Horizontal bar chart of class counts."""
    df = pd.DataFrame(list(counts.items()), columns=["Class", "Count"])
    df = df.sort_values("Count", ascending=True)
    colors = [CLASS_COLORS.get(c, INFO) for c in df["Class"]]

    fig = go.Figure(go.Bar(
        y=df["Class"], x=df["Count"],
        orientation="h",
        marker_color=colors,
        text=df["Count"], textposition="outside",
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor="#2d3250"),
        yaxis=dict(gridcolor="#2d3250"),
        margin=dict(t=50, b=20, l=20, r=60),
        height=400,
    )
    return fig


def confusion_matrix_chart(cm: np.ndarray, label_names: List[str],
                            title: str = "Confusion Matrix", normalize: bool = True) -> go.Figure:
    """Heatmap confusion matrix."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm.astype(float) / row_sums
        fmt_text = [[f"{cm_plot[i,j]:.2f}<br>({cm[i,j]})" for j in range(len(label_names))]
                    for i in range(len(label_names))]
    else:
        cm_plot = cm.astype(float)
        fmt_text = [[str(cm[i,j]) for j in range(len(label_names))]
                    for i in range(len(label_names))]

    fig = go.Figure(go.Heatmap(
        z=cm_plot, x=label_names, y=label_names,
        text=fmt_text, texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        textfont=dict(size=9),
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(title="Predicted", tickangle=-45),
        yaxis=dict(title="Actual"),
        height=500,
        margin=dict(t=60, b=100, l=100, r=20),
    )
    return fig


def anomaly_score_histogram(scores: np.ndarray, threshold: float,
                             labels: Optional[np.ndarray] = None,
                             normal_class_idx: int = 6) -> go.Figure:
    """Histogram of anomaly scores with threshold line."""
    fig = go.Figure()

    if labels is not None:
        normal_mask = labels == normal_class_idx
        fig.add_trace(go.Histogram(
            x=scores[normal_mask], name="Normal",
            marker_color=CLASS_COLORS["Normal"], opacity=0.7, nbinsx=60,
        ))
        fig.add_trace(go.Histogram(
            x=scores[~normal_mask], name="Attack",
            marker_color=DANGER, opacity=0.7, nbinsx=60,
        ))
    else:
        fig.add_trace(go.Histogram(x=scores, name="Anomaly Scores",
                                    marker_color=INFO, opacity=0.8, nbinsx=60))

    fig.add_vline(x=threshold, line_dash="dash", line_color=WARNING,
                  annotation_text=f"Threshold={threshold:.4f}",
                  annotation_position="top right",
                  annotation_font_color=WARNING)

    fig.update_layout(
        title="Anomaly Score Distribution",
        barmode="overlay",
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis_title="Anomaly Score (higher = more anomalous)",
        yaxis_title="Count",
        legend=dict(bgcolor=CARD_BG),
        height=380,
    )
    return fig


def shap_importance_chart(feature_names: List[str], mean_abs_shap: np.ndarray,
                           top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of mean |SHAP| feature importance."""
    order = np.argsort(mean_abs_shap)[-top_n:]
    fig = go.Figure(go.Bar(
        y=[feature_names[i] for i in order],
        x=mean_abs_shap[order],
        orientation="h",
        marker_color=ACCENT,
    ))
    fig.update_layout(
        title=f"Top {top_n} Features by Mean |SHAP|",
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(title="Mean |SHAP|", gridcolor="#2d3250"),
        yaxis=dict(gridcolor="#2d3250"),
        height=450,
        margin=dict(t=50, b=30, l=140, r=20),
    )
    return fig


def local_shap_chart(feature_names: List[str], shap_values: np.ndarray,
                     feature_values: np.ndarray, top_n: int = 10) -> go.Figure:
    """Waterfall-style bar chart for local SHAP explanation."""
    order = np.argsort(np.abs(shap_values))[-top_n:]
    names = [f"{feature_names[i]}={feature_values[i]:.3g}" for i in order]
    vals = shap_values[order]
    colors = [ACCENT if v > 0 else DANGER for v in vals]

    fig = go.Figure(go.Bar(
        y=names, x=vals, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title=f"Local Explanation - Top {top_n} Feature Contributions",
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(title="SHAP Value", gridcolor="#2d3250"),
        height=400,
        margin=dict(t=50, b=30, l=200, r=60),
    )
    return fig


def metrics_table_chart(per_class: dict) -> go.Figure:
    """Table showing per-class precision, recall, F1."""
    classes = list(per_class.keys())
    prec = [round(per_class[c].get("precision", 0), 4) for c in classes]
    rec  = [round(per_class[c].get("recall", 0), 4) for c in classes]
    f1   = [round(per_class[c].get("f1", 0), 4) for c in classes]

    fig = go.Figure(go.Table(
        header=dict(
            values=["Class", "Precision", "Recall", "F1"],
            fill_color=CARD_BG,
            font=dict(color=ACCENT, size=12),
            line_color="#2d3250",
        ),
        cells=dict(
            values=[classes, prec, rec, f1],
            fill_color=DARK_BG,
            font=dict(color=TEXT, size=11),
            line_color="#2d3250",
        ),
    ))
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        margin=dict(t=10, b=10, l=10, r=10),
        height=max(300, 35 * len(classes) + 50),
    )
    return fig
