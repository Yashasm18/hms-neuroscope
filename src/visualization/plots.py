"""
Interactive Plotly visualizations for EEG traces, spectra, timelines, and ROC curves.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal

from config import EEG_BANDS, DEFAULT_SAMPLING_RATE


def plot_multichannel_eeg(
    data: np.ndarray,
    channel_names: List[str],
    sfreq: float = DEFAULT_SAMPLING_RATE,
    t_start_sec: float = 0.0,
    duration_sec: float = 10.0,
    highlight_intervals: Optional[List[Tuple[float, float, str]]] = None,
    channel_spacing_uV: float = 120.0,
    title: str = "Multichannel EEG Traces",
    data_offset_sec: float = 0.0,
) -> go.Figure:
    """
    Render stacked multichannel EEG waveforms in standard neurophysiology review format.
    highlight_intervals: list of (start_sec, end_sec, label) e.g. for ground-truth or predicted seizures.
    """
    # data_offset_sec: absolute time of data[:, 0], so a slice of a longer
    # recording can be plotted on the recording's own time axis.
    n_channels, total_samples = data.shape
    start_idx = int(round((t_start_sec - data_offset_sec) * sfreq))
    end_idx = min(total_samples, start_idx + int(round(duration_sec * sfreq)))

    if start_idx >= total_samples:
        start_idx = 0
        end_idx = min(total_samples, int(duration_sec * sfreq))

    sub_data = data[:, start_idx:end_idx]
    actual_samples = sub_data.shape[1]
    t = np.linspace(t_start_sec, t_start_sec + (actual_samples / sfreq), actual_samples, endpoint=False)

    fig = go.Figure()

    # Stack channels vertically with offsets
    # Top channel gets largest offset
    y_tickvals = []
    y_ticktext = []

    for i in range(n_channels):
        ch_idx = n_channels - 1 - i  # Invert so first channel is on top
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch {ch_idx+1}"
        offset = i * channel_spacing_uV
        y_tickvals.append(offset)
        y_ticktext.append(ch_name)

        trace_y = sub_data[ch_idx] + offset

        fig.add_trace(
            go.Scatter(
                x=t,
                y=trace_y,
                mode="lines",
                name=ch_name,
                line=dict(width=1.1, color="#7DD3FC"),  # Ice blue
                hoverinfo="x+y+name",
            )
        )

    # Highlight intervals (e.g. Seizure)
    if highlight_intervals:
        for int_start, int_end, label in highlight_intervals:
            # Check overlap with current view window
            ov_start = max(t_start_sec, int_start)
            ov_end = min(t_start_sec + duration_sec, int_end)
            if ov_start < ov_end:
                fig.add_vrect(
                    x0=ov_start,
                    x1=ov_end,
                    fillcolor="rgba(239, 68, 68, 0.22)",  # Translucent crimson
                    layer="below",
                    line_width=1,
                    line_color="rgba(239, 68, 68, 0.8)",
                    annotation_text=label,
                    annotation_position="top left",
                    annotation_font=dict(color="#F87171", size=11),
                )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#F1F5F9", size=16)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)", font=dict(family="Geist, -apple-system, sans-serif", color="#C3C9D6"),
        margin=dict(l=70, r=30, t=50, b=40),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="#475569",
            color="#94A3B8",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            gridcolor="rgba(255,255,255,0.06)",
            color="#94A3B8",
            title="Channels (Stacked μV)",
        ),
        showlegend=False,
        height=max(450, n_channels * 32),
    )
    return fig


def plot_psd_spectrum(
    sig: np.ndarray,
    sfreq: float = DEFAULT_SAMPLING_RATE,
    max_freq: float = 45.0,
    title: str = "Power Spectral Density (Welch)",
) -> go.Figure:
    """Plot Welch PSD with overlaid frequency band demarcations."""
    freqs, psd = signal.welch(sig, fs=sfreq, nperseg=min(len(sig), int(sfreq * 2.0)))
    mask = freqs <= max_freq
    f_sub = freqs[mask]
    p_sub = psd[mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=f_sub,
            y=10 * np.log10(np.maximum(p_sub, 1e-12)),
            mode="lines",
            name="PSD (dB/Hz)",
            line=dict(color="#10B981", width=2.0),
        )
    )

    # Shaded band regions
    band_colors = {
        "delta": "rgba(99, 102, 241, 0.15)",
        "theta": "rgba(14, 165, 233, 0.15)",
        "alpha": "rgba(16, 185, 129, 0.15)",
        "beta": "rgba(245, 158, 11, 0.15)",
        "gamma": "rgba(239, 68, 68, 0.15)",
    }

    for band, (f_l, f_h) in EEG_BANDS.items():
        if f_l <= max_freq:
            f_end = min(max_freq, f_h)
            fig.add_vrect(
                x0=f_l,
                x1=f_end,
                fillcolor=band_colors.get(band, "rgba(255,255,255,0.05)"),
                layer="below",
                line_width=0,
                annotation_text=band.capitalize(),
                annotation_position="top",
                annotation_font=dict(color="#94A3B8", size=10),
            )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#F1F5F9", size=14)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)", font=dict(family="Geist, -apple-system, sans-serif", color="#C3C9D6"),
        xaxis=dict(title="Frequency (Hz)", gridcolor="rgba(255,255,255,0.06)", color="#94A3B8"),
        yaxis=dict(title="Power (dB / Hz)", gridcolor="rgba(255,255,255,0.06)", color="#94A3B8"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=320,
    )
    return fig


def plot_seizure_score_timeline(
    timeline_df: pd.DataFrame,
    threshold: float = 0.5,
    reference_intervals: Optional[List[Tuple[float, float]]] = None,
    title: str = "Seizure Risk Score Timeline Across Recording",
) -> go.Figure:
    """Plot window risk score progression with decision threshold and detected events."""
    fig = go.Figure()

    t_mid = (timeline_df["start_sec"] + timeline_df["end_sec"]) / 2.0
    scores = timeline_df["seizure_score"]

    # Shaded reference seizures if provided
    if reference_intervals:
        for r_start, r_end in reference_intervals:
            fig.add_vrect(
                x0=r_start,
                x1=r_end,
                fillcolor="rgba(16, 185, 129, 0.2)",
                line=dict(color="#10B981", width=1, dash="dash"),
                annotation_text="Ground Truth Seizure",
                annotation_position="top left",
                annotation_font=dict(color="#34D399", size=10),
            )

    # Risk score line
    fig.add_trace(
        go.Scatter(
            x=t_mid,
            y=scores,
            mode="lines+markers",
            name="Seizure Score",
            line=dict(color="#818CF8", width=1.5),
            marker=dict(
                size=4,
                color=np.where(scores >= threshold, "#EF4444", "#38BDF8"),
            ),
            hovertext=[f"Window [{s:.1f}s - {e:.1f}s]<br>Score: {sc:.3f}" for s, e, sc in zip(timeline_df["start_sec"], timeline_df["end_sec"], scores)],
            hoverinfo="text",
        )
    )

    # Threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color="#F87171", width=2, dash="dash"),
        annotation_text=f"Decision Threshold ({threshold:.2f})",
        annotation_position="bottom right",
        annotation_font=dict(color="#F87171", size=11),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#F1F5F9", size=14)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)", font=dict(family="Geist, -apple-system, sans-serif", color="#C3C9D6"),
        xaxis=dict(title="Recording Time (seconds)", gridcolor="rgba(255,255,255,0.06)", color="#94A3B8"),
        yaxis=dict(title="Seizure Score [0.0 - 1.0]", range=[-0.05, 1.05], gridcolor="rgba(255,255,255,0.06)", color="#94A3B8"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=320,
    )
    return fig


def plot_confusion_matrix_figure(
    tn: int, fp: int, fn: int, tp: int,
    title: str = "Window Confusion Matrix",
) -> go.Figure:
    """Render 2x2 confusion matrix heatmap."""
    matrix = [[tn, fp], [fn, tp]]
    x_labels = ["Pred Non-Seizure (0)", "Pred Seizure (1)"]
    y_labels = ["True Non-Seizure (0)", "True Seizure (1)"]

    text_vals = [
        [f"TN: {tn:,}", f"FP: {fp:,}"],
        [f"FN: {fn:,}", f"TP: {tp:,}"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, "#0F172A"], [0.5, "#1E3A8A"], [1.0, "#3B82F6"]],
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#FFFFFF"),
            showscale=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#F1F5F9", size=14)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)", font=dict(family="Geist, -apple-system, sans-serif", color="#C3C9D6"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
    )
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: Optional[float] = None,
    title: str = "Receiver Operating Characteristic (ROC)",
) -> go.Figure:
    """Plot ROC curve with AUC metric and chance diagonal."""
    fig = go.Figure()

    auc_label = f"ROC (AUC = {auc_score:.3f})" if auc_score is not None else "ROC (AUC: N/A)"
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=auc_label,
            line=dict(color="#38BDF8", width=2.5),
        )
    )

    # Random chance diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Chance (AUC = 0.50)",
            line=dict(color="#64748B", width=1.5, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#F1F5F9", size=14)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)", font=dict(family="Geist, -apple-system, sans-serif", color="#C3C9D6"),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", gridcolor="rgba(255,255,255,0.06)", color="#94A3B8"),
        yaxis=dict(title="True Positive Rate (Sensitivity)", gridcolor="rgba(255,255,255,0.06)", color="#94A3B8"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=320,
    )
    return fig
