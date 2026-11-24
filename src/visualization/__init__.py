"""Interactive Plotly and Matplotlib scientific visualizations for EEG."""
from .plots import (
    plot_multichannel_eeg,
    plot_psd_spectrum,
    plot_seizure_score_timeline,
    plot_confusion_matrix_figure,
    plot_roc_curve,
)

__all__ = [
    "plot_multichannel_eeg",
    "plot_psd_spectrum",
    "plot_seizure_score_timeline",
    "plot_confusion_matrix_figure",
    "plot_roc_curve",
]
