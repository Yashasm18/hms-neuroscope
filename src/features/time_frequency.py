"""
Mathematical implementations of time-domain statistics and spectral band powers for EEG signals.
"""

from __future__ import annotations
import numpy as np
from scipy import signal, stats
from typing import Dict

from config import DEFAULT_SAMPLING_RATE, EEG_BANDS


def compute_time_domain_features(sig: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical and morphological features for a 1D EEG signal segment:
    mean, std, variance, skewness, kurtosis, peak-to-peak, RMS energy,
    line length, zero-crossing rate, and Hjorth parameters.
    """
    n = len(sig)
    if n < 4:
        return {
            "mean": 0.0,
            "std": 0.0,
            "variance": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "ptp": 0.0,
            "rms": 0.0,
            "line_length": 0.0,
            "zero_crossings": 0.0,
            "hjorth_activity": 0.0,
            "hjorth_mobility": 0.0,
            "hjorth_complexity": 0.0,
        }

    mean_val = float(np.mean(sig))
    var_val = float(np.var(sig))
    std_val = float(np.sqrt(var_val)) if var_val > 0 else 0.0
    ptp_val = float(np.ptp(sig))
    rms_val = float(np.sqrt(np.mean(sig**2)))

    # Skewness and kurtosis
    skew_val = float(stats.skew(sig)) if std_val > 1e-6 else 0.0
    kurt_val = float(stats.kurtosis(sig)) if std_val > 1e-6 else 0.0

    # Line length (sum of absolute first differences normalized by N)
    diff1 = np.diff(sig)
    line_length = float(np.sum(np.abs(diff1)) / n)

    # Zero-crossing rate
    zero_crossings = float(np.sum(np.diff(np.sign(sig - mean_val) != 0)) / n)

    # Hjorth parameters
    # Activity = variance
    hjorth_activity = var_val
    # Mobility = sqrt(var(diff1) / var(sig))
    var_diff1 = float(np.var(diff1))
    if var_val > 1e-8:
        hjorth_mobility = float(np.sqrt(var_diff1 / var_val))
    else:
        hjorth_mobility = 0.0

    # Complexity = Mobility(diff1) / Mobility(sig)
    diff2 = np.diff(diff1)
    var_diff2 = float(np.var(diff2))
    if var_diff1 > 1e-8 and hjorth_mobility > 1e-8:
        mobility_diff1 = float(np.sqrt(var_diff2 / var_diff1))
        hjorth_complexity = float(mobility_diff1 / hjorth_mobility)
    else:
        hjorth_complexity = 0.0

    # Protect against any residual non-finite values
    feats = {
        "mean": mean_val,
        "std": std_val,
        "variance": var_val,
        "skewness": 0.0 if np.isnan(skew_val) else skew_val,
        "kurtosis": 0.0 if np.isnan(kurt_val) else kurt_val,
        "ptp": ptp_val,
        "rms": rms_val,
        "line_length": line_length,
        "zero_crossings": zero_crossings,
        "hjorth_activity": hjorth_activity,
        "hjorth_mobility": 0.0 if np.isnan(hjorth_mobility) else hjorth_mobility,
        "hjorth_complexity": 0.0 if np.isnan(hjorth_complexity) else hjorth_complexity,
    }
    return feats


def compute_frequency_domain_features(
    sig: np.ndarray,
    sfreq: float = DEFAULT_SAMPLING_RATE,
    bands: Optional[Dict[str, tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    Compute frequency-domain features using Welch's power spectral density (PSD):
    band powers (delta, theta, alpha, beta, gamma), relative band powers,
    spectral entropy, and spectral edge frequency (SEF95).
    """
    if bands is None:
        bands = EEG_BANDS

    n = len(sig)
    nperseg = min(n, int(sfreq * 2.0))  # 2-second sub-window for Welch
    if nperseg < 8:
        nperseg = n

    freqs, psd = signal.welch(
        sig,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total_power = float(np.sum(psd) * freq_res)
    total_power = max(total_power, 1e-12)

    feats = {}
    for band_name, (f_low, f_high) in bands.items():
        band_mask = (freqs >= f_low) & (freqs <= f_high)
        if np.any(band_mask):
            band_pow = float(np.sum(psd[band_mask]) * freq_res)
        else:
            band_pow = 0.0
        
        feats[f"abs_power_{band_name}"] = band_pow
        feats[f"rel_power_{band_name}"] = band_pow / total_power

    feats["total_spectral_power"] = total_power

    # Peak frequency
    peak_idx = int(np.argmax(psd))
    feats["peak_frequency"] = float(freqs[peak_idx])

    # Spectral entropy: normalized Shannon entropy of PSD distribution
    psd_norm = psd / np.sum(psd)
    # Filter zeros to avoid log(0)
    psd_norm_nz = psd_norm[psd_norm > 1e-12]
    if len(psd_norm_nz) > 1:
        spectral_entropy = float(-np.sum(psd_norm_nz * np.log2(psd_norm_nz)) / np.log2(len(psd_norm)))
    else:
        spectral_entropy = 0.0
    feats["spectral_entropy"] = max(0.0, min(1.0, spectral_entropy))

    # Spectral edge frequency (SEF95): frequency below which 95% of power is concentrated
    cumsum_power = np.cumsum(psd) * freq_res
    sef95_threshold = 0.95 * total_power
    sef95_idx = np.searchsorted(cumsum_power, sef95_threshold)
    sef95_freq = float(freqs[min(sef95_idx, len(freqs) - 1)])
    feats["sef95"] = sef95_freq

    return feats


def extract_single_channel_features(
    sig: np.ndarray,
    sfreq: float = DEFAULT_SAMPLING_RATE,
) -> Dict[str, float]:
    """Combine time-domain and frequency-domain features for a single channel."""
    t_feats = compute_time_domain_features(sig)
    f_feats = compute_frequency_domain_features(sig, sfreq=sfreq)
    combined = {**t_feats, **f_feats}
    return combined
