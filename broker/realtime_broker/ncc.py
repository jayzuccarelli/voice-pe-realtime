"""Windowed normalized cross-correlation for the echo gate.

Same math as tools/m2_analyze.py (verified there on synthetic fixtures),
but FFT-based so a 320ms window against a ~1.5s reference segment costs
single-digit milliseconds — cheap enough to run inline in the pipeline
every 320ms while the bot speaks.
"""

from __future__ import annotations

import numpy as np


def max_ncc(window: np.ndarray, segment: np.ndarray) -> float:
    """Max |NCC| of `window` at every sample offset within `segment`.

    Sample-accurate (NCC collapses within a few samples of the true lag).
    Numerator via FFT cross-correlation, per-offset mean/energy via
    cumulative sums. Requires len(segment) >= len(window).
    """
    w = window - window.mean()
    ew = float((w * w).sum())
    if ew == 0.0:
        return 0.0
    L, n = len(w), len(segment)

    size = 1 << (n + L - 1).bit_length()
    conv = np.fft.irfft(np.fft.rfft(segment, size) * np.fft.rfft(w[::-1], size), size)
    num = conv[L - 1 : n]  # valid cross-correlation: num[off] = sum(w * seg[off:off+L])

    c1 = np.concatenate(([0.0], np.cumsum(segment)))
    c2 = np.concatenate(([0.0], np.cumsum(segment * segment)))
    ssum = c1[L:] - c1[:-L]
    var = (c2[L:] - c2[:-L]) - ssum * ssum / L
    den = np.sqrt(ew * np.clip(var, 1e-12, None))
    return float(np.max(np.abs(num / den)))
