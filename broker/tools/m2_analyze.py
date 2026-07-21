"""M2: measure real echo-residual NCC between the reference clip and a capture.

    uv run --with numpy tools/m2_analyze.py m2_captures/ref.wav m2_captures/capture_1.pcm

Answers the M2 decision question for the planned broker echo gate
(_MicInputGate NCC gate): how correlated is the XMOS post-AEC mic residual
(ch1 @ 24x gain) with the audio the speaker actually played?

  - windows where max |NCC| >= 0.6  -> the gate can see the echo; ship M1.
  - 0.3-0.6                         -> gray zone; redesign before building
                                       (volume cap / aec_corr DFU / sw AEC).
  - < 0.3 with mic RMS ~ room noise -> AEC already buries the echo; the gate
                                       may not even be needed at this volume.

Method: coarse global alignment via envelope cross-correlation (the capture
starts at an arbitrary time before playback), then per-window (320ms, hop
160ms) normalized cross-correlation at the global lag +/- 150ms search
range, scored only over windows where the reference is active. Baseline is
the same statistic over reference-silent windows, it shows what NCC noise
looks like for this capture.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

RATE = 24000
WIN = int(0.320 * RATE)
HOP = int(0.160 * RATE)
SEARCH = int(0.150 * RATE)  # per-window lag search around global alignment


def load_ref(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1, "want 24k mono"
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float64)


def load_pcm(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.int16).astype(np.float64)


def global_lag(ref: np.ndarray, mic: np.ndarray) -> int:
    """Offset of ref within mic, by envelope cross-correlation at 100 Hz."""
    dec = RATE // 100
    env = lambda x: np.abs(x[: len(x) // dec * dec].reshape(-1, dec)).mean(axis=1)
    er, em = env(ref), env(mic)
    er -= er.mean()
    em -= em.mean()
    corr = np.correlate(em, er, mode="valid")
    return int(np.argmax(corr)) * dec


def max_ncc(r: np.ndarray, m: np.ndarray) -> float:
    """Max |NCC| of window r against every sample offset of segment m.

    Sample-accurate: NCC collapses within a few samples of the true lag, so
    any strided search misses the peak. Vectorized, numerator via one
    cross-correlation, per-offset mean/energy via cumulative sums.
    """
    r0 = r - r.mean()
    er = float((r0 * r0).sum())
    if er == 0:
        return 0.0
    L = len(r)
    num = np.correlate(m, r0, "valid")
    c1 = np.concatenate(([0.0], np.cumsum(m)))
    c2 = np.concatenate(([0.0], np.cumsum(m * m)))
    msum = c1[L:] - c1[:-L]
    var = (c2[L:] - c2[:-L]) - msum * msum / L
    den = np.sqrt(er * np.clip(var, 1e-12, None))
    return float(np.max(np.abs(num / den)))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ref = load_ref(Path(sys.argv[1]))
    mic = load_pcm(Path(sys.argv[2]))
    print(f"ref {len(ref)/RATE:.1f}s   capture {len(mic)/RATE:.1f}s")
    if len(mic) < len(ref):
        # np.correlate silently swaps operands when the first array is the
        # shorter one, which would produce a garbage global lag and a false
        # "residual uncorrelated" verdict. The capture must cover the clip.
        raise SystemExit(
            "capture shorter than the reference. Bad take, retake it "
            f"(capture {len(mic)/RATE:.1f}s < ref {len(ref)/RATE:.1f}s)"
        )

    lag = global_lag(ref, mic)
    print(f"global alignment: ref starts at {lag/RATE:.2f}s into the capture")

    ref_gate = np.abs(ref).mean() * 0.3  # "reference active" per-window bar
    active, silent = [], []
    for start in range(0, len(ref) - WIN, HOP):
        r = ref[start : start + WIN]
        lo = max(0, lag + start - SEARCH)
        hi = min(len(mic), lag + start + WIN + SEARCH)
        m = mic[lo:hi]
        if len(m) < WIN:
            continue
        best = max_ncc(r, m)
        rms = np.sqrt((mic[lag + start : lag + start + WIN] ** 2).mean()) if lag + start + WIN <= len(mic) else 0.0
        (active if np.abs(r).mean() > ref_gate else silent).append((best, rms))

    if not active:
        raise SystemExit("no reference-active windows found, alignment failed?")

    a_ncc = np.array([x[0] for x in active])
    a_rms = np.array([x[1] for x in active])
    print(f"\nreference-active windows: {len(active)}")
    print(f"  echo-residual NCC : p50={np.median(a_ncc):.3f}  p90={np.percentile(a_ncc, 90):.3f}  max={a_ncc.max():.3f}")
    print(f"  mic RMS           : p50={np.median(a_rms):.0f}  p90={np.percentile(a_rms, 90):.0f}")
    if silent:
        s_ncc = np.array([x[0] for x in silent])
        s_rms = np.array([x[1] for x in silent])
        print(f"reference-silent windows: {len(silent)}  (NCC noise floor)")
        print(f"  baseline NCC      : p50={np.median(s_ncc):.3f}  p90={np.percentile(s_ncc, 90):.3f}")
        print(f"  mic RMS           : p50={np.median(s_rms):.0f}")

    p50 = float(np.median(a_ncc))
    if p50 >= 0.6:
        print("\nVERDICT: NCC gate viable, echo residual clearly correlated. Build M1.")
    elif p50 >= 0.3:
        print("\nVERDICT: gray zone (0.3-0.6): per the plan, STOP and redesign "
              "(volume cap -> aec_corr_factor DFU -> software AEC).")
    else:
        print("\nVERDICT: residual uncorrelated. If mic RMS during playback is "
              "near the silent baseline, XMOS AEC already buries the echo at "
              "this volume, measure at max volume before concluding.")


if __name__ == "__main__":
    main()
