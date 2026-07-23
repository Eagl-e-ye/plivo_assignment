"""Audio utilities for the EOT assignment.

These are UTILITIES, not features. Turning them into informative features
(slopes, ratios, statistics over time) is your job.

Causality reminder: for a pause at `pause_start`, you may only touch
audio[0 : pause_start]. Note that `pause_end` is FUTURE information for a
hold pause — using it (e.g., pause duration) in features is a violation.
"""
import numpy as np
import librosa
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10
N_MFCC = 13


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def speech_before(x, sr, pause_start, window_s=1.5):
    """The last `window_s` seconds of audio strictly before the pause."""
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    """Short-time energy per frame, in dB."""
    fr = frames(x, sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    """Fundamental frequency of one frame via autocorrelation.

    Returns 0.0 for unvoiced/silent frames.
    """
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    """Per-frame F0 (Hz), 0.0 where unvoiced. Longer frames help pitch."""
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)


def trailing_voiced_run(f0):
    """Length (in frames) and values of the last contiguous voiced stretch."""
    voiced_mask = f0 > 0
    if not voiced_mask.any():
        return np.array([])
    idx = np.where(voiced_mask)[0]
    # find break in contiguity from the end
    end = idx[-1]
    start = end
    for i in range(len(idx) - 1, 0, -1):
        if idx[i] - idx[i - 1] <= 1:
            start = idx[i - 1]
        else:
            break
    return f0[start:end + 1]


def f0_slope(run, hop_ms=10):
    if len(run) < 3:
        return 0.0
    t = np.arange(len(run)) * hop_ms / 1000.0
    slope = np.polyfit(t, run, 1)[0]
    return float(slope)


def energy_slope(e_db, n_frames=30):
    tail = e_db[-n_frames:]
    if len(tail) < 3:
        return 0.0
    t = np.arange(len(tail))
    return float(np.polyfit(t, tail, 1)[0])


###############################################################
#                  FEATURE EXTRACTION
###############################################################

def extract_energy_features(seg, sr):
    """Energy based features."""
    energy = frame_energy_db(seg, sr)

    if len(energy) == 0:
        return np.zeros(8, dtype=np.float32)

    tail = energy[-30:] if len(energy) >= 30 else energy
    slope = energy_slope(energy)

    return np.array([
        np.mean(energy),
        np.std(energy),
        np.min(energy),
        np.max(energy),
        energy[-1],
        np.mean(tail),
        slope,
        tail[-1] - tail[0]
    ], dtype=np.float32)


def extract_voicing_features(f0):
    """Voicing related features."""

    voiced_mask = f0 > 0

    if len(f0) == 0 or not np.any(voiced_mask):
        return np.zeros(5, dtype=np.float32)

    run = trailing_voiced_run(f0)
    voiced_vals = f0[voiced_mask]

    return np.array([
        np.sum(voiced_mask),
        np.mean(voiced_mask),
        len(run),
        np.max(voiced_vals),
        np.min(voiced_vals)
    ], dtype=np.float32)


def extract_mfcc_statistics(mfcc, delta, delta2):
    """Statistical summary of MFCC features."""

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(delta, axis=1),
        np.std(delta, axis=1),

        np.mean(delta2, axis=1),
        np.std(delta2, axis=1)
    ]).astype(np.float32)

def compute_mfcc(seg, sr, n_mfcc=N_MFCC, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """MFCC + delta + delta-delta for a (causal) audio segment.

    n_fft / hop_length are derived from FRAME_MS / HOP_MS and the segment's
    sample rate, matching the framing used everywhere else in this file.
    """
    n_fft = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    mfcc = librosa.feature.mfcc(
        y=seg,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return mfcc, delta, delta2

def extract_spectral_features(seg, sr):
    """Spectral statistics."""

    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=seg)
    zcr = librosa.feature.zero_crossing_rate(seg)

    features = []

    for feat in [centroid, bandwidth, rolloff, flatness, zcr]:
        features.extend([
            np.mean(feat),
            np.std(feat)
        ])

    return np.array(features, dtype=np.float32)

def extract_pitch_features(f0):
    """Pitch related statistics."""

    voiced = f0[f0 > 0]

    if len(voiced) == 0:
        return np.zeros(7, dtype=np.float32)

    run = trailing_voiced_run(f0)

    return np.array([
        np.mean(voiced),
        np.std(voiced),
        np.min(voiced),
        np.max(voiced),
        voiced[-1],
        len(run),
        f0_slope(run)
    ], dtype=np.float32)

def extract_features(x, sr, pause_start, window_s=1.5):
    """
    Complete feature extraction pipeline.
    """
    
    seg = speech_before(x, sr, pause_start, window_s)
    if len(seg) == 0:
        return np.zeros(108, dtype=np.float32)

    energy_features = extract_energy_features(seg, sr)

    f0 = f0_contour(seg, sr)

    pitch_features = extract_pitch_features(f0)

    voicing_features = extract_voicing_features(f0)

    mfcc, delta, delta2 = compute_mfcc(seg, sr)

    mfcc_features = extract_mfcc_statistics(
        mfcc,
        delta,
        delta2
    )

    spectral_features = extract_spectral_features(seg, sr)

    return np.concatenate([
        energy_features,
        pitch_features,
        voicing_features,
        mfcc_features,
        spectral_features
    ])