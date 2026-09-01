"""
Heuristic (no-torch) spoof detector — the portability safety net.

This runs on numpy alone, so the app ALWAYS starts and always returns a verdict,
even with no ML deps, no GPU and no internet. It is a signal-artefact detector,
not a learned model, and it is honestly weaker than the ML backend — the API
labels every heuristic result as such.

It measures the classic tells of vocoder / TTS output:

  1. high-band energy    - synthesised speech usually lacks breath/air energy
                           above ~6 kHz
  2. high-band cliff     - many vocoders and codecs roll off hard well below
                           Nyquist
  3. spectral flux       - frame-to-frame spectral change is smoother in
                           generated speech
  4. voiced ratio        - TTS is densely voiced; humans insert breaths, pauses
                           and unvoiced noise
  5. pitch micro-jitter  - human f0 wobbles frame to frame; TTS contours are
                           smooth
  6. f0 spread           - human intonation covers a wider pitch range
  7. syllable modulation - 2-8 Hz energy modulation differs between real and
                           generated prosody

Each feature is z-scored (clipped at +/-3 sigma) and pushed through a logistic
function. The coefficients below are FITTED, not guessed — see
backend/calibrate_heuristic.py for the script, the calibration set and the
measured leave-one-file-out score (92.8% window accuracy / 7.8% EER on a small
LibriSpeech-vs-macOS-TTS set with band-pass and noise augmentation).

That calibration set is small and covers exactly two sources, so treat this
backend as a smoke test that keeps the product alive, not as evidence. The API
says so in every heuristic response.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

import config
from preprocessing import stft_magnitude

_EPS = 1e-10

MODEL_NAME = "heuristic-fallback"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _frame_rms(wav: np.ndarray, sr: int, frame_ms: float = 25.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000.0))
    n = wav.size // frame
    if n < 2:
        return np.array([float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + _EPS))])
    frames = wav[: n * frame].reshape(n, frame).astype(np.float64)
    return np.sqrt(np.mean(frames**2, axis=1) + _EPS)


def _f0_track(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Cheap autocorrelation pitch tracker (60-400 Hz) over 40 ms frames.
    Returns f0 in Hz for voiced frames only.
    """
    frame = int(sr * 0.040)
    hop = int(sr * 0.020)
    lo, hi = int(sr / 400.0), int(sr / 60.0)
    out = []
    for start in range(0, max(0, wav.size - frame), hop):
        seg = wav[start : start + frame].astype(np.float64)
        seg = seg - seg.mean()
        energy = float(np.dot(seg, seg))
        if energy < 1e-6:
            continue
        corr = np.correlate(seg, seg, mode="full")[frame - 1 :]
        if corr.size <= hi:
            continue
        window = corr[lo:hi]
        peak = int(np.argmax(window)) + lo
        if corr[peak] / (corr[0] + _EPS) < 0.30:   # unvoiced / noise
            continue
        out.append(sr / float(peak))
    return np.asarray(out, dtype=np.float64)


def _band_cliff_hz(mean_spec_db: np.ndarray, freqs: np.ndarray) -> float:
    """Highest frequency still within 45 dB of the spectral peak."""
    threshold = mean_spec_db.max() - 45.0
    above = np.nonzero(mean_spec_db > threshold)[0]
    return float(freqs[above[-1]]) if above.size else float(freqs[-1])


def _modulation_strength(rms: np.ndarray, sr_env: float) -> float:
    """Energy in the 2-8 Hz syllable band of the loudness envelope, normalised."""
    if rms.size < 8:
        return 0.0
    env = np.log(rms + _EPS)
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env)) ** 2
    freqs = np.fft.rfftfreq(env.size, d=1.0 / sr_env)
    band = (freqs >= 2.0) & (freqs <= 8.0)
    total = float(spec[1:].sum()) + _EPS
    return float(spec[band].sum() / total)


def extract_features(wav: np.ndarray, sr: int = config.TARGET_SR) -> Dict[str, float]:
    """All heuristic features for one waveform. Cheap: one STFT + one f0 pass."""
    mag = stft_magnitude(wav)                     # (freq, frames)
    freqs = np.fft.rfftfreq(512, d=1.0 / sr)
    power = mag.astype(np.float64) ** 2 + _EPS

    # --- spectral flatness (geometric / arithmetic mean per frame)
    log_mean = np.exp(np.mean(np.log(power), axis=0))
    arith_mean = np.mean(power, axis=0)
    flatness = log_mean / (arith_mean + _EPS)

    # --- centroid + rolloff
    frame_energy = power.sum(axis=0)
    centroid = (freqs[:, None] * power).sum(axis=0) / frame_energy
    cumulative = np.cumsum(power, axis=0) / frame_energy
    rolloff95 = freqs[np.argmax(cumulative >= 0.95, axis=0)]

    # --- high band share (6-8 kHz) of total energy
    hf = power[freqs >= 6000].sum() / power.sum()

    # --- spectral flux (frame-to-frame change of the normalised spectrum)
    norm_spec = mag / (np.linalg.norm(mag, axis=0, keepdims=True) + _EPS)
    flux = float(np.mean(np.abs(np.diff(norm_spec, axis=1)))) if mag.shape[1] > 1 else 0.0

    # --- loudness dynamics + noise floor
    rms = _frame_rms(wav, sr)
    rms_db = 20.0 * np.log10(rms + _EPS)
    floor_db = float(np.percentile(rms_db, 5))
    peak_db = float(np.percentile(rms_db, 95))
    silence_gap_db = peak_db - floor_db            # big gap => digital silence

    # --- pitch micro-jitter
    f0 = _f0_track(wav, sr)
    if f0.size >= 4:
        rel = np.abs(np.diff(f0)) / (f0[:-1] + _EPS)
        jitter = float(np.median(rel))
        f0_std = float(np.std(f0) / (np.mean(f0) + _EPS))
        voiced_ratio = float(f0.size / max(1, (wav.size // int(sr * 0.020))))
    else:
        jitter, f0_std, voiced_ratio = 0.02, 0.05, 0.0

    return {
        "flatness_mean": float(np.mean(flatness)),
        "flatness_std": float(np.std(flatness)),
        "centroid_mean_hz": float(np.mean(centroid)),
        "centroid_std_hz": float(np.std(centroid)),
        "rolloff95_mean_hz": float(np.mean(rolloff95)),
        "high_band_ratio": float(hf),
        "band_cliff_hz": _band_cliff_hz(20.0 * np.log10(mag.mean(axis=1) + _EPS), freqs),
        "spectral_flux": flux,
        "silence_gap_db": silence_gap_db,
        "rms_std_db": float(np.std(rms_db)),
        "pitch_jitter": jitter,
        "f0_rel_std": f0_std,
        "voiced_ratio": voiced_ratio,
        # the loudness envelope is sampled at one value per 25 ms frame -> 40 Hz
        "modulation_2_8hz": _modulation_strength(rms, 1.0 / 0.025),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
# (feature, human_reference, scale, weight)
#   weight > 0 : values ABOVE the reference push towards FAKE
#   weight < 0 : values BELOW the reference push towards FAKE
_RULES = (
    ("high_band_ratio",  0.0200282, 0.0347044, -2.359),
    ("band_cliff_hz",    6750.43,   1988.82,   +2.348),
    ("spectral_flux",    0.0125232, 0.00369075, -1.971),
    ("voiced_ratio",     0.720656,  0.1274,    +1.894),
    ("pitch_jitter",     0.0264988, 0.00790672, -1.013),
    ("f0_rel_std",       0.416281,  0.156241,  -0.828),
    ("modulation_2_8hz", 0.446495,  0.14624,   +0.637),
)
_BIAS = -2.420


def spoof_probability(features: Dict[str, float]) -> float:
    """Combine the cues into a spoof probability in [0.02, 0.98]."""
    logit = _BIAS
    for name, reference, scale, weight in _RULES:
        z = (features.get(name, reference) - reference) / scale
        logit += weight * float(np.clip(z, -3.0, 3.0))
    prob = 1.0 / (1.0 + math.exp(-logit))
    return float(min(0.98, max(0.02, prob)))


def top_reasons(features: Dict[str, float], limit: int = 3) -> list[str]:
    """Human-readable explanation of what drove the score (for the UI)."""
    labels = {
        "high_band_ratio": "breath / air energy above 6 kHz",
        "band_cliff_hz": "where the high band rolls off",
        "spectral_flux": "frame-to-frame spectral change",
        "voiced_ratio": "share of continuously voiced speech",
        "pitch_jitter": "pitch micro-jitter",
        "f0_rel_std": "pitch range across the clip",
        "modulation_2_8hz": "syllable-rate (2-8 Hz) energy modulation",
        "flatness_mean": "spectral flatness of the excitation",
        "silence_gap_db": "cleanliness of the silence between words",
        "rms_std_db": "loudness dynamics",
        "centroid_std_hz": "timbre variation",
    }
    scored = []
    for name, reference, scale, weight in _RULES:
        z = float(np.clip((features.get(name, reference) - reference) / scale, -3.0, 3.0))
        scored.append((abs(weight * z), weight * z, name))
    scored.sort(reverse=True)
    # Report the measurement and which way it pushed the score — never a causal
    # claim, because a single logistic coefficient does not license one.
    return [
        f"{labels.get(name, name)} pushed the score towards "
        f"{'FAKE' if signed > 0 else 'HUMAN'}"
        for _, signed, name in scored[:limit]
        if abs(signed) > 0.25
    ]


def predict(wav: np.ndarray, sr: int = config.TARGET_SR) -> Dict[str, object]:
    """Score one window. Mirrors the ML detector's return shape."""
    feats = extract_features(wav, sr)
    return {
        "spoof_probability": spoof_probability(feats),
        "features": feats,
    }
