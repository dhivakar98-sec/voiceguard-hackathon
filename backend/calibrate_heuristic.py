"""
Dev tool: fit the weights used by heuristic.py.

The heuristic detector is a logistic regression over 8 hand-designed signal
features. This script fits those coefficients on a folder of real clips and a
folder of synthetic clips, then prints a ready-to-paste _RULES table.

    python backend/calibrate_heuristic.py --real DIR_OF_REAL --fake DIR_OF_FAKE

Requires scikit-learn (see backend/requirements-ml.txt). Augmentation
(telephone band-pass, added noise, gain change) is applied to BOTH classes so
the fit cannot cheat by keying on channel bandwidth alone.

Provenance of the shipped weights: 14 LibriSpeech dev-clean utterances
(real, public domain) vs 14 macOS AVSpeechSynthesis voices at 16/22.05 kHz
(synthetic), x4 augmentations, 4-second windows, grouped leave-one-file-out CV.
Small and single-source on both sides — good enough for a working fallback,
which is exactly why the ML backend is the default.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import heuristic  # noqa: E402
import preprocessing as P  # noqa: E402

FEATURES = [name for name, *_ in heuristic._RULES]


# --------------------------------------------------------------- augmentation
def telephone_band(wav: np.ndarray, sr: int = config.TARGET_SR) -> np.ndarray:
    """300-3400 Hz band-pass in the FFT domain (mimics a phone channel)."""
    spec = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(wav.size, d=1.0 / sr)
    spec[(freqs < 300) | (freqs > 3400)] = 0
    return np.fft.irfft(spec, n=wav.size).astype(np.float32)


def add_noise(wav: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.randn(wav.size).astype(np.float32)
    sig_p = float(np.mean(wav**2)) + 1e-12
    noise *= np.sqrt(sig_p / (10 ** (snr_db / 10.0)) / (np.mean(noise**2) + 1e-12))
    return (wav + noise).astype(np.float32)


def augmentations(wav: np.ndarray) -> List[np.ndarray]:
    return [
        wav,
        telephone_band(wav),
        add_noise(wav, 20.0),
        P.peak_normalise(add_noise(telephone_band(wav), 25.0), 0.4),
    ]


# ------------------------------------------------------------------- dataset
def collect(folder: str, label: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, groups = [], [], []
    files = sorted(
        f for f in glob.glob(os.path.join(folder, "*"))
        if os.path.splitext(f)[1].lower() in config.SUPPORTED_FORMATS
    )
    if not files:
        raise SystemExit(f"No audio files found in {folder}")
    for path in files:
        try:
            wav, _, _ = P.prepare(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {os.path.basename(path)}: {exc}")
            continue
        for variant in augmentations(wav):
            for _, _, window in P.sliding_windows(variant):
                if window.size < config.TARGET_SR * 2:
                    continue
                feats = heuristic.extract_features(window)
                X.append([feats[k] for k in FEATURES])
                y.append(label)
                groups.append(os.path.basename(path))
    print(f"  {folder}: {len(files)} files -> {len(y)} windows (label={label})")
    return np.asarray(X, dtype=np.float64), np.asarray(y), groups


def eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="folder of genuine human clips")
    ap.add_argument("--fake", required=True, help="folder of synthetic/cloned clips")
    ap.add_argument("--C", type=float, default=0.35, help="inverse L2 strength")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut

    print("Extracting features ...")
    Xr, yr, gr = collect(args.real, 0)
    Xf, yf, gf = collect(args.fake, 1)
    X = np.vstack([Xr, Xf])
    y = np.concatenate([yr, yf])
    groups = np.array(gr + gf)

    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.clip((X - mean) / std, -3, 3)

    # Honest generalisation estimate: never score a file the fit has seen.
    logo = LeaveOneGroupOut()
    oof = np.zeros(len(y))
    for tr, te in logo.split(Z, y, groups):
        clf = LogisticRegression(C=args.C, max_iter=2000).fit(Z[tr], y[tr])
        oof[te] = clf.predict_proba(Z[te])[:, 1]
    acc = float(np.mean((oof > 0.5).astype(int) == y))
    print(f"\nLeave-one-file-out: window accuracy = {acc * 100:.1f}%  EER = {eer(y, oof) * 100:.1f}%")

    clf = LogisticRegression(C=args.C, max_iter=2000).fit(Z, y)
    w, b = clf.coef_[0], float(clf.intercept_[0])

    print("\n# ---- paste into heuristic.py ----")
    print("_RULES = (")
    order = np.argsort(-np.abs(w))
    for i in order:
        print(f'    ("{FEATURES[i]}", {mean[i]:.6g}, {std[i]:.6g}, {w[i]:+.3f}),')
    print(")")
    print(f"_BIAS = {b:+.3f}")


if __name__ == "__main__":
    main()
