"""
VoiceGuard configuration.

Everything here can be overridden with environment variables so the teammate
never has to edit code. Example:

    VG_DETECTOR_MODE=heuristic PORT=8080 ./run.sh
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- paths -------
# Resolved relative to this file so the app runs from ANY working directory.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SAMPLE_AUDIO_DIR = PROJECT_ROOT / "sample_audio"
MODELS_DIR = BACKEND_DIR / "models"

# ------------------------------------------------------------- detector -------
# "auto"      -> try ML, silently fall back to the heuristic on ANY failure
# "ml"        -> require ML (still falls back rather than crashing, but logs loudly)
# "heuristic" -> never touch torch/transformers (fastest start, works offline)
DETECTOR_MODE = _env_str("VG_DETECTOR_MODE", "auto").lower()

# Primary pretrained audio-deepfake classifier.
#
# Chosen by benchmarking 8 Hub candidates on a real-vs-synthetic mini set under
# clean and telephone-band conditions — see IMPLEMENTATION.md section "Model
# selection" for the table. This one was the only candidate that was BOTH well
# separated (0% EER) and correctly calibrated at a 0.5 cut, and it held up under
# 300-3400 Hz band-pass. wav2vec2-base, 378 MB, Apache-2.0.
#
# The candidates are tried in order; the first one that loads wins. Every one of
# them is a transformers audio-classification model, and detector.py reads the
# real/fake label order out of the config, so swapping the id is safe.
MODEL_ID = _env_str("VG_MODEL_ID", "Bisher/wav2vec2_ASV_deepfake_audio_detection")
MODEL_FALLBACK_IDS = [
    "mo-thecreator/Deepfake-audio-detection",
    "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection",
    "Om-Parab/distilhubert-finetuned-audio-deepfake-in-the-wild",
]

# A locally fine-tuned checkpoint directory (produced by backend/train.py).
# If it exists it takes priority over the Hub model — and it lets the ML backend
# work with no internet at all.
LOCAL_MODEL_DIR = Path(_env_str("VG_LOCAL_MODEL_DIR", str(MODELS_DIR / "finetuned")))

# spoof_probability above this => FAKE.
THRESHOLD = _env_float("VG_THRESHOLD", 0.5)
THRESHOLD_IS_EXPLICIT = "VG_THRESHOLD" in os.environ

# Pretrained Hub classifiers separate the two classes well but are calibrated
# for their own training distribution, so a raw 0.5 cut is often in the wrong
# place. These operating points were measured (see README section 6) — a model
# not listed here just uses THRESHOLD. VG_THRESHOLD always wins if you set it.
MODEL_THRESHOLDS = {
    # Measured 0% EER at a plain 0.5 cut — no correction needed.
    "Bisher/wav2vec2_ASV_deepfake_audio_detection": 0.5,
    # Separates well (3.6% EER) but scores almost everything high, so 0.5 would
    # call every clip FAKE.
    "mo-thecreator/Deepfake-audio-detection": 0.95,
    "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection": 0.95,
    "Om-Parab/distilhubert-finetuned-audio-deepfake-in-the-wild": 0.5,
}


def threshold_for(model_id: str) -> float:
    """Operating point for the active model."""
    if THRESHOLD_IS_EXPLICIT:
        return THRESHOLD
    return MODEL_THRESHOLDS.get(model_id, THRESHOLD)

# ------------------------------------------------------------ audio I/O -------
TARGET_SR = 16_000                       # every detector sees 16 kHz mono
WINDOW_SEC = _env_float("VG_WINDOW_SEC", 4.0)     # sliding-window length
WINDOW_OVERLAP = 0.5                     # 50 % overlap -> hop = 2 s
MIN_DURATION_SEC = _env_float("VG_MIN_DURATION_SEC", 0.7)   # reject shorter clips
MAX_DURATION_SEC = _env_float("VG_MAX_DURATION_SEC", 120.0)  # analyse at most this
MAX_UPLOAD_BYTES = _env_int("VG_MAX_UPLOAD_MB", 40) * 1024 * 1024

# Formats libsndfile handles with no external tools. m4a/aac need ffmpeg.
SOUNDFILE_FORMATS = {".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aiff", ".aif", ".au"}
FFMPEG_ONLY_FORMATS = {".m4a", ".aac", ".mp4", ".wma", ".webm", ".amr", ".3gp"}
SUPPORTED_FORMATS = SOUNDFILE_FORMATS | FFMPEG_ONLY_FORMATS

# ---------------------------------------------------------------- server ------
HOST = _env_str("HOST", "0.0.0.0")
PORT = _env_int("PORT", 8000)

HONEST_NOTE = (
    "Result is strong evidence, not absolute proof. Accuracy drops on heavily "
    "compressed, noisy or unseen-generator audio — always second-check a "
    "high-stakes decision."
)
HEURISTIC_NOTE = (
    "Running on the lightweight signal-heuristic detector. It is noticeably "
    "less accurate than the ML backend — treat this as a smoke-test result, "
    "not evidence. Install backend/requirements-ml.txt to enable the "
    "pretrained model."
)
HEURISTIC_FORCED_NOTE = (
    "Running on the lightweight signal-heuristic detector because "
    "VG_DETECTOR_MODE=heuristic was requested. Unset it to use the pretrained "
    "ML model."
)
