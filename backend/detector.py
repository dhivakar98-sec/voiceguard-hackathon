"""
Detection core: pretrained ML model with an automatic heuristic fallback.

Contract (never changes, whichever backend is live):

    detector.analyse(waveform_16k) -> {
        "verdict": "HUMAN" | "FAKE",
        "confidence": int 50-100,         # distance from the decision boundary
        "spoof_probability": float 0-1,   # mean over the sliding windows
        "max_segment_probability": float 0-1,
        "threshold": float 0-1,           # boundary actually used
        "backend": "ml" | "heuristic",
        "model": "<model id>",
        "segments": [{start, end, spoof_probability}, ...],
        "reasons": [str, ...],            # heuristic backend only
    }

Backend selection (env var VG_DETECTOR_MODE, default "auto"):
    auto      -> try ML; on ANY failure log one line and use the heuristic
    ml        -> same, but log loudly that ML was expected
    heuristic -> never import torch (fastest start, fully offline)

Nothing in here is allowed to raise during startup. A missing model, no
internet, a blocked proxy or an OOM all degrade to the heuristic instead.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional

import numpy as np

import config
import heuristic
import preprocessing

log = logging.getLogger("voiceguard.detector")

# Labels that mean "this is synthetic" in a HuggingFace audio-classifier config.
_FAKE_LABEL_HINTS = ("fake", "spoof", "synthetic", "deepfake", "generated", "ai", "clone")
_REAL_LABEL_HINTS = ("real", "bonafide", "human", "genuine", "authentic", "live")


class MLBackend:
    """Thin wrapper around a HuggingFace audio-classification model."""

    def __init__(self, model_id: str) -> None:
        import torch  # imported lazily so the core install never needs it
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.torch = torch
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.eval().to(self.device)
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2))))

        self.fake_index = self._resolve_fake_index()
        # A tiny warm-up so the first real request is not the slowest one.
        self.predict(np.zeros(config.TARGET_SR, dtype=np.float32))

    # -- label mapping -------------------------------------------------------
    def _resolve_fake_index(self) -> int:
        """
        Work out which logit means "spoof" from id2label, so swapping the model
        id never silently inverts the verdict.
        """
        id2label = getattr(self.model.config, "id2label", None) or {}
        normalised = {int(k): str(v).strip().lower() for k, v in id2label.items()}

        for idx, label in normalised.items():
            if any(hint == label or hint in label.split("_") for hint in _FAKE_LABEL_HINTS):
                return idx
        for idx, label in normalised.items():
            if any(hint in label for hint in _FAKE_LABEL_HINTS):
                return idx
        for idx, label in normalised.items():
            if any(hint in label for hint in _REAL_LABEL_HINTS):
                return 1 - idx if len(normalised) == 2 else idx
        log.warning(
            "Could not map %s labels %s to real/fake — assuming index 0 = fake.",
            self.model_id, id2label,
        )
        return 0

    @property
    def label_map(self) -> Dict[int, str]:
        return dict(getattr(self.model.config, "id2label", {}) or {})

    # -- inference -----------------------------------------------------------
    def predict(self, wav: np.ndarray) -> Dict[str, object]:
        torch = self.torch
        inputs = self.feature_extractor(
            [np.asarray(wav, dtype=np.float32)],
            sampling_rate=config.TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0].float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        p_fake = float(probs[self.fake_index]) if probs.size > 1 else float(probs[0])
        return {"spoof_probability": p_fake}


class Detector:
    """Public entry point. Owns backend selection and the sliding-window logic."""

    def __init__(self) -> None:
        self.mode = config.DETECTOR_MODE if config.DETECTOR_MODE in {"auto", "ml", "heuristic"} else "auto"
        self.backend = "heuristic"
        self.model_name = heuristic.MODEL_NAME
        self.ml: Optional[MLBackend] = None
        self.load_error: Optional[str] = None
        self._lock = threading.Lock()   # HF models are not thread-safe
        self._load()

    # -- startup -------------------------------------------------------------
    def _candidate_model_ids(self) -> List[str]:
        candidates: List[str] = []
        if config.LOCAL_MODEL_DIR.is_dir() and any(config.LOCAL_MODEL_DIR.iterdir()):
            candidates.append(str(config.LOCAL_MODEL_DIR))
        candidates.append(config.MODEL_ID)
        candidates.extend(m for m in config.MODEL_FALLBACK_IDS if m != config.MODEL_ID)
        return candidates

    def _load(self) -> None:
        if self.mode == "heuristic":
            log.info("VG_DETECTOR_MODE=heuristic -> using the numpy heuristic detector.")
            return

        errors: List[str] = []
        for model_id in self._candidate_model_ids():
            try:
                log.info("Loading ML detector '%s' ...", model_id)
                self.ml = MLBackend(model_id)
                self.backend = "ml"
                self.model_name = model_id
                log.info(
                    "ML detector ready: %s on %s (labels=%s, fake index=%d)",
                    model_id, self.ml.device, self.ml.label_map, self.ml.fake_index,
                )
                return
            except ImportError as exc:
                # torch / transformers are simply not installed — trying the
                # other candidates would log the same error three more times.
                errors.append(f"{exc}")
                break
            except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash
                errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
                log.info("ML detector '%s' unavailable (%s: %s)", model_id, type(exc).__name__, exc)

        self.load_error = " | ".join(errors)
        message = (
            "ML detector unavailable -> falling back to the heuristic detector. "
            "Install backend/requirements-ml.txt (and allow the first model download) "
            "to enable the pretrained model. Reason: "
            + (errors[0] if errors else "unknown")
        )
        (log.warning if self.mode == "ml" else log.info)(message)

    @property
    def threshold(self) -> float:
        """Operating point for whichever backend is currently live."""
        if self.backend == "ml":
            return config.threshold_for(self.model_name)
        return config.THRESHOLD

    # -- inference -----------------------------------------------------------
    def _score_window(self, window: np.ndarray) -> Dict[str, object]:
        if self.backend == "ml" and self.ml is not None:
            try:
                with self._lock:
                    return self.ml.predict(window)
            except Exception as exc:  # noqa: BLE001 - runtime failure -> degrade live
                log.warning("ML inference failed (%s) — using the heuristic for this clip.", exc)
                self.backend = "heuristic"
                self.model_name = heuristic.MODEL_NAME
                self.ml = None
        return heuristic.predict(window)

    def analyse(self, wav: np.ndarray) -> Dict[str, object]:
        """Sliding-window inference + averaging -> the API payload fields."""
        segments: List[Dict[str, float]] = []
        probabilities: List[float] = []
        last_features: Dict[str, float] = {}

        for start, end, window in preprocessing.sliding_windows(wav):
            result = self._score_window(window)
            p = float(result["spoof_probability"])
            probabilities.append(p)
            last_features = result.get("features") or last_features
            segments.append(
                {"start": round(start, 2), "end": round(end, 2), "spoof_probability": round(p, 3)}
            )

        if not probabilities:            # cannot happen, but never divide by zero
            probabilities = [0.5]

        # Mean over windows is the stable estimate; the max is reported so a
        # short spliced-in fake segment is still visible in the UI.
        mean_p = float(np.mean(probabilities))
        threshold = self.threshold
        verdict = "FAKE" if mean_p > threshold else "HUMAN"

        # Confidence = distance from the decision boundary, rescaled to 50-100.
        # At threshold 0.5 this is identical to round(100 * p_predicted_class);
        # with a tuned threshold it stays meaningful instead of reporting "6%
        # confident FAKE" for a score just under a 0.95 cut.
        if verdict == "FAKE":
            margin = (mean_p - threshold) / max(1e-6, 1.0 - threshold)
        else:
            margin = (threshold - mean_p) / max(1e-6, threshold)
        confidence = int(round(50 + 50 * min(1.0, max(0.0, margin))))

        reasons: List[str] = []
        if self.backend == "heuristic" and last_features:
            reasons = heuristic.top_reasons(last_features)

        return {
            "verdict": verdict,
            "confidence": confidence,
            "spoof_probability": round(mean_p, 3),
            "threshold": round(threshold, 3),
            "max_segment_probability": round(float(np.max(probabilities)), 3),
            "backend": self.backend,
            "model": self.model_name,
            "segments": segments,
            "reasons": reasons,
        }

    # -- introspection -------------------------------------------------------
    def status(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "backend": self.backend,
            "model": self.model_name,
            "mode": self.mode,
            "device": getattr(self.ml, "device", "cpu"),
            "threshold": self.threshold,
            "ffmpeg": preprocessing.ffmpeg_available(),
            "ml_load_error": self.load_error if self.backend != "ml" else None,
        }


_detector: Optional[Detector] = None
_init_lock = threading.Lock()


def get_detector() -> Detector:
    """Process-wide singleton (the model is loaded exactly once, thread-safe)."""
    global _detector
    if _detector is None:
        with _init_lock:
            if _detector is None:
                _detector = Detector()
    return _detector


def is_ready() -> bool:
    return _detector is not None
