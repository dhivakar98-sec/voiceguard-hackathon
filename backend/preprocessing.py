"""
Audio loading + preprocessing + spectrogram rendering — pure numpy.

Deliberately has NO librosa / scipy / matplotlib dependency:

* decoding      -> soundfile (libsndfile, bundled in the wheel) with an
                   automatic ffmpeg-CLI fallback for m4a/aac if ffmpeg happens
                   to be installed,
* resampling    -> FFT-based resampler (inherently anti-aliased),
* silence trim  -> frame energy gate,
* spectrogram   -> numpy STFT + a ~90-line pure-stdlib PNG encoder (zlib).

That keeps the core install to 6 pure-Python wheels, which is what makes this
project survive being zipped to another machine.
"""

from __future__ import annotations

import base64
import io
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import soundfile as sf

import config


class AudioError(ValueError):
    """Raised for anything the user can fix — surfaces as an HTTP 400."""


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def ffmpeg_available() -> bool:
    """True if an ffmpeg binary is on PATH (only needed for m4a/aac/wma)."""
    return shutil.which("ffmpeg") is not None


def _decode_with_soundfile(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, int(sr)


def _decode_with_ffmpeg(path: Path) -> Tuple[np.ndarray, int]:
    """Last resort for containers libsndfile cannot open (m4a/aac/mp4/...)."""
    if not ffmpeg_available():
        raise AudioError(
            f"'{path.suffix or 'this format'}' needs ffmpeg, which is not installed. "
            "Either convert the clip to .wav / .flac / .mp3 / .ogg, or install ffmpeg "
            "(macOS: brew install ffmpeg · Windows: choco install ffmpeg · "
            "Linux: sudo apt install ffmpeg)."
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(path),
                "-ac", "1", "-ar", str(config.TARGET_SR), "-f", "wav",
                str(wav_path),
            ],
            capture_output=True,
            timeout=120,
        )
        if proc.returncode != 0 or wav_path.stat().st_size == 0:
            detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            raise AudioError(
                "ffmpeg could not decode this file — it may be corrupt or not audio. "
                + (detail[-1] if detail else "")
            )
        return _decode_with_soundfile(wav_path)
    except subprocess.TimeoutExpired:
        raise AudioError("Decoding timed out — the file is too large or corrupt.")
    finally:
        wav_path.unlink(missing_ok=True)


def load_audio(path: str | Path, display_name: str | None = None) -> Tuple[np.ndarray, int]:
    """
    Decode any supported file to a float32 array (samples, channels) + sample rate.
    Raises AudioError with an actionable message on anything unreadable.
    """
    path = Path(path)
    name = display_name or path.name
    suffix = Path(name).suffix.lower()

    if suffix and suffix not in config.SUPPORTED_FORMATS:
        raise AudioError(
            f"Unsupported file type '{suffix}'. Supported: "
            + ", ".join(sorted(config.SUPPORTED_FORMATS))
        )

    if suffix in config.FFMPEG_ONLY_FORMATS:
        return _decode_with_ffmpeg(path)

    try:
        return _decode_with_soundfile(path)
    except AudioError:
        raise
    except Exception as exc:  # libsndfile failure -> try ffmpeg, then give up
        try:
            return _decode_with_ffmpeg(path)
        except AudioError as ff_exc:
            raise AudioError(
                f"Could not read '{name}' as audio ({exc}). {ff_exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def to_mono(data: np.ndarray) -> np.ndarray:
    """(samples, channels) or (samples,) -> (samples,) float32."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """
    FFT-domain resampler. Truncating the spectrum when downsampling doubles as
    the anti-alias filter, so no filter design is needed.
    """
    if sr_in == sr_out or wav.size == 0:
        return wav.astype(np.float32, copy=False)

    n_in = wav.size
    n_out = int(round(n_in * sr_out / float(sr_in)))
    if n_out < 1:
        raise AudioError("Clip is too short to resample.")

    spec = np.fft.rfft(wav)
    keep = min(spec.size, n_out // 2 + 1)
    new_spec = np.zeros(n_out // 2 + 1, dtype=complex)
    new_spec[:keep] = spec[:keep]
    out = np.fft.irfft(new_spec, n=n_out) * (n_out / float(n_in))
    return out.astype(np.float32, copy=False)


def peak_normalise(wav: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak < 1e-9:
        return wav
    return (wav * (target / peak)).astype(np.float32, copy=False)


def trim_silence(
    wav: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    top_db: float = 45.0,
) -> np.ndarray:
    """
    Drop leading/trailing frames more than `top_db` below the loudest frame.
    Returns the original array if trimming would leave (almost) nothing.
    """
    frame = max(1, int(sr * frame_ms / 1000.0))
    if wav.size < frame * 3:
        return wav

    n_frames = wav.size // frame
    frames = wav[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    ref = float(rms.max())
    if ref < 1e-8:
        return wav  # all-silent: let the caller report it

    keep = rms > ref * (10.0 ** (-top_db / 20.0))
    if not keep.any():
        return wav
    first, last = int(np.argmax(keep)), int(n_frames - np.argmax(keep[::-1]))
    trimmed = wav[first * frame : min(wav.size, last * frame)]
    return trimmed if trimmed.size >= frame * 3 else wav


def is_effectively_silent(wav: np.ndarray) -> bool:
    if wav.size == 0:
        return True
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    return rms < 1e-4


def prepare(path: str | Path, display_name: str | None = None) -> Tuple[np.ndarray, float, float]:
    """
    Full pipeline: decode -> mono -> 16 kHz -> normalise -> trim silence.

    Returns (waveform_16k, original_duration_sec, analysed_duration_sec).
    """
    raw, sr = load_audio(path, display_name)
    wav = to_mono(raw)

    if wav.size == 0:
        raise AudioError("The file contains no audio samples.")

    original_duration = wav.size / float(sr)
    if original_duration > config.MAX_DURATION_SEC:
        wav = wav[: int(sr * config.MAX_DURATION_SEC)]

    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    wav = resample(wav, sr, config.TARGET_SR)

    if is_effectively_silent(wav):
        raise AudioError(
            "This clip is silent (or near-silent) — there is no voice to analyse. "
            "Please upload a clip with audible speech."
        )

    wav = peak_normalise(wav)
    wav = trim_silence(wav, config.TARGET_SR)

    analysed_duration = wav.size / float(config.TARGET_SR)
    if analysed_duration < config.MIN_DURATION_SEC:
        raise AudioError(
            f"Clip is too short after trimming silence ({analysed_duration:.2f}s). "
            f"Please upload at least {config.MIN_DURATION_SEC:.1f}s of speech."
        )
    return wav, original_duration, analysed_duration


def sliding_windows(
    wav: np.ndarray,
    sr: int = config.TARGET_SR,
    window_sec: float = config.WINDOW_SEC,
    overlap: float = config.WINDOW_OVERLAP,
) -> Iterator[Tuple[float, float, np.ndarray]]:
    """
    Yield (start_sec, end_sec, samples) windows with `overlap` fraction of
    overlap. Clips shorter than one window yield exactly one window.
    """
    win = max(1, int(sr * window_sec))
    if wav.size <= win:
        yield 0.0, wav.size / float(sr), wav
        return

    hop = max(1, int(win * (1.0 - overlap)))
    for start in range(0, wav.size - win + 1, hop):
        yield start / float(sr), (start + win) / float(sr), wav[start : start + win]

    # Cover a trailing remainder longer than a quarter window.
    tail_start = ((wav.size - win) // hop) * hop + hop
    if wav.size - tail_start > win // 4:
        yield tail_start / float(sr), wav.size / float(sr), wav[tail_start:]


# ---------------------------------------------------------------------------
# STFT + spectrogram PNG (no matplotlib)
# ---------------------------------------------------------------------------

def stft_magnitude(wav: np.ndarray, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    """Magnitude STFT with a Hann window -> (n_freq_bins, n_frames)."""
    if wav.size < n_fft:
        wav = np.pad(wav, (0, n_fft - wav.size))
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (wav.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = wav[idx] * window
    return np.abs(np.fft.rfft(frames, axis=1)).T.astype(np.float32)


_COLORMAP_ANCHORS = np.array(
    [  # a viridis-like ramp (dark blue -> teal -> green -> yellow)
        (13, 8, 60),
        (39, 62, 138),
        (32, 118, 142),
        (48, 167, 119),
        (145, 209, 58),
        (253, 231, 37),
    ],
    dtype=np.float32,
)


def _colormap(norm: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB uint8 via linear interpolation of anchors."""
    x = np.clip(norm, 0.0, 1.0) * (len(_COLORMAP_ANCHORS) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(_COLORMAP_ANCHORS) - 1)
    frac = (x - lo)[..., None]
    rgb = _COLORMAP_ANCHORS[lo] * (1 - frac) + _COLORMAP_ANCHORS[hi] * frac
    return rgb.astype(np.uint8)


def _png_encode(rgb: np.ndarray) -> bytes:
    """Minimal but valid PNG encoder (RGB8, no filtering) using only stdlib."""
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw, 9)),
            chunk(b"IEND", b""),
        ]
    )


def _resize_nearest(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    ys = np.clip((np.arange(out_h) * h // max(1, out_h)), 0, h - 1)
    xs = np.clip((np.arange(out_w) * w // max(1, out_w)), 0, w - 1)
    return img[ys][:, xs]


def spectrogram_png_base64(
    wav: np.ndarray,
    sr: int = config.TARGET_SR,
    out_height: int = 256,
    max_width: int = 900,
) -> str | None:
    """Log-magnitude spectrogram as a base64 PNG (frequency increases upward)."""
    try:
        mag = stft_magnitude(wav)
        db = 20.0 * np.log10(mag + 1e-8)
        db = np.maximum(db, db.max() - 80.0)          # 80 dB dynamic range
        norm = (db - db.min()) / max(1e-6, db.max() - db.min())
        # Quantising to 48 levels is visually indistinguishable but roughly
        # halves the PNG (and therefore the JSON response) size.
        norm = np.round(norm * 47.0) / 47.0
        img = _colormap(norm[::-1])                    # low freq at the bottom
        width = min(max_width, max(120, img.shape[1]))
        img = _resize_nearest(img, out_height, width)
        return base64.b64encode(_png_encode(np.ascontiguousarray(img))).decode("ascii")
    except Exception:
        return None  # a missing picture must never fail an analysis


def looks_like_speech(wav: np.ndarray) -> Tuple[bool, float]:
    """
    Cheap sanity check: does this clip plausibly contain speech at all?

    Uses per-frame normalised spectral entropy. Speech spreads energy over
    formants and keeps moving, so entropy sits around 0.4-0.55 and varies frame
    to frame. Tones, hum, music beds and DTMF sit far below that and barely
    vary. Measured on the calibration set: real speech >= 0.48, TTS >= 0.33,
    a swept tone 0.21 with an entropy std of 0.02.

    Returns (looks_like_speech, mean_entropy). We do not refuse the analysis —
    the API just warns that the verdict is unreliable, which is the honest
    answer for input the detector was never trained on.
    """
    try:
        mag = stft_magnitude(wav)
        power = mag.astype(np.float64) ** 2 + 1e-12
        power /= power.sum(axis=0, keepdims=True)
        entropy = -(power * np.log(power)).sum(axis=0) / np.log(power.shape[0])
        mean_h, std_h = float(np.mean(entropy)), float(np.std(entropy))
        return (mean_h >= 0.30 and std_h >= 0.06), mean_h
    except Exception:
        return True, float("nan")   # never block an analysis on this check


def write_wav(path: str | Path, wav: np.ndarray, sr: int = config.TARGET_SR) -> None:
    """Small helper used by the sample-audio generator and tests."""
    sf.write(str(path), np.clip(wav, -1.0, 1.0).astype(np.float32), sr, subtype="PCM_16")


def wav_bytes(wav: np.ndarray, sr: int = config.TARGET_SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav, -1.0, 1.0).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
