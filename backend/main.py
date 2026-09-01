"""
VoiceGuard — FastAPI app.

Serves BOTH the API and the frontend on one port, so there is no CORS setup,
no second dev server and no port juggling for whoever runs this next.

    GET  /              -> frontend/index.html
    GET  /api/health    -> which detector loaded
    POST /api/analyze   -> multipart 'file' -> verdict + confidence

Run:
    python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import config
import detector as detector_module
import preprocessing
from preprocessing import AudioError
from schemas import AnalyzeResponse, HealthResponse

VERSION = "1.0.0"

logging.basicConfig(
    level=os.environ.get("VG_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voiceguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm the detector in a background thread so the server answers immediately
    even while a (potentially large) model is still downloading.
    """
    threading.Thread(
        target=detector_module.get_detector, name="detector-warmup", daemon=True
    ).start()
    log.info("VoiceGuard %s starting — open http://localhost:%s", VERSION, config.PORT)
    yield


app = FastAPI(
    title="VoiceGuard",
    version=VERSION,
    description="AI voice-clone / deepfake audio detector (SIH26104 MVP).",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Error handling — a bad upload must never take the server down
# ---------------------------------------------------------------------------
@app.exception_handler(AudioError)
async def audio_error_handler(_: Request, exc: AudioError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error while analysing the audio: {exc}"},
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if not detector_module.is_ready():
        # Still warming up (usually only on the very first ML model download).
        return HealthResponse(
            status="loading",
            backend="loading",
            model="loading",
            mode=config.DETECTOR_MODE,
            device="cpu",
            threshold=config.THRESHOLD,
            ffmpeg=preprocessing.ffmpeg_available(),
            ml_load_error=None,
            version=VERSION,
        )
    status = detector_module.get_detector().status()
    return HealthResponse(version=VERSION, **status)


@app.get("/api/samples")
async def samples() -> dict:
    """
    List whatever clips are sitting in sample_audio/ so the UI can offer them as
    one-click demos. Drop a new file in the folder and it appears — no code edit.
    """
    if not config.SAMPLE_AUDIO_DIR.is_dir():
        return {"samples": []}
    files = sorted(
        p.name
        for p in config.SAMPLE_AUDIO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_FORMATS
    )
    return {"samples": files}


@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    responses={400: {"description": "Unsupported, corrupt, silent or too-short audio"}},
)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    started = time.perf_counter()

    filename = (file.filename or "upload").strip() or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. Supported formats: "
                + ", ".join(sorted(config.SUPPORTED_FORMATS))
            ),
        )

    payload = await file.read()
    await file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(payload) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File is too large ({len(payload) / 1e6:.1f} MB). "
                f"Limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB — "
                "trim the clip or raise VG_MAX_UPLOAD_MB."
            ),
        )

    with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)

    try:
        result = await run_in_threadpool(_analyse_file, tmp_path, filename)
    except AudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    result["processing_ms"] = int((time.perf_counter() - started) * 1000)
    return AnalyzeResponse(**result)


def _analyse_file(path: Path, filename: str) -> dict:
    """Blocking work: decode -> preprocess -> detect -> spectrogram."""
    wav, duration, analysed = preprocessing.prepare(path, filename)

    det = detector_module.get_detector()
    result = det.analyse(wav)

    note = config.HONEST_NOTE
    if result["backend"] == "heuristic":
        prefix = (
            config.HEURISTIC_FORCED_NOTE if det.mode == "heuristic" else config.HEURISTIC_NOTE
        )
        note = prefix + " " + config.HONEST_NOTE

    warnings: list[str] = []
    is_speech, entropy = preprocessing.looks_like_speech(wav)
    if not is_speech:
        warnings.append(
            "This clip does not look like speech (tone, music or noise). "
            "The detector is built for voice, so treat this verdict as meaningless."
        )
    if analysed < config.WINDOW_SEC:
        warnings.append(
            f"Only {analysed:.1f}s of speech was analysed. Scores get noticeably "
            f"more reliable above {config.WINDOW_SEC:.0f}s."
        )
    result.update(
        {
            "duration_sec": round(duration, 2),
            "analysed_sec": round(analysed, 2),
            "filename": filename,
            "spectrogram_png_base64": preprocessing.spectrogram_png_base64(wav),
            "note": note,
            "warnings": warnings,
        }
    )
    log.info(
        "%s -> %s (%d/100, p=%.3f, %s, %.1fs)",
        filename, result["verdict"], result["confidence"],
        result["spoof_probability"], result["backend"], analysed,
    )
    return result


# ---------------------------------------------------------------------------
# Frontend (mounted last so /api/* always wins)
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    index_file = config.FRONTEND_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=500, detail="frontend/index.html is missing.")
    return FileResponse(index_file)


if config.SAMPLE_AUDIO_DIR.is_dir():
    app.mount(
        "/sample_audio",
        StaticFiles(directory=str(config.SAMPLE_AUDIO_DIR)),
        name="sample_audio",
    )

if config.FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":  # `python backend/main.py` also works
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
