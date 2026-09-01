# VoiceGuard — bulletproof containerised run.
#
#   docker compose up          -> http://localhost:8000   (heuristic detector)
#   docker compose build --build-arg WITH_ML=true && docker compose up
#                              -> same app with the pretrained ML model
#
# ffmpeg is installed so EVERY audio format works inside the container,
# including m4a/aac.

FROM python:3.11-slim

# Set to "true" to bake the heavy ML stack (torch + transformers) into the image.
ARG WITH_ML=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HF_HOME=/home/app/.cache/huggingface

# libsndfile1 -> soundfile decoding · ffmpeg -> m4a/aac support
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first so Docker caches the (slow) install layer.
COPY backend/requirements.txt backend/requirements-ml.txt ./backend/
RUN pip install --upgrade pip \
    && pip install -r backend/requirements.txt \
    && if [ "$WITH_ML" = "true" ]; then \
         pip install --extra-index-url https://download.pytorch.org/whl/cpu \
             -r backend/requirements-ml.txt ; \
       fi

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY sample_audio/ ./sample_audio/

# Run as a non-root user; give it a writable HOME for the HuggingFace cache.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app /home/app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fs http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
