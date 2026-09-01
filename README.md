# VoiceGuard — AI Voice Clone / Deepfake Audio Detector (MVP)

> **SIH26104** — Real-time detection & prevention of voice-cloning impersonation attacks.
> This document is a **build specification**. Hand it to **Claude Code** and ask it to build the project **from scratch**, following the steps in order.

---

## 0. Instructions for Claude Code (read first)

**Goal:** Build a working web app where a user uploads an audio file, the backend deeply analyzes it with an ML model, and the UI shows whether the voice is **HUMAN** or **FAKE (AI-cloned)** with a **confidence score out of 100**.

**Build rules:**
1. Build in the exact **phase order** in Section 9. Get each phase running before moving on.
2. Prefer a **pretrained model** for immediate results; add **fine-tuning** only as the accuracy-boost step (Section 6).
3. Before hardcoding any HuggingFace model ID, **verify it exists and loads** (`transformers` / `huggingface_hub`). If a listed candidate is unavailable, search the Hub for a current audio deepfake / anti-spoofing model and swap it in — keep the API contract unchanged.
4. Keep the code clean, commented, and runnable with the commands in Section 10. Do not add features beyond this MVP scope unless Section 12 (roadmap) is explicitly requested.
5. Handle errors gracefully (bad file, wrong format, silence, too-short clip) and never crash the server on a bad upload.
6. **PORTABILITY IS A HARD REQUIREMENT.** This project will be zipped and sent to a teammate on a different machine (possibly Windows, possibly no GPU, possibly flaky internet). It must run there with **near-zero setup errors**. Follow Section 14 (Portability & Packaging) strictly:
   - Provide **one-command run scripts** for both Windows (`run.bat`) and macOS/Linux (`run.sh`) that create the environment, install dependencies, and start the app.
   - Provide a **Docker + docker-compose** path as the bulletproof fallback.
   - **Pin every dependency version** so installs are reproducible.
   - Make the **ML model optional at runtime**: if torch/transformers or the model download is unavailable, the app must **automatically fall back to a lightweight heuristic detector** and still start and return a verdict — never crash on a missing model.
   - Ship a **no-build frontend** (plain HTML/CSS/JS served by the backend) as the default so the teammate needs **no Node.js/npm**. The React version is optional.

---

## 1. What we're building (MVP scope)

**In scope (build this now):**
- A simple, clean **frontend**: one page with a drag-and-drop / file-picker **audio upload** area, an "Analyze" action, a loading state, and a **result card**.
- A **backend** that accepts the audio, preprocesses it, runs an ML deepfake-detection model, and returns a verdict + confidence.
- **Output:** `HUMAN` or `FAKE`, plus a **confidence score 0-100**, plus (nice-to-have) a spectrogram image and per-segment scores.

**Out of scope for MVP (future — see Section 12):** real-time streaming, live call interception, call blocking, speaker identity verification, mobile app.

**Supported input formats:** `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg` (backend resamples everything to 16 kHz mono internally).

---

## 2. Core requirement (the one-line contract)

> Upload audio -> backend deeply analyzes it with the best available ML detector -> returns **{ verdict: HUMAN | FAKE, confidence: 0-100 }** -> frontend displays it clearly.

---

## 3. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | **Python 3.10+, FastAPI, Uvicorn** | Async, scales to real-time later |
| ML (optional) | **PyTorch, torchaudio, transformers** (+ `huggingface_hub`) | SSL-based deepfake detection; **optional install** |
| Fallback detector | **librosa + numpy** signal heuristic | Runs with **no torch, no internet**; keeps app alive |
| Audio | **librosa, soundfile** (+ optional `ffmpeg`) | wav/flac/ogg work out of the box; mp3/m4a need ffmpeg |
| **Frontend (default)** | **Plain HTML + CSS + vanilla JS, served by FastAPI** | **No Node.js, no npm, no build step** — most portable |
| Frontend (optional) | React + Vite + Tailwind | Only if the teammate has Node; not required to run |

> **Why the default frontend is plain HTML:** the teammate must be able to run the whole app with **only Python installed**. Serving a static page from FastAPI removes the entire Node/npm/build class of errors. Keep the React version as an *optional* upgrade, not a dependency.

---

## 4. Architecture

```
+--------------+        POST /api/analyze (multipart audio)        +------------------------+
|  Frontend    |  ---------------------------------------------->  |  FastAPI Backend       |
| (React/Vite) |                                                   |                        |
|              |                                                   | 1. Save temp file      |
| - upload box |                                                   | 2. Load + preprocess   |
| - Analyze btn|                                                   |    (16kHz mono, trim)  |
| - result card|  <----------------------------------------------  | 3. ML detector infer   |
+--------------+   { verdict, confidence, spectrogram, segments }  | 4. Build response      |
                                                                   +------------------------+
                                                                             |
                                                                   +------------------------+
                                                                   |  Detection Core (ML)   |
                                                                   |  wav2vec2/SSL + head   |
                                                                   |  -> spoof probability  |
                                                                   +------------------------+
```

---

## 5. Project structure (create exactly this)

```
voiceguard/
|-- README.md                  # this file (or a short RUNME.md for the teammate)
|-- run.sh                     # one-command setup+run for macOS/Linux
|-- run.bat                    # one-command setup+run for Windows
|-- Dockerfile                 # bulletproof containerized run
|-- docker-compose.yml         # `docker compose up` -> done
|-- .gitignore
|-- backend/
|   |-- requirements.txt       # CORE deps only (light, almost always installs)
|   |-- requirements-ml.txt    # OPTIONAL heavy ML deps (torch, transformers)
|   |-- main.py                # FastAPI app: API routes + serves the frontend
|   |-- detector.py            # ML detector WITH automatic heuristic fallback
|   |-- heuristic.py           # signal-based fallback detector (no torch)
|   |-- preprocessing.py       # load, resample 16k mono, VAD/trim, spectrogram
|   |-- schemas.py             # Pydantic response models
|   |-- config.py              # model id, thresholds, mode (env-overridable)
|   |-- train.py               # (Step B, optional) ASVspoof fine-tuning + EER
|   \-- models/                # saved/fine-tuned checkpoints (gitignored)
|-- frontend/                  # DEFAULT: no-build static UI (served by FastAPI)
|   |-- index.html             # upload UI + result card
|   |-- style.css
|   \-- app.js                 # calls /api/analyze with fetch
|-- sample_audio/              # a few real + fake clips for the demo
\-- (optional) frontend-react/ # optional React+Vite version; NOT required to run
```

---

## 6. Detection core (the ML — most important part)

### Strategy: pretrained first, fine-tune to boost accuracy

**Step A — Zero-training MVP (do this first):**
Use a **pretrained self-supervised audio deepfake / anti-spoofing model** from HuggingFace so the app works immediately without training.
- Candidate approach: a **wav2vec2 / XLS-R model fine-tuned for real-vs-fake (bonafide-vs-spoof) audio classification.**
- Claude Code: **search the HuggingFace Hub** for a current audio deepfake detection / ASVspoof anti-spoofing model, verify it loads via `transformers`, and wire it into `detector.py` behind a simple `predict(waveform) -> {label, confidence}` function.
- If no reliable ready-made classifier is found, fall back to: `facebook/wav2vec2-xls-r-300m` as a **frozen feature extractor** + a small trained MLP head (needs Step B data).

**Step B — Accuracy boost (fine-tune on ASVspoof):**
Fine-tune / train the classifier head on **ASVspoof 2019 LA** (bonafide vs spoof).
- Report **EER** and **accuracy** on the eval set. **Target: >=95% accuracy / low EER** (SSL models reach this on ASVspoof LA).
- Save the checkpoint to `backend/models/` and load it in `detector.py`.

**Robustness (keeps accuracy high on real audio — do if time allows):**
- Train with **augmentation**: MP3/codec compression, added background noise, telephone band-pass (300-3400 Hz). This is what makes it survive real phone audio and is your key differentiator.
- Add **non-English / Indian-language** samples to the eval set to prove generalization.

### Automatic fallback (REQUIRED for portability)
`detector.py` must select its backend at startup based on a `VG_DETECTOR_MODE` env var (`auto` | `ml` | `heuristic`, default `auto`):
- **`auto`:** try to import torch/transformers and load the model. On **any** failure (not installed, no internet, model download blocked, out of memory) **catch it, log a clear one-line message, and fall back to the heuristic detector in `heuristic.py`.** The app must still start and `/api/analyze` must still return a valid verdict.
- The API response includes a `"backend"` field (`"ml"` or `"heuristic"`) and a `"model"` field so it's always clear which detector produced the result.
- **`heuristic.py`:** a lightweight detector using librosa/numpy features only (e.g. spectral flatness, high-frequency energy roll-off, pitch/energy variance, silence/breath cues) that maps to a spoof probability. It is **not** as accurate as the ML model — label it honestly in the response note — but it guarantees the app runs on any machine with only the core deps.

### Inference logic (`detector.py`)
1. Input: 16 kHz mono waveform.
2. If clip > ~4 s, run a **sliding window** (e.g. 4 s windows, 50% overlap), score each, and **average** -> stable score + per-segment scores.
3. Model outputs spoof probability `p in [0,1]`.
4. `verdict = "FAKE" if p > THRESHOLD else "HUMAN"` (default `THRESHOLD = 0.5`, tune on the eval set).
5. `confidence = round(100 * (p if verdict=="FAKE" else 1-p))` -> confidence out of 100 in the predicted class.

### Honest accuracy note (put a short version of this in the UI/docs)
High accuracy (95%+) is realistic **on standard benchmarks**; accuracy can drop on heavily compressed, noisy, or unseen-generator audio — this is true of all detectors and is why augmentation + a "second-check" recommendation matter. Present the score as **strong evidence, not absolute proof**.

---

## 7. API contract

### `POST /api/analyze`
- **Request:** `multipart/form-data`, field `file` = audio file.
- **Response `200`:**
```json
{
  "verdict": "FAKE",
  "confidence": 96,
  "spoof_probability": 0.962,
  "duration_sec": 7.4,
  "backend": "ml",
  "model": "<model id used, or 'heuristic-fallback'>",
  "segments": [
    { "start": 0.0, "end": 4.0, "spoof_probability": 0.95 },
    { "start": 2.0, "end": 6.0, "spoof_probability": 0.97 }
  ],
  "spectrogram_png_base64": "iVBORw0KGgo...",
  "note": "Result is strong evidence, not absolute proof."
}
```
- **Errors:** `400` for unsupported/corrupt/too-short audio, with a clear `detail` message.

### `GET /api/health`
- Returns `{ "status": "ok", "backend": "ml" | "heuristic", "model": "<id>" }` so the teammate can instantly see which detector loaded.

---

## 8. Frontend spec (default = `frontend/index.html` + `style.css` + `app.js`)

Build the default UI as **plain HTML/CSS/vanilla JS served by FastAPI** (no Node, no build). Keep it clean and simple:
- **Header:** app name "VoiceGuard" + one-line tagline.
- **Upload zone:** drag-and-drop + click-to-browse; show selected filename; accept only supported audio types.
- **Analyze button:** disabled until a file is chosen; shows a spinner + "Deeply analyzing audio..." while the request is in flight (`fetch` POST to `/api/analyze`).
- **Result card:**
  - Big verdict badge: **HUMAN** (green) or **FAKE / AI-CLONED** (red).
  - **Confidence: XX/100** with a progress bar.
  - A small tag showing which **backend** produced the result (`ml` or `heuristic`).
  - Small print: the honest "strong evidence, not proof" note.
  - (Nice-to-have) render the returned **spectrogram** image and a small per-segment score list.
- **Error state:** friendly message if the backend returns an error.
- Plain CSS; mobile-friendly, centered, minimal. Served from the same origin as the API so there are **no CORS issues** for the teammate.

> Optional React+Vite+Tailwind version may live in `frontend-react/` but must NOT be required to run the app.

---

## 9. Build order for Claude Code (do in sequence)

1. **Scaffold** the folder structure (Section 5) and init git with `.gitignore` (ignore `venv/`, `node_modules/`, `backend/models/`, `__pycache__/`, temp uploads).
2. **Backend skeleton:** FastAPI app with `/api/health`; serve `frontend/index.html` at `/`. Confirm it runs with only the core deps.
3. **Preprocessing:** implement load -> resample 16 kHz mono -> trim silence -> spectrogram helper.
4. **Heuristic detector (`heuristic.py`) FIRST:** implement the no-torch fallback so the app is fully working end to end with only core deps. Wire `/api/analyze`.
5. **No-build frontend:** `index.html` + `style.css` + `app.js`; upload UI + result card calling `/api/analyze`. Test end to end.
6. **ML detector (Step A):** add `detector.py` that loads a verified pretrained model and, on any failure, **falls back to the heuristic** (Section 6). Same API contract.
7. **Portability layer:** write `run.sh`, `run.bat`, `Dockerfile`, `docker-compose.yml`, pin all versions, split `requirements.txt` / `requirements-ml.txt` (Section 14).
8. **End-to-end test:** on a clean environment, run each entry point (script + Docker) and confirm the app starts and returns a verdict even with ML deps absent.
9. **Accuracy boost (Step B, optional):** add ASVspoof fine-tuning (`train.py`) + augmentation; report EER/accuracy; switch `detector.py` to the fine-tuned model.
10. **Polish:** loading states, error handling, spectrogram display, verify `RUNME` instructions on both OSes.

---

## 10. Setup & run (what the teammate actually does)

The teammate should need **one** of these, and nothing else:

**Option A — one command (recommended):**
```bash
# macOS / Linux
./run.sh
```
```bat
:: Windows (double-click or run in terminal)
run.bat
```
The script creates a virtual environment, installs the **core** dependencies, starts the server, and prints the URL. Then open **http://localhost:8000**.

**Option B — Docker (most bulletproof, needs Docker installed):**
```bash
docker compose up
```
Then open **http://localhost:8000**.

**Optional — enable the real ML model** (bigger download, needs internet):
```bash
# after Option A, with the venv active:
pip install -r backend/requirements-ml.txt
```
Without this, the app runs on the built-in heuristic detector and still works.

### `backend/requirements.txt` — CORE (light, pin exact versions)
Only what's needed to run with the heuristic detector. This install almost never fails:
```
fastapi==<pinned>
uvicorn[standard]==<pinned>
python-multipart==<pinned>
librosa==<pinned>
soundfile==<pinned>
numpy==<pinned>
matplotlib==<pinned>
```

### `backend/requirements-ml.txt` — OPTIONAL (heavy, pin exact versions)
```
torch==<pinned>
torchaudio==<pinned>
transformers==<pinned>
huggingface_hub==<pinned>
scikit-learn==<pinned>
```
> Claude Code: replace `<pinned>` with the actual latest compatible versions at build time and verify the core set installs cleanly in a fresh venv.
> `ffmpeg` is only needed for **mp3/m4a**; wav/flac/ogg work without it. The UI should tell the user this instead of crashing.

---

## 11. Testing / demo
- Put 2-3 **real** clips and 2-3 **AI-cloned** clips in `sample_audio/`.
- **Killer demo:** record a teammate, clone their voice with a free TTS/voice-cloning tool, and show the app flag the clone (FAKE) while passing the real voice (HUMAN).
- Verify: correct verdicts, sensible confidence, graceful handling of a non-speech / silent / corrupt file.

---

## 12. Future roadmap (NOT in MVP — build later on request)
1. **Real-time streaming:** WebSocket audio in ~1-2 s windows, sliding-window inference, live verdict meter (<1 s latency).
2. **Live call protection:** integrate into a mock call/meeting UI; **alert, interrupt, or block** on detection; audit log.
3. **Speaker impersonation check (identity):** enroll a reference voiceprint (ECAPA-TDNN) and verify the caller is actually who they claim — catches impersonation even of genuine-sounding audio.
4. **Challenge-response prevention:** on suspicion, prompt for a random phrase that cloning pipelines struggle to produce live.
5. **Mobile app / on-device** inference; multilingual (Indian languages) + phone-channel hardened models.
6. **Explainability:** highlight the spectrogram regions / artifacts that drove the FAKE verdict.

---

## 13. One-line pitch (for judges)
"VoiceGuard detects AI-cloned voices from an audio clip and tells you HUMAN or FAKE with a confidence score — built to be real-time, multilingual, and phone-channel robust, targeting the exact conditions where today's enterprise tools drop accuracy."

---

## 14. Portability & packaging (SHIP-TO-TEAMMATE — build this carefully)

**Objective:** the teammate unzips the folder and runs it on their machine with near-zero errors, regardless of OS, GPU, or internet quality.

### 14.1 One-command run scripts
Create both scripts. Each must: detect Python, create/reuse a venv, upgrade pip, install `backend/requirements.txt` (core only), then launch uvicorn on port 8000. On failure, print a **clear, human-readable message** telling the teammate exactly what to do (see Section 15).

**`run.sh` (macOS/Linux)** must:
- check `python3` exists (else print the install link and exit cleanly),
- `python3 -m venv .venv` (reuse if present),
- `.venv/bin/pip install --upgrade pip`,
- `.venv/bin/pip install -r backend/requirements.txt`,
- `.venv/bin/python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000`,
- be marked executable (`chmod +x`).

**`run.bat` (Windows)** must do the equivalent with `py -3` / `python`, `\.venv\Scripts\`, and clear `echo` messages. Avoid PowerShell-only syntax so it works from double-click and cmd.

### 14.2 Docker (bulletproof fallback)
- **`Dockerfile`:** `python:3.11-slim` base, `apt-get install -y ffmpeg` (so all audio formats work), copy project, `pip install -r backend/requirements.txt`, expose 8000, `CMD` runs uvicorn. Keep the ML deps out of the default image (optional build arg to include them).
- **`docker-compose.yml`:** one `web` service, build from the Dockerfile, map `8000:8000`, mount `sample_audio/` for convenience. `docker compose up` must be the entire process.

### 14.3 Reproducible dependencies
- **Pin exact versions** in both requirements files (no floating versions).
- Core file must **not** contain torch/transformers — those live only in `requirements-ml.txt`.
- Verify in a **fresh** venv that `requirements.txt` installs with no build errors on Python 3.10/3.11.

### 14.4 Zero-config, same-origin
- Frontend is served by FastAPI at `/` from the same port as the API -> **no CORS, no second server, no ports to configure**.
- No hardcoded absolute paths; use paths relative to the project root so it runs from any folder.
- No secrets or API keys required to run the MVP.

### 14.5 What goes in the zip (and what doesn't)
- **Include:** all source, both requirements files, run scripts, Dockerfile/compose, README/RUNME, a couple of tiny sample clips in `sample_audio/`.
- **Exclude via `.gitignore` / clean before zipping:** `.venv/`, `__pycache__/`, `node_modules/`, `backend/models/` (large checkpoints), any large datasets, OS junk (`.DS_Store`).
- Add a short **`RUNME.md`** at the root with just the 3 lines the teammate needs (Option A / Option B / open the URL), so they don't have to read this whole spec.

### 14.6 Pre-send self-check (Claude Code must run this before declaring done)
In a clean environment with **ML deps NOT installed**:
1. `./run.sh` (and mentally verify `run.bat` mirrors it) starts the server with no traceback.
2. `GET /api/health` returns `backend: "heuristic"`.
3. Uploading a sample wav returns a valid verdict + confidence.
4. `docker compose up` serves the same working app.
If all four pass, the package is teammate-ready.

---

## 15. Worst case: what might still need the teammate's input

The app is designed to run with **only Python installed**. These are the only things that could require action on their machine — surface each as a clear message, never a crash:

1. **Python not installed / too old.** Needs **Python 3.10+**. Script should detect this and print the download link (python.org). *This is the single most likely blocker.*
2. **Docker not installed** (only if they choose Option B). Needs Docker Desktop. Not required if they use Option A.
3. **mp3 / m4a uploads without ffmpeg.** wav/flac/ogg work out of the box. For mp3/m4a they either install ffmpeg (`brew install ffmpeg` / `choco install ffmpeg` / `apt install ffmpeg`) or just use a wav file. The UI must say this rather than erroring.
4. **Enabling the real ML model.** Optional. Requires `pip install -r backend/requirements-ml.txt` (a large torch download) and internet for the first model fetch. Until then it runs on the heuristic detector — fully functional, lower accuracy.
5. **Port 8000 already in use.** Rare. The script should let them override the port (e.g. `PORT` env var) and print how.
6. **Corporate network / proxy blocking pip or the model download.** Only affects the optional ML step; the core app still runs offline once core deps are installed.

> Everything above except #1 is optional or has an automatic fallback. If the teammate has Python 3.10+, Option A should "just work."
