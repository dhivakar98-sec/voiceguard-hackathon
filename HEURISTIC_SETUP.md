# VoiceGuard — Heuristic-Only Setup (Lightweight, No Big Installs)

Run the full flow — **upload audio → backend processing → accuracy check → HUMAN/FAKE → accuracy score** —
using only the built-in **heuristic detector**. No PyTorch, no transformers, no model download, no internet after setup.

> **Windows guide. Use Command Prompt (cmd), not PowerShell.**

---

## Important: what this mode is (read first)

The heuristic detector gives you the complete pipeline and runs instantly with tiny installs, but it is a
**logistic regression fitted on a small 2-source calibration set**, not a deep model trained on a real
anti-spoofing corpus. Measured 92.8% window accuracy / 7.8% EER leave-one-file-out on that set — but it
collapses on telephone-quality audio, and it can be wrong fairly often on real-world clips.

- Use it to **build and demo the pipeline** quickly, and on machines that can't handle big installs.
- Do **not** present its score as "95% accurate". No number in this project is a benchmark result yet —
  see IMPLEMENTATION.md section 2 for what was actually measured and on how little data.
- You can upgrade to the real ML model anytime later (see the bottom of this file).

The app will report `"backend": "heuristic"` so it's always clear which detector produced the result.

---

## What's REQUIRED
- **Python 3.10+** (the app is Python — non-negotiable)
- **Core dependencies only**: `backend/requirements.txt` (fastapi, uvicorn, python-multipart, pydantic, soundfile, numpy) — 6 pure-Python wheels, no compiler needed, almost never fails.
  *Note: librosa and matplotlib are deliberately NOT dependencies — all DSP is numpy and the spectrogram uses a stdlib PNG encoder. See IMPLEMENTATION.md section 4.*

## What's SAFE TO SKIP
- ❌ `backend/requirements-ml.txt` — the big torch/transformers install. **Skip entirely.**
- ❌ The torch CPU-wheel fallback command — not needed (no torch at all).
- ❌ Waiting for a model download on first start — there's no model, so startup is instant and needs no internet.
- ❌ **ffmpeg** — skippable **if** you only upload `.wav`, `.flac`, or `.ogg`. Only install it if you need `.mp3` / `.m4a`.
- ❌ The "`/api/health` must say ml" check — in this mode it correctly shows `"heuristic"`. That's expected, not an error.

---

## Steps (Windows / cmd)

### 1. Install Python only
Open **Command Prompt as Administrator**:
```bat
winget install Python.Python.3.11
```
(Skip ffmpeg unless you specifically need mp3/m4a. If `winget` isn't recognized, install Python from
https://www.python.org/downloads/ and tick **"Add Python to PATH"**.)

### 2. Close this terminal and open a NEW one
Needed so Python is picked up on your PATH. Open a normal (non-admin) Command Prompt.

### 3. Verify Python
```bat
python --version
```
Must show **3.10 or higher**.

### 4. Get into the project folder
Clone it (or unzip and cd into it):
```bat
cd C:\Users\%USERNAME%\Downloads
git clone https://github.com/dhivakar98-sec/voiceguard-hackathon.git
cd voiceguard-hackathon
```
(If you don't have Git, just unzip the folder and `cd` into it instead.)

### 5. Create and activate the virtual environment
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```
Your prompt should now start with `(.venv)`.

### 6. Install CORE dependencies only (NO ml requirements)
```bat
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

### 7. Force heuristic mode and start the app
```bat
set VG_DETECTOR_MODE=heuristic
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```
The `set VG_DETECTOR_MODE=heuristic` line tells the app to skip even trying to load the ML model —
so startup is clean and instant. Leave this window open; it's the running server.

### 8. Open it
Go to **http://localhost:8000** and upload a **`.wav`** file to test.

### 9. Confirm the mode
Open **http://localhost:8000/api/health** — it should show `"backend": "heuristic"`. That's correct for this mode.

---

## Everyday use (after first setup)

**Stop the server:** `Ctrl + C`.

**Run it again another day:**
```bat
cd C:\Users\%USERNAME%\Downloads\voiceguard-hackathon
.venv\Scripts\activate.bat
set VG_DETECTOR_MODE=heuristic
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Reinstall Python with "Add to PATH" ticked; reopen the terminal |
| Activation fails in PowerShell | Use cmd, or run `Set-ExecutionPolicy -Scope Process -Bypass` first |
| mp3/m4a upload errors | Use a `.wav` file, or install ffmpeg (`winget install Gyan.FFmpeg`) |
| Port 8000 already in use | Change `--port 8000` to `--port 8080`, open http://localhost:8080 |
| `/api/health` errors instead of showing `heuristic` | Shouldn't happen — `heuristic.py` and the `VG_DETECTOR_MODE` switch are both in the repo. Run `.venv\Scripts\python backend\selftest.py` and share the output. |

---

## Upgrading to the real ML model later

When you want the high-accuracy model, keep everything above and add:
```bat
pip install -r backend\requirements-ml.txt
set VG_DETECTOR_MODE=auto
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```
The first start will download the model once (needs internet), then `/api/health` should show `"backend": "ml"`.
