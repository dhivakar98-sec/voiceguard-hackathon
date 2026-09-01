# VoiceGuard — run it in 3 lines

You need **Python 3.10 or newer**. Nothing else. No Node, no ffmpeg, no GPU.

### Option A — one command (recommended)

**macOS / Linux**
```bash
./run.sh
```

**Windows** — double-click `run.bat`, or in `cmd`:
```bat
run.bat
```

Then open **<http://localhost:8000>** and upload an audio clip.

Not sure it worked? This proves the whole thing end to end and says exactly what
failed if anything did:

```bash
.venv/bin/python backend/selftest.py          # macOS / Linux
.venv\Scripts\python backend\selftest.py     # Windows
```

### Option B — Docker (if Option A gives you any trouble)

```bash
docker compose up
```
Then open **<http://localhost:8000>**.

---

## What you get

Upload a clip → the app answers **HUMAN** or **FAKE (AI-cloned)** with a
confidence score out of 100, a spectrogram, and per-segment scores.

There are demo clips in `sample_audio/` — they show up as one-click buttons in
the UI. Drop your own clips in that folder and they appear too.

## Two detection engines

The badge in the top-right corner tells you which one is live.

| Engine | When | Quality |
|---|---|---|
| **heuristic** | default — needs nothing but the core install | Works, but it is a signal heuristic. Fine for a smoke test, not evidence. |
| **ml** | after you install the optional ML deps | The real pretrained deepfake detector. Use this for the demo. |

**Turn on the ML engine** (~400 MB of packages + a 378 MB model, needs internet once):

```bash
# macOS / Linux
.venv/bin/pip install -r backend/requirements-ml.txt && ./run.sh
```
```bat
:: Windows
.venv\Scripts\pip install -r backend\requirements-ml.txt
run.bat
```

Check it worked: <http://localhost:8000/api/health> should say `"backend": "ml"`.
The first start after installing also downloads the model weights (378 MB), so
give it a minute — the page stays usable and shows "loading detector…".

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `Python 3.10+ not found` | Install from <https://www.python.org/downloads/>. On Windows tick **"Add python.exe to PATH"**, then open a new terminal. |
| `pip install failed` | You are offline or behind a company proxy. Try `docker compose up` instead. |
| `ensurepip` / `pyexpat` error on macOS | Your Homebrew Python is broken. `brew reinstall python@3.12 expat`, or just use `docker compose up`. |
| Port 8000 is busy | `PORT=8080 ./run.sh` (macOS/Linux) or `set PORT=8080 && run.bat` (Windows). |
| An `.m4a` upload is rejected | Only m4a/aac need ffmpeg. Use a `.wav`/`.mp3`/`.flac`/`.ogg`, or install ffmpeg (`brew install ffmpeg` / `choco install ffmpeg` / `apt install ffmpeg`). Docker already has it. |
| Badge says "backend offline" | The server is not running — check the terminal where you started it. |
| Windows: Microsoft Store opens instead of Python | Python is not really installed. Get it from python.org and tick **"Add python.exe to PATH"**. |
| Windows firewall prompt | Shouldn't appear — the server binds `127.0.0.1` by default. Only `HOST=0.0.0.0` triggers it. |

## Useful environment variables

```bash
PORT=8080                     # different port
HOST=0.0.0.0                  # reach the app from another device (firewall may prompt)
VG_DETECTOR_MODE=heuristic    # never load the ML model (fast start, fully offline)
VG_DETECTOR_MODE=ml           # complain loudly if the ML model cannot load
VG_THRESHOLD=0.45             # spoof probability above which the verdict is FAKE
VG_MAX_UPLOAD_MB=80           # allow bigger uploads
```

---

**One honest caveat to repeat when you demo it:** a verdict is strong evidence,
not proof. Accuracy drops on heavily compressed, noisy, or unseen-generator
audio — that is true of every detector on the market. `IMPLEMENTATION.md` §2
has the numbers we actually measured, and how small the test set was.
