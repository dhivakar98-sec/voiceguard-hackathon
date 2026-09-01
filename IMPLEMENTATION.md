# VoiceGuard — implementation notes

What was actually built against the spec in `README.md`, what was measured, and
where the build deviates from the spec on purpose.

* **To just run it:** `RUNME.md`.
* **The spec:** `README.md` (unchanged — it is the brief).
* **Strategy / roadmap:** `SIH26104_Build_Blueprint.md` (unchanged).

Status: **MVP complete.** Sections 1-8, 10, 11 and 14 of the spec are done and
verified end to end. Step B (ASVspoof fine-tuning) ships as a runnable script but
has not been run — it needs the dataset.

---

## 1. What runs

```
threat-detection-hackathon/
├── RUNME.md                     3 lines the teammate needs
├── IMPLEMENTATION.md            this file
├── run.sh / run.bat             one-command setup + run (macOS/Linux, Windows)
├── Dockerfile / docker-compose.yml
├── backend/
│   ├── requirements.txt         CORE: 6 pure-Python wheels, no compiler needed
│   ├── requirements-ml.txt      OPTIONAL: torch + transformers
│   ├── main.py                  FastAPI: /api/health, /api/samples, /api/analyze, serves the UI
│   ├── detector.py              ML backend + automatic heuristic fallback + sliding window
│   ├── heuristic.py             no-torch detector (fitted logistic over 7 signal features)
│   ├── preprocessing.py         decode, 16 kHz mono, trim, STFT, PNG spectrogram
│   ├── calibrate_heuristic.py   dev tool that fits heuristic.py's coefficients
│   ├── selftest.py              end-to-end check: start server, analyse every sample, assert
│   ├── train.py                 OPTIONAL ASVspoof 2019 LA fine-tuning + EER
│   ├── schemas.py, config.py
│   └── models/                  fine-tuned checkpoints land here (gitignored)
├── frontend/                    index.html + style.css + app.js — no Node, no build
└── sample_audio/                demo clips (anything you drop here appears in the UI)
```

Verify any machine with one command:

```bash
.venv/bin/python backend/selftest.py     # Windows: .venv\Scripts\python backend\selftest.py
```

It boots the real server on a free port, waits for `/api/health`, pushes every
clip in `sample_audio/` through `/api/analyze`, feeds it a corrupt file, checks
the server survived, and exits non-zero if anything failed.

---

## 2. Model selection (spec §6 step A)

The spec says: verify the model exists and loads before hardcoding it. It did
not say the first plausible model would be any good — it isn't. Eight Hub
candidates were benchmarked on the same set before one was chosen.

**Test set:** 14 genuine human utterances (LibriSpeech dev-clean, public domain)
vs 14 synthetic clips (macOS system TTS, 14 different voices, half at 16 kHz and
half at 22.05 kHz), each also rendered through a 300-3400 Hz telephone band-pass
= 56 clips. Mean spoof probability over 4 s / 50 %-overlap windows.

| Model | clean acc | clean EER | phone acc | phone EER | overall EER | ms/clip |
|---|---|---|---|---|---|---|
| **Bisher/wav2vec2_ASV_deepfake_audio_detection** | **96.4 %** | **0.0 %** | **92.9 %** | **0.0 %** | **0.0 %** | 145 |
| MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection | 50.0 % | 0.0 % | 50.0 % | 7.1 % | 3.6 % | 495 |
| mo-thecreator/Deepfake-audio-detection | 82.1 % | 3.6 % | 50.0 % | 7.1 % | 3.6 % | 141 |
| Hemgg/Deepfake-audio-detection | 53.6 % | 3.6 % | 71.4 % | 0.0 % | 7.1 % | 139 |
| Gustking/wav2vec2-large-xlsr-deepfake-audio-classification | 50.0 % | 10.7 % | 92.9 % | 17.9 % | 23.2 % | 379 |
| Om-Parab/distilhubert-finetuned-audio-deepfake-in-the-wild | 92.9 % | 7.1 % | 53.6 % | 7.1 % | 30.4 % | 71 |
| MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection | 50.0 % | 35.7 % | 50.0 % | 57.1 % | 46.4 % | 501 |
| MelodyMachine/Deepfake-audio-detection-V2 | 57.1 % | 64.3 % | 46.4 % | 57.1 % | 50.0 % | 139 |
| *heuristic fallback (for reference)* | *96.4 %* | *7.1 %* | *see §3* | | | *3* |

Accuracy is at a plain 0.5 cut; EER ignores the threshold and measures
separation only. The gap between the two columns is the interesting part.

**Two things this table decided:**

1. **The default model.** `Bisher/wav2vec2_ASV_deepfake_audio_detection`
   (wav2vec2-base, 378 MB, Apache-2.0) is the only candidate that was both well
   separated *and* correctly calibrated at 0.5, and it kept working through the
   telephone band-pass. It is now `config.MODEL_ID`.
   `MelodyMachine/Deepfake-audio-detection-V2` — the obvious pick by download
   count — is at chance on this set and would have shipped a broken demo.
2. **Thresholds are per-model.** Several models separate the classes almost
   perfectly but score nearly everything above 0.5, so a fixed 0.5 cut calls
   every clip FAKE. `config.MODEL_THRESHOLDS` carries a measured operating point
   per model, `config.threshold_for()` applies it, and `VG_THRESHOLD` always
   overrides. `confidence` is measured as distance from *that* boundary, so it
   stays meaningful (at threshold 0.5 it reduces exactly to the spec's
   `round(100 * p)`).

`detector.py` reads the real/fake label order from each model's `id2label`
(handling `fake`/`real`, `Bonafide`/`Spoof`, `AIVoice`/`HumanVoice`, and the
inverted orderings), so swapping `VG_MODEL_ID` cannot silently invert a verdict.

### Honest limits of that table

28 source clips is a **small, two-source** test set: LibriSpeech on one side,
macOS system TTS on the other. "0.0 % EER" means *these* two sources separate
perfectly — it is not an ASVspoof number and it is not a claim about modern
voice cloning. Any of the following will move it: real cloning tools (ElevenLabs
and friends), other languages, phone codecs, or a different recording chain. To
get a number worth quoting to judges, run `backend/train.py` on ASVspoof 2019 LA
and report the eval EER it prints.

---

## 3. The heuristic fallback (spec §6, required for portability)

Used whenever torch/transformers are missing, the download is blocked, or
`VG_DETECTOR_MODE=heuristic`. It is a logistic regression over 7 numpy signal
features — high-band energy ratio, high-band roll-off point, spectral flux,
voiced ratio, pitch micro-jitter, f0 spread, and 2-8 Hz syllable modulation.

The coefficients are **fitted, not guessed**: `backend/calibrate_heuristic.py`
fits them on the same real-vs-TTS set, with band-pass / noise / gain
augmentation applied to *both* classes so the fit cannot cheat on channel
bandwidth alone.

**Measured, leave-one-file-out (never scoring a file the fit has seen):**
**92.8 % window accuracy, 7.8 % EER** over 360 augmented 4 s windows.

Per-clip on the clean set it hits 96.4 % accuracy / 7.1 % EER — but on
telephone-band audio it collapses (it starts calling everything FAKE). It is
there to keep the product alive on a machine with only Python, and every
heuristic response says so in `note`. Do not demo on it.

To refit it on your own clips:

```bash
.venv/bin/python backend/calibrate_heuristic.py --real DIR_OF_REAL --fake DIR_OF_FAKE
```

It prints a ready-to-paste `_RULES` table plus the leave-one-file-out score.

---

## 4. Deliberate deviations from the spec

Each of these trades a spec detail for the spec's own stated hard requirement —
"PORTABILITY IS A HARD REQUIREMENT … near-zero setup errors" (§0.6, §14).

| Spec says | Built instead | Why |
|---|---|---|
| `librosa` in core requirements | **numpy only** (own FFT resampler, STFT, silence trim) | librosa needs `numba`, which lags new Python releases by months. On the machine this was built on (Python 3.14) `pip install librosa` cannot even build. It is the single most likely install failure, and none of its features were needed. |
| `matplotlib` in core requirements | **~90-line stdlib PNG encoder** (`zlib` + `struct`) | Removes a heavy dependency and the headless-backend class of bugs. Output verified: valid 900×256 RGB PNG, CRCs check. |
| `uvicorn[standard]` | **plain `uvicorn`** | `[standard]` pulls `uvloop`/`httptools`/`watchfiles` — compiled wheels that may not exist yet for a new Python. Nothing here needs them. |
| Single pinned version per package | pinned **per Python version** where needed (`numpy==1.26.4` for 3.10, `2.3.5` for 3.11-3.14) | No single numpy release covers 3.10-3.14 (2.3+ dropped 3.10). Both pins are exact, so installs are still reproducible. |
| `torchaudio` in ML requirements | **dropped** | Audio is decoded by `soundfile`; torchaudio was an unused ~100 MB and one more version-pairing constraint. |
| `confidence = round(100 * p)` | distance from the **model's** threshold, rescaled to 50-100 | Identical at threshold 0.5; sane when the threshold is 0.95, where the spec formula would report "6 % confident FAKE". |
| project root named `voiceguard/` | built in the existing repo root | This folder *is* the project root. One less level to unzip. |

**Added beyond the spec** (small, and each closes a real gap): `/api/samples` so
anything dropped in `sample_audio/` becomes a one-click demo; a `warnings[]`
field that flags non-speech input and very short clips instead of quietly
returning a confident number; `selftest.py`; `calibrate_heuristic.py`.

---

## 5. API contract

`POST /api/analyze` — multipart, field `file`:

```json
{
  "verdict": "FAKE",
  "confidence": 81,
  "spoof_probability": 0.805,
  "max_segment_probability": 0.991,
  "threshold": 0.5,
  "duration_sec": 10.27,
  "analysed_sec": 10.25,
  "backend": "ml",
  "model": "Bisher/wav2vec2_ASV_deepfake_audio_detection",
  "filename": "ai_tts_english.wav",
  "segments": [{ "start": 0.0, "end": 4.0, "spoof_probability": 0.95 }],
  "reasons": [],
  "warnings": [],
  "spectrogram_png_base64": "iVBORw0KGgo...",
  "note": "Result is strong evidence, not absolute proof. …",
  "processing_ms": 277
}
```

`400` with a plain-English `detail` for unsupported / corrupt / silent /
too-short audio, or an upload over `VG_MAX_UPLOAD_MB` (40 by default).

`GET /api/health` → `status`, `backend` (`ml` | `heuristic` | `loading`),
`model`, `mode`, `device`, `threshold`, `ffmpeg`, `ml_load_error`, `version`.
The model loads in a background thread, so the server answers immediately and
reports `"loading"` while a first-run download is in flight.

`GET /api/samples` → the contents of `sample_audio/`.

Config, all env-overridable: `VG_DETECTOR_MODE`, `VG_MODEL_ID`, `VG_THRESHOLD`,
`VG_WINDOW_SEC`, `VG_MIN_DURATION_SEC`, `VG_MAX_DURATION_SEC`,
`VG_MAX_UPLOAD_MB`, `VG_LOCAL_MODEL_DIR`, `VG_LOG_LEVEL`, `HOST`, `PORT`.

---

## 6. Verified on this machine

| Check (spec §14.6) | Result |
|---|---|
| Core install in a fresh venv, no ML deps | 6 wheels, no build step |
| Server starts with no traceback | pass |
| `/api/health` → `backend: "heuristic"` without ML deps | pass |
| Uploading a sample returns a valid verdict + confidence | pass, ~25-100 ms/clip |
| Same package with ML deps → `backend: "ml"` | pass, ~60-280 ms/clip on CPU |
| Corrupt upload → clean `400`, server survives | pass |
| Silent / too-short / non-speech input | `400` for silent & too-short; verdict + explicit warning for non-speech |
| Spectrogram PNG is valid | pass (900×256 RGB, CRCs verified) |
| `docker compose up` serves the same app | see the note below |

**Docker was not executed here** — no Docker daemon on the build machine. The
`Dockerfile` and `docker-compose.yml` are written and reviewed (python:3.11-slim,
`libsndfile1` + `ffmpeg`, non-root user, healthcheck, `WITH_ML` build arg, HF
cache volume) but the image has not been built. Run `docker compose up` once on a
machine that has Docker before relying on Option B in front of judges.

`run.bat` was likewise not executed (no Windows machine); it mirrors `run.sh`
step for step in plain `cmd` syntax.

One machine-specific note: the Homebrew Python 3.14 on the build machine has a
broken `pyexpat`, which makes `python3 -m venv` fail inside `ensurepip`. That is
a local install problem, not a project one — the venv used for testing was
created with `uv` on Python 3.12. `RUNME.md` lists the fix
(`brew reinstall python@3.12 expat`, or use Docker).

---

## 7. Next steps, in the order worth doing them

1. **Record the demo pair.** A teammate's real voice plus a clone of that same
   voice from a real cloning tool, in `sample_audio/`. This is both the demo that
   wins the room and the only way to know whether the shipped model actually
   catches modern cloning — the numbers in §2 are against system TTS.
2. **Run `train.py` on ASVspoof 2019 LA** with `--augment` and quote the eval
   EER it prints. The checkpoint lands in `backend/models/finetuned/` and
   `detector.py` picks it up automatically on the next start, with no internet.
3. **Refit the heuristic** on those real clips (`calibrate_heuristic.py`) so the
   fallback stops collapsing on phone-quality audio.
4. **Then** the roadmap in `README.md` §12 / blueprint §6-7: WebSocket streaming,
   the prevention layer (alert / block / challenge-response), speaker
   verification, spectrogram explainability.
