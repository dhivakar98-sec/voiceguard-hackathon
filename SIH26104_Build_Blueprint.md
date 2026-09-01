# SIH26104 — Voice Cloning Impersonation Detection
### End-to-End Build Blueprint

**Problem (restated):** Detect, in real time, when a voice is AI-generated/cloned (rather than a genuine human speaker) and **prevent** the impersonation from succeeding — by flagging, blocking, or alerting inside a voice channel (calls, voice notes, voice-based authentication).

This is fundamentally an **audio anti-spoofing / synthetic-speech detection** problem, with a real-time layer and a prevention layer bolted on top. The research is mature; your job is assembly, real-time engineering, and a differentiator.

---

## 1. The winning strategy (read this first)

Four things decide SIH outcomes: **a demo that visibly works, scope you finish, a problem judges get, and a differentiator.** Optimize for those, not for model novelty.

**Golden rules**
1. **Do not train from scratch.** Fine-tune a pretrained self-supervised front-end (wav2vec2 / XLS-R) on ASVspoof. From-zero training will eat your whole hackathon.
2. **Get a bad end-to-end pipeline working on Day 1**, then improve the model. A full loop (audio in → score out → action) beats a great model with no demo.
3. **Detection alone is a project; detection + real-time prevention is a *winning* project.** Judges must *see* an impersonation get blocked live.
4. **Pick a differentiator early** (below) and build the demo around it.

**Differentiators that win this PS**
- **Indian-language + phone-quality robustness.** Public models are trained on clean English. Show it working on Hindi/Tamil/etc. over a compressed/noisy channel.
- **Explainability.** Don't just say "FAKE 92%" — highlight the spectrogram regions or artifacts that triggered it.
- **True real-time streaming** (sliding-window inference, <1s latency), not file upload.
- **Live prevention action** — call interruption, challenge-response, or a "verified human" badge.

---

## 2. System architecture

```
                        ┌─────────────────────────────────────────────┐
   Audio source         │              PROCESSING PIPELINE             │
 (mic / call stream /   │                                             │
  uploaded clip)        │  1. Capture & buffer (WebSocket / stream)   │
        │               │  2. Preprocess: resample 16kHz, VAD,        │
        ▼               │     1–2s sliding windows                    │
  ┌───────────┐         │  3. Feature extraction:                     │
  │  Frontend  │◀──────▶│     wav2vec2/XLS-R embeddings               │
  │ (React /   │  WS    │  4. Detector model → spoof score [0..1]     │
  │  simple    │        │  5. (optional) Speaker verification         │
  │  web UI)   │        │     (ECAPA-TDNN) vs claimed identity        │
  └───────────┘         │  6. Decision engine (thresholds + smoothing)│
        │               │  7. PREVENTION: flag / block / alert / log  │
        │               └─────────────────────────────────────────────┘
        ▼
  Dashboard: live score, spectrogram, verdict, event log, alerts
```

**Two model tracks (build Track A first):**
- **Track A — Anti-spoofing (core, required):** "Is this speech real or synthetic?" Output: bonafide/spoof score.
- **Track B — Speaker verification (bonus):** "Is this actually the person it claims to be?" Catches replay/impersonation even of real audio. Adds depth if you have time.

---

## 3. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Ecosystem for audio ML |
| ML | PyTorch + torchaudio | Standard, GPU-ready |
| Models | HuggingFace Transformers, SpeechBrain | Pretrained wav2vec2, ECAPA-TDNN |
| Audio | librosa, soundfile, webrtcvad | Loading, features, voice activity |
| Backend | FastAPI + WebSockets | Async, easy streaming API |
| Frontend | React (or plain HTML+JS for speed) | Live dashboard |
| Storage | SQLite (demo) / Postgres | Event/alert log |
| Deploy | Docker + one GPU (Colab/Kaggle for training) | Reproducible; free GPU for training |

Keep the frontend simple if your team isn't strong on it — a single-page dashboard with a live meter, spectrogram, and log is enough.

---

## 4. Data

Use public anti-spoofing datasets. **You do not need to generate fakes yourself** (though generating a few Indian-language clones with an open TTS tool makes a killer demo).

| Dataset | Use | Notes |
|---|---|---|
| **ASVspoof 2019 LA** | Primary train/eval | Standard benchmark, bonafide + many TTS/VC attacks |
| **ASVspoof 2021 DF** | Robustness eval | Compressed/codec audio → closer to real calls |
| **In-the-Wild** | Generalization test | Real-world fakes; tests if you overfit to ASVspoof |
| **WaveFake** | Extra fake variety | Multiple vocoders |
| **Your own clones** | Demo | Clone a teammate's voice with an open TTS/VC tool for the live demo |

**Demo gold:** record a teammate, clone their voice with a free tool, and show your system flag the clone while passing the real voice. That single side-by-side wins rooms.

---

## 5. Model approach

**Baseline (get this working first — ~half a day):**
`wav2vec2/XLS-R` frozen front-end → mean-pool embeddings → small MLP classifier (spoof vs bonafide). Robust, fast to train, generalizes better than raw spectrogram CNNs because the front-end is pretrained on huge speech data.

**Upgrade (if time):**
- Fine-tune the wav2vec2 layers (unfreeze last few) for a big accuracy jump.
- Or plug in **AASIST** / **RawNet2** back-ends (SOTA anti-spoofing architectures) on top of the SSL front-end.
- Add **score smoothing** over the sliding window so a single noisy frame doesn't cause flip-flopping.

**Metric to report:** EER (Equal Error Rate) — the standard for this task — plus accuracy and a confusion matrix. Judges like a clear number ("2.3% EER on ASVspoof LA").

---

## 6. Real-time layer

The demo differentiator. Approach:
- Stream audio from the browser over a **WebSocket** in ~1s chunks.
- Maintain a rolling buffer; run inference on **overlapping 1.5–2s windows** (sliding window, e.g. 50% overlap).
- **Smooth** the per-window scores (moving average / exponential) → a stable live verdict.
- Target **< 1s** perceived latency. On CPU it's feasible with the MLP-on-frozen-embeddings baseline; use GPU if available.

---

## 7. Prevention layer (don't skip — it's in the PS title)

Detection produces a score; **prevention is the action.** Implement at least two:
- **Real-time flag/alert:** banner + sound the moment score crosses threshold ("⚠ Synthetic voice detected").
- **Block / interrupt:** in a mock call UI, drop or mute the caller and log the event.
- **Challenge–response:** on suspicion, prompt "say this random phrase" — cloned/TTS pipelines struggle with on-the-fly novel prompts.
- **Verified-human badge + audit log:** every decision logged with timestamp, score, and clip for forensics.

Frame it as an **API/middleware** that any voice app (banking IVR, call center, meeting app) could drop in. That framing scores on "impact/scalability."

---

## 8. Phased plan & timeline

Scale to your actual hackathon length; this assumes ~36 hours.

| Phase | Goal | Deliverable |
|---|---|---|
| **0. Setup (2h)** | Repo, env, data downloaded, roles assigned | Everyone can run `hello world` inference |
| **1. Offline detector (8h)** | wav2vec2 + MLP trained on ASVspoof; scores a file | `predict(file) → score` + EER number |
| **2. Real-time API (8h)** | FastAPI + WebSocket streaming inference | Live mic → live score in terminal/UI |
| **3. Prevention + dashboard (8h)** | Decision engine, alerts, block action, live spectrogram | Working end-to-end demo |
| **4. Differentiator + polish (6h)** | Indian-lang/noisy robustness OR explainability; edge cases | The "wow" moment |
| **5. Demo + PPT (4h)** | Rehearse script, prep slides, failure fallbacks | Rock-solid 5-min demo |

**Critical path:** Phase 1 must finish on time. If the model lags, ship the baseline and move on — a working pipeline with a mediocre model beats a great model with no UI.

---

## 9. Team split (6 members)

| Role | Owns |
|---|---|
| **ML lead** | Model, training, EER, thresholds |
| **Data engineer** | Datasets, preprocessing, VAD, demo clones |
| **Backend** | FastAPI, WebSocket streaming, decision engine, DB |
| **Frontend** | Dashboard, live meter, spectrogram, alerts |
| **Integration/demo** | Wire it together, real-time tuning, demo script, fallbacks |
| **Docs/PPT/pitch** | Slides, architecture diagram, impact story, presenter |

Backend + frontend + integration should agree on the WebSocket message format on Day 1 so they can work in parallel.

---

## 10. Judge demo script (5 minutes)

1. **Hook (30s):** "Voice-clone fraud is rising — CEO-voice scams, fake ransom calls, bypassed voice-auth. Here's a system that catches it live."
2. **Real voice (45s):** teammate speaks → dashboard shows **HUMAN / green**, low spoof score.
3. **The clone (60s):** play the cloned version of the *same* teammate → dashboard flips to **SYNTHETIC / red**, alert fires, call blocked.
4. **Why (45s):** show the spectrogram highlight / explanation of the artifacts. Show the EER number.
5. **Differentiator (60s):** repeat on an Indian language / over a compressed "phone" clip — still works.
6. **Impact (30s):** "Drop-in API for banks, call centers, meeting apps." Show the audit log.
7. **Q&A buffer.** Have a pre-recorded backup video in case live audio fails.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Model overfits ASVspoof, fails on real fakes | Test on In-the-Wild early; keep the SSL front-end (generalizes better) |
| Real-time too slow | Use frozen-embedding baseline; smaller window; GPU; degrade gracefully to near-real-time |
| Live audio fails on stage | Pre-recorded backup demo video; file-upload fallback path |
| Scope creep (Track B, fancy UI) | Track A + prevention + one differentiator is enough; treat rest as bonus |
| Noisy/phone audio breaks it | Train/augment with codec + noise augmentation (this is also your differentiator) |

---

## 12. Starter code

### 12.1 Environment
```bash
python -m venv venv && source venv/bin/activate
pip install torch torchaudio transformers speechbrain \
            librosa soundfile webrtcvad fastapi uvicorn[standard] \
            scikit-learn numpy
```

### 12.2 Baseline detector (wav2vec2 frozen + MLP head)
```python
# detector.py
import torch, torch.nn as nn, torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SSL_NAME = "facebook/wav2vec2-xls-r-300m"  # multilingual -> good for Indian langs

class SpoofDetector(nn.Module):
    def __init__(self, ssl_name=SSL_NAME, freeze=True):
        super().__init__()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(ssl_name)
        self.ssl = Wav2Vec2Model.from_pretrained(ssl_name)
        if freeze:
            for p in self.ssl.parameters():
                p.requires_grad = False
        h = self.ssl.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)            # [bonafide, spoof]
        )

    def forward(self, wav_16k):          # wav_16k: (B, T) float32 @16kHz
        inp = self.fe(list(wav_16k.cpu().numpy()), sampling_rate=16000,
                      return_tensors="pt", padding=True).input_values.to(wav_16k.device)
        emb = self.ssl(inp).last_hidden_state.mean(dim=1)   # mean-pool -> (B, h)
        return self.head(emb)

def load_16k(path):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.mean(0)  # mono (T,)

@torch.no_grad()
def score_file(model, path):
    model.eval()
    wav = load_16k(path).unsqueeze(0).to(DEVICE)
    prob = torch.softmax(model(wav), dim=-1)[0]
    return {"bonafide": prob[0].item(), "spoof": prob[1].item()}
```

### 12.3 Training loop skeleton
```python
# train.py
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from detector import SpoofDetector, DEVICE
# Build a Dataset that yields (waveform_16k, label)  label: 0=bonafide, 1=spoof
# from ASVspoof protocol files. Pad/crop clips to ~4s in a collate_fn.

def train(train_ds, val_ds, epochs=5, lr=1e-3):
    model = SpoofDetector().to(DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss()
    tl = DataLoader(train_ds, batch_size=8, shuffle=True)   # +collate_fn
    for ep in range(epochs):
        model.train(); tot = 0
        for wav, y in tl:
            wav, y = wav.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(wav), y)
            loss.backward(); opt.step(); tot += loss.item()
        print(f"epoch {ep}: loss={tot/len(tl):.4f}")
        # TODO: compute EER on val_ds each epoch
    torch.save(model.state_dict(), "spoof_detector.pt")
    return model
```
> **EER note:** compute Equal Error Rate on the spoof probability with `sklearn` ROC curve — it's the metric judges expect for this task.

### 12.4 Real-time streaming API (FastAPI + WebSocket)
```python
# server.py
import numpy as np, torch, collections
from fastapi import FastAPI, WebSocket
from detector import SpoofDetector, DEVICE

app = FastAPI()
model = SpoofDetector().to(DEVICE)
model.load_state_dict(torch.load("spoof_detector.pt", map_location=DEVICE))
model.eval()

WIN = 16000 * 2        # 2s window
THRESH = 0.5

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    buf = collections.deque(maxlen=WIN)
    scores = collections.deque(maxlen=5)   # smoothing
    try:
        while True:
            chunk = np.frombuffer(await ws.receive_bytes(), dtype=np.float32)
            buf.extend(chunk.tolist())
            if len(buf) >= WIN:
                wav = torch.tensor([list(buf)], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    p_spoof = torch.softmax(model(wav), -1)[0, 1].item()
                scores.append(p_spoof)
                smooth = sum(scores) / len(scores)
                verdict = "SYNTHETIC" if smooth > THRESH else "HUMAN"
                await ws.send_json({"spoof": round(smooth, 3),
                                    "verdict": verdict,
                                    "action": "BLOCK" if verdict=="SYNTHETIC" else "ALLOW"})
    except Exception:
        await ws.close()
# run: uvicorn server:app --reload
```

Frontend: capture mic with the Web Audio API, send Float32 PCM chunks over the WebSocket, and render `spoof`, `verdict`, and a live spectrogram. The `action` field drives your prevention UI (block banner, etc.).

---

## 13. Next builds (ask Claude for any of these)
- The **Dataset/collate class** for ASVspoof protocol files
- The **EER evaluation** script
- The **React dashboard** (live meter + spectrogram + alert + log)
- **Speaker verification (Track B)** with SpeechBrain ECAPA-TDNN
- **Noise/codec augmentation** for the phone-audio differentiator
- The **pitch deck** outline
