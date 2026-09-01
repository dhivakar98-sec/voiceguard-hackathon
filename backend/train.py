"""
OPTIONAL accuracy boost: fine-tune the detector on ASVspoof 2019 LA.

This is Step B from the build spec. The app does NOT need it — it ships working
on the pretrained Hub model (and on the numpy heuristic if torch is absent).
Run this when you want a checkpoint tuned for your own operating conditions.

    pip install -r backend/requirements-ml.txt

    python backend/train.py \
        --data-root /path/to/LA \
        --epochs 3 --batch-size 8 --augment

Expected layout (the standard ASVspoof 2019 LA release):

    LA/
      ASVspoof2019_LA_train/flac/*.flac
      ASVspoof2019_LA_dev/flac/*.flac
      ASVspoof2019_LA_eval/flac/*.flac
      ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019.LA.cm.eval.trl.txt

Protocol lines look like:  LA_0079 LA_T_1138215 - - bonafide

What it does
------------
* wav2vec2 SSL front-end + a small classification head (the head trains first;
  --unfreeze-last N also fine-tunes the top N encoder layers for a big jump)
* 4-second random crops, 16 kHz mono
* --augment adds telephone band-pass (300-3400 Hz), additive noise and random
  gain. This is what keeps accuracy up on real phone audio and is the
  differentiator worth demoing.
* reports EER + accuracy on dev each epoch and on eval at the end
* saves the best checkpoint to backend/models/finetuned/, which detector.py
  loads automatically on the next server start (no code change, no internet)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import preprocessing as P  # noqa: E402

BONAFIDE, SPOOF = 0, 1
CROP_SEC = 4.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Item:
    path: Path
    label: int


def read_protocol(protocol: Path, audio_dir: Path) -> List[Item]:
    """Parse an ASVspoof CM protocol file into (path, label) items."""
    if not protocol.is_file():
        raise SystemExit(f"Protocol file not found: {protocol}")
    items: List[Item] = []
    missing = 0
    for line in protocol.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        utt_id, key = parts[1], parts[-1].lower()
        label = BONAFIDE if key == "bonafide" else SPOOF
        path = audio_dir / f"{utt_id}.flac"
        if not path.is_file():
            path = audio_dir / f"{utt_id}.wav"
        if not path.is_file():
            missing += 1
            continue
        items.append(Item(path, label))
    if not items:
        raise SystemExit(f"No audio found for {protocol} in {audio_dir}")
    if missing:
        print(f"  warning: {missing} utterances listed in the protocol are missing on disk")
    n_spoof = sum(1 for i in items if i.label == SPOOF)
    print(f"  {protocol.name}: {len(items)} clips ({len(items) - n_spoof} bonafide / {n_spoof} spoof)")
    return items


def telephone_band(wav: np.ndarray, sr: int = config.TARGET_SR) -> np.ndarray:
    spec = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(wav.size, d=1.0 / sr)
    spec[(freqs < 300) | (freqs > 3400)] = 0
    return np.fft.irfft(spec, n=wav.size).astype(np.float32)


def augment(wav: np.ndarray, rng: random.Random) -> np.ndarray:
    """Codec-ish / channel-ish augmentation, applied with probability."""
    if rng.random() < 0.35:
        wav = telephone_band(wav)
    if rng.random() < 0.35:
        snr = rng.uniform(10.0, 30.0)
        noise = np.random.randn(wav.size).astype(np.float32)
        sig_p = float(np.mean(wav**2)) + 1e-12
        noise *= np.sqrt(sig_p / (10 ** (snr / 10.0)) / (float(np.mean(noise**2)) + 1e-12))
        wav = wav + noise
    if rng.random() < 0.5:
        wav = wav * rng.uniform(0.3, 1.0)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


class ASVspoofDataset:
    """Yields fixed-length 16 kHz crops. Plain class — torch Dataset protocol."""

    def __init__(self, items: List[Item], train: bool, do_augment: bool, seed: int = 0):
        self.items = items
        self.train = train
        self.do_augment = do_augment
        self.crop = int(config.TARGET_SR * CROP_SEC)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        try:
            raw, sr = P.load_audio(item.path)
            wav = P.resample(P.to_mono(raw), sr, config.TARGET_SR)
        except Exception:
            wav = np.zeros(self.crop, dtype=np.float32)

        if wav.size < self.crop:
            wav = np.pad(wav, (0, self.crop - wav.size))
        elif self.train:
            start = self.rng.randrange(0, wav.size - self.crop + 1)
            wav = wav[start : start + self.crop]
        else:
            wav = wav[: self.crop]

        if self.train and self.do_augment:
            wav = augment(wav, self.rng)
        return wav.astype(np.float32), item.label


def collate(batch):
    import torch

    wavs = np.stack([b[0] for b in batch])
    labels = np.array([b[1] for b in batch], dtype=np.int64)
    return torch.from_numpy(wavs), torch.from_numpy(labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def equal_error_rate(labels: np.ndarray, spoof_scores: np.ndarray) -> float:
    """EER on the spoof score — the metric this task is judged on."""
    order = np.argsort(spoof_scores)
    labels = labels[order]
    n_spoof = max(1, int((labels == SPOOF).sum()))
    n_bona = max(1, int((labels == BONAFIDE).sum()))
    # sweep the threshold from low to high
    fn = np.cumsum(labels == SPOOF) / n_spoof          # spoofs below threshold
    tn = np.cumsum(labels == BONAFIDE) / n_bona
    fp = 1.0 - tn                                      # bonafide above threshold
    idx = int(np.nanargmin(np.abs(fn - fp)))
    return float((fn[idx] + fp[idx]) / 2)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def build_model(base_model: str, unfreeze_last: int):
    import torch
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification

    cfg = AutoConfig.from_pretrained(base_model)
    cfg.num_labels = 2
    cfg.id2label = {BONAFIDE: "bonafide", SPOOF: "spoof"}
    cfg.label2id = {"bonafide": BONAFIDE, "spoof": SPOOF}

    fe = AutoFeatureExtractor.from_pretrained(base_model)
    model = AutoModelForAudioClassification.from_pretrained(
        base_model, config=cfg, ignore_mismatched_sizes=True
    )

    encoder = getattr(model, model.base_model_prefix, None)
    if encoder is not None:
        for p in encoder.parameters():
            p.requires_grad = False
        if unfreeze_last > 0 and hasattr(encoder, "encoder"):
            layers = getattr(encoder.encoder, "layers", getattr(encoder.encoder, "layer", []))
            for layer in list(layers)[-unfreeze_last:]:
                for p in layer.parameters():
                    p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  {base_model}: {trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total params")
    return model, fe


def evaluate(model, fe, loader, device) -> Tuple[float, float]:
    """Accuracy at threshold 0.5 plus EER over the whole loader."""
    import torch

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for wavs, y in loader:
            inputs = fe(
                list(wavs.numpy()), sampling_rate=config.TARGET_SR,
                return_tensors="pt", padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits.float()
            scores.append(torch.softmax(logits, -1)[:, SPOOF].cpu().numpy())
            labels.append(y.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    acc = float(np.mean((scores > 0.5).astype(int) == labels))
    return acc, equal_error_rate(labels, scores)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune VoiceGuard on ASVspoof 2019 LA")
    ap.add_argument("--data-root", required=True, help="path to the LA/ folder")
    ap.add_argument("--base-model", default="facebook/wav2vec2-base",
                    help="SSL front-end (use facebook/wav2vec2-xls-r-300m for multilingual)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze-last", type=int, default=2,
                    help="fine-tune the top N encoder layers (0 = head only)")
    ap.add_argument("--augment", action="store_true", help="band-pass + noise + gain augmentation")
    ap.add_argument("--max-train", type=int, default=0, help="cap training clips (0 = all)")
    ap.add_argument("--max-eval", type=int, default=3000, help="cap dev/eval clips (0 = all)")
    ap.add_argument("--out", default=str(config.MODELS_DIR / "finetuned"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cpu":
        print("  (CPU training is slow — try Colab/Kaggle with a free GPU, or --max-train 20000)")

    root = Path(args.data_root)
    proto = root / "ASVspoof2019_LA_cm_protocols"
    print("\nLoading protocols ...")
    train_items = read_protocol(proto / "ASVspoof2019.LA.cm.train.trn.txt", root / "ASVspoof2019_LA_train" / "flac")
    dev_items = read_protocol(proto / "ASVspoof2019.LA.cm.dev.trl.txt", root / "ASVspoof2019_LA_dev" / "flac")
    eval_items = read_protocol(proto / "ASVspoof2019.LA.cm.eval.trl.txt", root / "ASVspoof2019_LA_eval" / "flac")

    rng = random.Random(1234)
    rng.shuffle(train_items)
    if args.max_train:
        train_items = train_items[: args.max_train]
    if args.max_eval:
        rng.shuffle(dev_items); rng.shuffle(eval_items)
        dev_items, eval_items = dev_items[: args.max_eval], eval_items[: args.max_eval]

    print("\nBuilding model ...")
    model, fe = build_model(args.base_model, args.unfreeze_last)
    model.to(device)

    dl = lambda items, train: DataLoader(  # noqa: E731
        ASVspoofDataset(items, train, args.augment),
        batch_size=args.batch_size, shuffle=train, collate_fn=collate,
        num_workers=args.workers, drop_last=train,
    )
    train_loader, dev_loader, eval_loader = dl(train_items, True), dl(dev_items, False), dl(eval_items, False)

    # class weights: ASVspoof LA is ~9:1 spoof-heavy
    n_spoof = sum(1 for i in train_items if i.label == SPOOF)
    n_bona = len(train_items) - n_spoof
    weights = torch.tensor(
        [len(train_items) / (2 * max(1, n_bona)), len(train_items) / (2 * max(1, n_spoof))],
        dtype=torch.float32, device=device,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader))
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_eer = 1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for step, (wavs, y) in enumerate(train_loader, 1):
            inputs = fe(list(wavs.numpy()), sampling_rate=config.TARGET_SR,
                        return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            loss = loss_fn(model(**inputs).logits.float(), y.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            running += float(loss.item()); seen += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running / seen:.4f} ({(time.time() - t0) / step:.2f}s/step)")

        acc, eer = evaluate(model, fe, dev_loader, device)
        print(f"\nepoch {epoch}: train_loss={running / max(1, seen):.4f}  "
              f"dev_acc={acc * 100:.2f}%  dev_EER={eer * 100:.2f}%")
        if eer < best_eer:
            best_eer = eer
            model.save_pretrained(out_dir)
            fe.save_pretrained(out_dir)
            print(f"  new best — saved to {out_dir}")

    print("\nFinal evaluation on the ASVspoof 2019 LA eval set ...")
    acc, eer = evaluate(model, fe, eval_loader, device)
    print(f"eval_acc={acc * 100:.2f}%  eval_EER={eer * 100:.2f}%")
    print(f"\nCheckpoint: {out_dir}")
    print("Restart the server — detector.py loads backend/models/finetuned/ automatically.")


if __name__ == "__main__":
    main()
