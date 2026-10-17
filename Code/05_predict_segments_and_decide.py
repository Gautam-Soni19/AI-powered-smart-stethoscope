# 05_predict_segments_and_decide.py
# Segment-wise prediction + final decision (majority/weighted rules) for ICBHI classes:
# 0 normal, 1 crackles, 2 wheezes, 3 both

import os
import json
import argparse
import numpy as np

import librosa

import torch
import torch.nn as nn

# =========================
# Paths (your setup)
# =========================
BASE = r"C:\Users\tosaw\Music\PQ11\ICBHI_final_database"
PROC = os.path.join(BASE, "processed")

MODEL_PATH = os.path.join(PROC, "best_model.pt")
META_PATH  = os.path.join(PROC, "best_model_meta.json")

# =========================
# Audio / Feature settings
# (must match what you used in 02)
# =========================
TARGET_SR = 4000          # typical for ICBHI pipelines (often 4k)
N_MELS    = 64
N_FFT     = 1024
HOP_LEN   = 256
WIN_LEN   = 1024
FMIN      = 50
FMAX      = 2000

SEG_SECONDS = 5.0         # segment length in seconds
SEG_HOP_SEC = 2.5         # hop between segments (overlap)
MAX_AUDIO_SECONDS = None  # set e.g. 60 to cut very long files (optional)

EPS = 1e-8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Model (must match 03)
# =========================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        w = self.attn(x).squeeze(-1)    # (B, T)
        w = torch.softmax(w, dim=1)     # (B, T)
        out = torch.sum(x * w.unsqueeze(-1), dim=1)  # (B, H)
        return out

class CNN_BiGRU_Attn(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.3),
        )

        self.gru = nn.GRU(
            input_size=64, hidden_size=128, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.attn = AttentionPooling(hidden_dim=256)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, n_mels, time)
        f = self.cnn(x)            # (B, 64, m', t')
        f = f.mean(dim=2)          # (B, 64, t')
        f = f.transpose(1, 2)      # (B, t', 64)
        out, _ = self.gru(f)       # (B, t', 256)
        pooled = self.attn(out)    # (B, 256)
        logits = self.fc(pooled)   # (B, classes)
        return logits

# =========================
# Utilities
# =========================
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_id2name(meta: dict, num_classes_fallback: int = 4):
    """
    Supports BOTH formats saved in meta:
    A) name -> id   {"normal": 0, "crackles": 1, ...}
    B) id -> name   {"0": "normal", "1": "crackles", ...}  (your case)
    Returns: (id2name_dict, num_classes_int)
    """
    label_map = meta.get("label_map", None)

    # fallback
    if not isinstance(label_map, dict) or len(label_map) == 0:
        id2name = {0: "normal", 1: "crackles", 2: "wheezes", 3: "both"}
        num_classes = int(meta.get("num_classes", num_classes_fallback))
        return id2name, num_classes

    sample_k, sample_v = next(iter(label_map.items()))
    id2name = {}

    # Case B: id -> name
    if str(sample_k).isdigit() and isinstance(sample_v, str):
        for k, v in label_map.items():
            if str(k).isdigit():
                id2name[int(k)] = str(v)

    # Case A: name -> id
    else:
        for k, v in label_map.items():
            try:
                idx = int(v)
            except Exception:
                continue
            id2name[idx] = str(k)

    if "num_classes" in meta:
        num_classes = int(meta["num_classes"])
    else:
        num_classes = max(id2name.keys()) + 1 if len(id2name) else num_classes_fallback

    return id2name, num_classes

def safe_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)

def load_audio_mono(path: str, sr: int):
    y, _sr = librosa.load(path, sr=sr, mono=True)
    if MAX_AUDIO_SECONDS is not None:
        max_len = int(sr * MAX_AUDIO_SECONDS)
        y = y[:max_len]
    return y

def log_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )
    # log-mel
    logmel = librosa.power_to_db(mel + EPS, ref=np.max)
    # normalize to roughly [0, 1] for stability (same idea as common pipelines)
    # NOTE: if your 02 script used a different normalization, update this to match it.
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)

def pad_or_trim_time(x: np.ndarray, target_T: int) -> np.ndarray:
    # x: (n_mels, T)
    T = x.shape[1]
    if T == target_T:
        return x
    if T > target_T:
        return x[:, :target_T]
    # pad
    pad = target_T - T
    return np.pad(x, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

def slice_segments(y: np.ndarray, sr: int, seg_sec: float, hop_sec: float):
    seg_len = int(sr * seg_sec)
    hop_len = int(sr * hop_sec)

    if len(y) < seg_len:
        return [(0, len(y), y)]

    segments = []
    start = 0
    while start < len(y):
        end = start + seg_len
        seg = y[start:end]
        if len(seg) == 0:
            break
        segments.append((start, min(end, len(y)), seg))
        if end >= len(y):
            break
        start += hop_len

    return segments

def expected_frames_for_segment(sr: int, seg_sec: float) -> int:
    # Rough estimate of frames for a fixed-length segment using hop_length
    seg_len = int(sr * seg_sec)
    # librosa melspectrogram gives frames approx = 1 + floor((len - n_fft)/hop)
    # but due to padding/center behaviour it can differ. We'll compute target from a dummy.
    dummy = np.zeros(seg_len, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=dummy,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )
    return mel.shape[1]

def load_model_and_meta():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing meta: {META_PATH}")

    meta = load_json(META_PATH)
    id2name, num_classes = build_id2name(meta, num_classes_fallback=4)

    model = CNN_BiGRU_Attn(num_classes=num_classes).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    # Support both ways of saving:
    # - state_dict directly
    # - dict with 'model_state'
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, meta, id2name, num_classes

@torch.no_grad()
def predict_segment(model, seg_logmel: np.ndarray):
    # seg_logmel: (n_mels, T)
    x = torch.tensor(seg_logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,mels,T)
    x = x.to(DEVICE)
    logits = model(x)                    # (1, C)
    probs = safe_softmax(logits)[0]      # (C,)
    pred_id = int(torch.argmax(probs).item())
    return pred_id, probs.detach().cpu().numpy()

def decide_from_segment_preds(pred_ids, probs_list, num_classes, id2name):
    """
    Final patient-level decision.
    Practical rule for clinical-ish behavior:
      - If "both" appears with meaningful share => both
      - Else if wheezes share > crackles share => wheezes
      - Else if crackles share > 0 => crackles
      - Else normal
    Additionally, use average probability as tie-breaker.
    """
    pred_ids = np.array(pred_ids, dtype=int)
    counts = np.bincount(pred_ids, minlength=num_classes)

    # mean probs per class across segments
    probs_arr = np.vstack(probs_list) if len(probs_list) else np.zeros((0, num_classes), dtype=np.float32)
    mean_probs = probs_arr.mean(axis=0) if len(probs_list) else np.zeros((num_classes,), dtype=np.float32)

    total = counts.sum() if counts.sum() > 0 else 1
    share = counts / total

    # IDs by name if available
    name2id = {v: k for k, v in id2name.items()}
    id_normal = name2id.get("normal", 0)
    id_crack  = name2id.get("crackles", 1)
    id_wheeze = name2id.get("wheezes", 2)
    id_both   = name2id.get("both", 3)

    # decision logic
    # 1) strong "both"
    if share[id_both] >= 0.25 or (counts[id_both] >= 2 and mean_probs[id_both] >= 0.35):
        final_id = id_both
        reason = f"'both' detected in {counts[id_both]}/{total} segments (share={share[id_both]:.2f})."
        return final_id, reason, counts, share, mean_probs

    # 2) wheezes vs crackles (dominant)
    if counts[id_wheeze] > counts[id_crack] and counts[id_wheeze] >= 1:
        final_id = id_wheeze
        reason = f"More wheezes segments than crackles ({counts[id_wheeze]} vs {counts[id_crack]})."
        return final_id, reason, counts, share, mean_probs

    if counts[id_crack] > 0:
        final_id = id_crack
        reason = f"Crackles detected in {counts[id_crack]}/{total} segments."
        return final_id, reason, counts, share, mean_probs

    # 3) default normal
    final_id = id_normal
    reason = "No consistent abnormal segments detected; defaulting to normal."
    return final_id, reason, counts, share, mean_probs

@torch.no_grad()
def predict_file(wav_path: str):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    model, meta, id2name, num_classes = load_model_and_meta()

    y = load_audio_mono(wav_path, sr=TARGET_SR)
    segments = slice_segments(y, TARGET_SR, SEG_SECONDS, SEG_HOP_SEC)

    target_T = expected_frames_for_segment(TARGET_SR, SEG_SECONDS)

    seg_results = []
    pred_ids = []
    probs_list = []

    for i, (s, e, seg) in enumerate(segments):
        lm = log_mel_spectrogram(seg, TARGET_SR)              # (mels, Tvar)
        lm = pad_or_trim_time(lm, target_T)                   # (mels, Tfixed)

        pred_id, probs = predict_segment(model, lm)

        pred_ids.append(pred_id)
        probs_list.append(probs)

        seg_results.append({
            "segment_index": i,
            "start_sec": float(s / TARGET_SR),
            "end_sec": float(e / TARGET_SR),
            "pred_id": int(pred_id),
            "pred_name": id2name.get(pred_id, f"id_{pred_id}"),
            "probs": {id2name.get(j, str(j)): float(probs[j]) for j in range(num_classes)}
        })

    final_id, reason, counts, share, mean_probs = decide_from_segment_preds(
        pred_ids, probs_list, num_classes, id2name
    )

    out = {
        "wav": wav_path,
        "device": DEVICE,
        "segment_seconds": SEG_SECONDS,
        "segment_hop_seconds": SEG_HOP_SEC,
        "num_segments": len(seg_results),
        "final_pred_id": int(final_id),
        "final_pred_name": id2name.get(final_id, f"id_{final_id}"),
        "reason": reason,
        "segment_pred_counts": {id2name.get(i, str(i)): int(counts[i]) for i in range(num_classes)},
        "segment_pred_share": {id2name.get(i, str(i)): float(share[i]) for i in range(num_classes)},
        "mean_prob_per_class": {id2name.get(i, str(i)): float(mean_probs[i]) for i in range(num_classes)},
        "segments": seg_results,
    }
    return out

def print_human_readable(out: dict):
    print("\n=== Patient Decision (Segment-wise) ===")
    print("WAV:", out["wav"])
    print("Device:", out["device"])
    print(f"Segments: {out['num_segments']}  (len={out['segment_seconds']}s, hop={out['segment_hop_seconds']}s)")
    print("\nFinal Predicted:", f"{out['final_pred_name']} (id={out['final_pred_id']})")
    print("Reason:", out["reason"])

    print("\nSegment counts:")
    for k, v in out["segment_pred_counts"].items():
        print(f"  {k:9s}: {v}")

    print("\nMean probability per class:")
    for k, v in out["mean_prob_per_class"].items():
        print(f"  {k:9s}: {v:.4f}")

    print("\n--- Per-segment predictions ---")
    for seg in out["segments"]:
        s = seg["start_sec"]
        e = seg["end_sec"]
        name = seg["pred_name"]
        pid = seg["pred_id"]
        print(f"[{seg['segment_index']:02d}] {s:7.2f}s -> {e:7.2f}s | {name} (id={pid})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to .wav file")
    parser.add_argument("--save_json", action="store_true", help="Save full output json into processed/")
    args = parser.parse_args()

    out = predict_file(args.wav)
    print_human_readable(out)

    if args.save_json:
        os.makedirs(PROC, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.wav))[0]
        save_path = os.path.join(PROC, f"{base}_segment_decision.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("\n✅ Saved:", save_path)

if __name__ == "__main__":
    main()
