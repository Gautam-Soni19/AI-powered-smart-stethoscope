# 04_predict_patient.py
# Predict respiratory class from a single .wav using the trained CNN+BiGRU+Attention model.
# Works with checkpoints saved as:
#   torch.save({"model_state": model.state_dict(), "label_map": {...}, "num_classes": N, ...}, path)
# or plain state_dict

import os
import json
import argparse
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn


# =========================
# PATHS (your exact base)
# =========================
BASE = r"C:\Users\tosaw\Music\PQ11\ICBHI_final_database"
PROC = os.path.join(BASE, "processed")

# This should match your training output path
MODEL_PATH = os.path.join(PROC, "best_model.pt")

# Optional meta json (if you saved it)
META_PATH = os.path.join(PROC, "best_model_meta.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# MODEL DEFINITIONS (must match 03 exactly)
# =========================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        w = self.attn(x).squeeze(-1)   # (B, T)
        w = torch.softmax(w, dim=1)    # (B, T)
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
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
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
        f = self.cnn(x)  # (B, 64, m', t')
        f = f.mean(dim=2)       # (B, 64, t')
        f = f.transpose(1, 2)   # (B, t', 64)
        out, _ = self.gru(f)    # (B, t', 256)
        pooled = self.attn(out) # (B, 256)
        logits = self.fc(pooled)
        return logits


# =========================
# CHECKPOINT LOADING (FIXES YOUR ERROR)
# =========================
def _is_intlike(x):
    try:
        int(x)
        return True
    except:
        return False


def load_checkpoint():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    # If it's the "bundle" checkpoint
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model_state = ckpt["model_state"]
        label_map_raw = ckpt.get("label_map", None)
        num_classes = ckpt.get("num_classes", None)
    else:
        # plain state_dict
        model_state = ckpt
        label_map_raw = None
        num_classes = None

    # If no label_map provided, default (same ordering we used everywhere)
    if label_map_raw is None:
        id2name = {0: "normal", 1: "crackles", 2: "wheezes", 3: "both"}
        name2id = {v: k for k, v in id2name.items()}
        num_classes = 4
        return model_state, id2name, name2id, num_classes

    # Normalize label_map into:
    #   id2name: {int: str}
    #   name2id: {str: int}
    keys = list(label_map_raw.keys())
    vals = list(label_map_raw.values())

    # Case: id->name (keys are ints OR string ints; values are class names)
    if all(_is_intlike(k) for k in keys) and all(isinstance(v, str) for v in vals):
        id2name = {int(k): v for k, v in label_map_raw.items()}
        name2id = {v: k for k, v in id2name.items()}

    # Case: name->id (keys are names; values are ints or string ints)
    elif all(isinstance(k, str) for k in keys) and all(_is_intlike(v) for v in vals):
        name2id = {k: int(v) for k, v in label_map_raw.items()}
        id2name = {v: k for k, v in name2id.items()}

    else:
        # Last resort: try to infer
        id2name = {}
        for k, v in label_map_raw.items():
            if _is_intlike(k) and isinstance(v, str):
                id2name[int(k)] = v
            elif isinstance(k, str) and _is_intlike(v):
                id2name[int(v)] = k

        if not id2name:
            raise ValueError(f"Unrecognized label_map format: {label_map_raw}")

        name2id = {v: k for k, v in id2name.items()}

    if num_classes is None:
        num_classes = max(id2name.keys()) + 1
    else:
        num_classes = int(num_classes)

    return model_state, id2name, name2id, num_classes


# =========================
# FEATURE EXTRACTION
# =========================
def _stft_logmel(y, sr=16000, n_fft=1024, hop_length=256, n_mels=64):
    """
    Pure-torch-ish log-mel without librosa dependency.
    But we will keep it simple using numpy + torch operations.
    """
    # Simple magnitude spectrogram (numpy)
    # Pad to at least one frame
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))

    # STFT
    window = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(y) - n_fft) // hop_length
    if frames <= 0:
        frames = 1

    spec = []
    for i in range(frames):
        start = i * hop_length
        frame = y[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frame = frame * window
        fft = np.fft.rfft(frame)
        mag = np.abs(fft).astype(np.float32)
        spec.append(mag)

    spec = np.stack(spec, axis=1)  # (freq_bins, time)

    # Build mel filterbank (simple implementation)
    # freq bins for rfft
    freq_bins = spec.shape[0]
    f_min, f_max = 0.0, sr / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, freq_bins - 1)

    fb = np.zeros((n_mels, freq_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
        if f_center == f_left:
            f_center += 1
        if f_right == f_center:
            f_right += 1
        f_right = min(f_right, freq_bins - 1)

        for k in range(f_left, f_center):
            fb[m - 1, k] = (k - f_left) / (f_center - f_left + 1e-9)
        for k in range(f_center, f_right):
            fb[m - 1, k] = (f_right - k) / (f_right - f_center + 1e-9)

    mel_spec = fb @ spec  # (n_mels, time)
    mel_spec = np.maximum(mel_spec, 1e-10)
    logmel = np.log(mel_spec).astype(np.float32)
    return logmel


def _pad_or_trim_time(x, target_T=300):
    """
    x: (n_mels, T)
    """
    n_mels, T = x.shape
    if T == target_T:
        return x
    if T > target_T:
        return x[:, :target_T]
    pad = target_T - T
    return np.pad(x, ((0, 0), (0, pad)), mode="constant")


# =========================
# PREDICTION
# =========================
@torch.no_grad()
def predict_file(wav_path, sr_target=16000, target_T=300):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Missing wav: {wav_path}")

    # Load checkpoint + label map safely
    model_state, id2name, name2id, num_classes = load_checkpoint()

    # Build model with correct num classes
    model = CNN_BiGRU_Attn(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    # Load audio
    y, sr = sf.read(wav_path, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample if needed (simple linear resample)
    if sr != sr_target:
        # linear interpolation resample (good enough for inference)
        x_old = np.linspace(0, 1, num=len(y), endpoint=False)
        x_new = np.linspace(0, 1, num=int(len(y) * sr_target / sr), endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        sr = sr_target

    # Extract log-mel
    feat = _stft_logmel(y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
    feat = _pad_or_trim_time(feat, target_T=target_T)

    # To tensor => (1,1,n_mels,T)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_name = id2name.get(pred_id, f"class_{pred_id}")

    out = {
        "wav": wav_path,
        "device": DEVICE,
        "num_classes": num_classes,
        "prediction_id": pred_id,
        "prediction_name": pred_name,
        "probabilities": {id2name.get(i, f"class_{i}"): float(probs[i]) for i in range(num_classes)}
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Full path to .wav file")
    ap.add_argument("--save_json", default="", help="Optional: path to save prediction json")
    args = ap.parse_args()

    out = predict_file(args.wav)

    print("\n=== Patient Prediction ===")
    print("WAV:", out["wav"])
    print("Device:", out["device"])
    print("Predicted:", out["prediction_name"], f"(id={out['prediction_id']})")
    print("\nProbabilities:")
    for k, v in sorted(out["probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {k:10s}: {v:.4f}")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("\n✅ Saved:", args.save_json)


if __name__ == "__main__":
    main()
