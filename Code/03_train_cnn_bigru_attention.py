# 03_train_cnn_bigru_attention.py
# Train CNN -> BiGRU -> Attention on Log-Mel segments saved by 02_make_segments_and_mels.py
# Saves best checkpoint + label map + config to: ...\processed\best_model.pt

import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Paths (YOUR EXACT PATH)
# =========================
BASE = r"C:\Users\tosaw\Music\PQ11\ICBHI_final_database"
PROC = os.path.join(BASE, "processed")
MANIFEST = os.path.join(PROC, "segments_manifest.csv")

# =========================
# Train config
# =========================
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
NUM_CLASSES = 4
SEED = 42

LABEL_MAP = {0: "normal", 1: "crackles", 2: "wheezes", 3: "both"}

# =========================
# Reproducibility
# =========================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic (a bit slower but stable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Dataset
# =========================
class SegDataset(Dataset):
    """
    Expects manifest columns:
      - feature_path : path to .npy log-mel (n_mels, time)
      - label        : int in [0..3]
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fp = row["feature_path"]
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing feature file: {fp}")

        x = np.load(fp)  # (n_mels, time)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D mel array (n_mels,time). Got shape={x.shape} at {fp}")

        # Optional: normalize per-sample (helps stability)
        x = (x - x.mean()) / (x.std() + 1e-6)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, time)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y

# =========================
# Model blocks
# =========================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        w = self.attn(x).squeeze(-1)      # (B, T)
        w = torch.softmax(w, dim=1)       # (B, T)
        out = torch.sum(x * w.unsqueeze(-1), dim=1)  # (B, H)
        return out

class CNN_BiGRU_Attn(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.20),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.20),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.30),
        )

        # BiGRU + attention
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
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, n_mels, time)
        f = self.cnn(x)           # (B, 64, m', t')
        f = f.mean(dim=2)         # (B, 64, t')  mean over mel axis -> keep time
        f = f.transpose(1, 2)     # (B, t', 64)  sequence for GRU

        out, _ = self.gru(f)      # (B, t', 256)
        pooled = self.attn(out)   # (B, 256)
        logits = self.fc(pooled)  # (B, classes)
        return logits

# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_y, all_p = [], []

    for x, y in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        all_y.extend(y.detach().cpu().numpy().tolist())
        all_p.extend(pred.detach().cpu().numpy().tolist())

    return total_loss / max(total, 1), correct / max(total, 1), np.array(all_y), np.array(all_p)

# =========================
# Main
# =========================
def main():
    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(
            f"Missing manifest: {MANIFEST}\n"
            f"Run 02_make_segments_and_mels.py first."
        )

    os.makedirs(PROC, exist_ok=True)

    df = pd.read_csv(MANIFEST)

    required_cols = {"feature_path", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}. Found: {list(df.columns)}")

    # ensure labels are ints
    df["label"] = df["label"].astype(int)

    # sanity check label range
    bad = df[(df["label"] < 0) | (df["label"] >= NUM_CLASSES)]
    if len(bad) > 0:
        raise ValueError(f"Found labels outside [0..{NUM_CLASSES-1}] in manifest. Example rows:\n{bad.head()}")

    # Stratified split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )

    train_ds = SegDataset(train_df)
    val_ds = SegDataset(val_df)

    # windows friendly: num_workers=0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = CNN_BiGRU_Attn(num_classes=NUM_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_path = os.path.join(PROC, "best_model.pt")
    train_log_path = os.path.join(PROC, "train_log.csv")

    print("Device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    history = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn)
        va_loss, va_acc, _, _ = eval_one_epoch(model, val_loader, loss_fn)

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        })

        # Save best checkpoint (WITH metadata so later inference is easy)
        if va_acc > best_val:
            best_val = va_acc
            ckpt = {
                "model_state": model.state_dict(),
                "label_map": LABEL_MAP,
                "num_classes": NUM_CLASSES,
                "seed": SEED,
            }
            torch.save(ckpt, best_path)
            print("✅ Saved best:", best_path)

        pd.DataFrame(history).to_csv(train_log_path, index=False)

    print("\n=== Final Validation Report (best checkpoint) ===")
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    _, _, y_true, y_pred = eval_one_epoch(model, val_loader, loss_fn)

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4, target_names=[LABEL_MAP[i] for i in range(NUM_CLASSES)]))

    # Also save a small json for quick reference
    meta_path = os.path.join(PROC, "best_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_path, "label_map": LABEL_MAP, "device": DEVICE}, f, indent=2)
    print("✅ Saved meta:", meta_path)

if __name__ == "__main__":
    main()
