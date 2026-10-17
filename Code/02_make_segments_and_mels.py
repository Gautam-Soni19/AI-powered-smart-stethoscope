# 02_make_segments_and_mels.py
import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

BASE = r"C:\Users\tosaw\Music\PQ11\ICBHI_final_database"
OUT_DIR = os.path.join(BASE, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Audio / feature settings (safe defaults)
SR = 16000
N_MELS = 64
N_FFT = 1024
HOP = 256
WIN = 1024
MAX_SEC = 6.0  # pad/trim each segment to this length

CLASS_MAP = {
    (0, 0): 0,  # Normal
    (1, 0): 1,  # Crackles
    (0, 1): 2,  # Wheezes
    (1, 1): 3,  # Both
}

def find_wav_for_stem(stem: str):
    # typical: same stem wav exists
    cand = os.path.join(BASE, f"{stem}.wav")
    if os.path.exists(cand):
        return cand
    # fallback: search recursively
    hits = glob.glob(os.path.join(BASE, "**", f"{stem}.wav"), recursive=True)
    return hits[0] if hits else None

def pad_or_trim(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

def audio_to_logmel(y, sr):
    # log-mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, win_length=WIN, n_mels=N_MELS, power=2.0
    )
    logS = librosa.power_to_db(S, ref=np.max)
    # normalize per-sample
    logS = (logS - logS.mean()) / (logS.std() + 1e-8)
    return logS.astype(np.float32)  # shape: (n_mels, time)

def main():
    ann_files = sorted(glob.glob(os.path.join(BASE, "*.txt")))
    if not ann_files:
        raise FileNotFoundError(f"No .txt annotation files found in: {BASE}")

    rows = []
    seg_count = 0

    for ann_path in tqdm(ann_files, desc="Building segments"):
        stem = os.path.splitext(os.path.basename(ann_path))[0]
        wav_path = find_wav_for_stem(stem)
        if wav_path is None:
            # skip if wav missing
            continue

        # read full audio once
        y, sr = librosa.load(wav_path, sr=SR, mono=True)

        # annotation format typically:
        # start_time end_time crackles wheezes
        ann = pd.read_csv(ann_path, sep=r"\s+", header=None)
        if ann.shape[1] < 4:
            # some files might be comma-separated
            ann = pd.read_csv(ann_path, header=None)
        if ann.shape[1] < 4:
            raise ValueError(f"Annotation format not recognized for: {ann_path}")

        ann = ann.iloc[:, :4]
        ann.columns = ["start", "end", "crackles", "wheezes"]

        for i, r in ann.iterrows():
            start = float(r["start"])
            end = float(r["end"])
            c = int(r["crackles"])
            w = int(r["wheezes"])

            label = CLASS_MAP.get((c, w))
            if label is None:
                continue

            s_idx = int(start * SR)
            e_idx = int(end * SR)
            if e_idx <= s_idx:
                continue

            seg = y[s_idx:e_idx]

            target_len = int(MAX_SEC * SR)
            seg = pad_or_trim(seg, target_len)

            logmel = audio_to_logmel(seg, SR)

            # save as .npy (fast)
            out_name = f"{stem}_seg{i:03d}.npy"
            out_path = os.path.join(OUT_DIR, out_name)
            np.save(out_path, logmel)

            rows.append({
                "stem": stem,
                "wav_path": wav_path,
                "ann_path": ann_path,
                "segment_index": i,
                "start": start,
                "end": end,
                "crackles": c,
                "wheezes": w,
                "label": label,
                "feature_path": out_path,
                "n_mels": N_MELS
            })
            seg_count += 1

    if not rows:
        raise RuntimeError("No segments were created. Check that .wav files exist and annotations are correct.")

    manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(OUT_DIR, "segments_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print("\n✅ Done")
    print("Segments:", seg_count)
    print("Saved:", manifest_path)
    print("\nClass counts:")
    print(manifest["label"].value_counts().sort_index())

if __name__ == "__main__":
    main()
