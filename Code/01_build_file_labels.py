# 01_build_file_labels.py
import os
import pandas as pd

DATASET_DIR = r"C:\Users\tosaw\Music\PQ11\ICBHI_final_database"

def main():
    diag_path = os.path.join(DATASET_DIR, "patient_diagnosis.csv")

    if not os.path.exists(diag_path):
        print("⚠️ patient_diagnosis.csv NOT found in dataset folder.")
        print("✅ This is OK for 4-class event training (Normal/Crackles/Wheezes/Both).")
        print("➡️ Skip this script and run: 02_make_segments_and_mels.py")
        return

    df = pd.read_csv(diag_path)

    cols = {c.lower().strip(): c for c in df.columns}
    pcol = cols.get("patient number") or cols.get("patient_number") or cols.get("patient")
    dcol = cols.get("diagnosis") or cols.get("disease")

    if not pcol or not dcol:
        raise ValueError(f"Columns not recognized. Found: {list(df.columns)}")

    df = df[[pcol, dcol]].rename(columns={pcol: "patient_id", dcol: "diagnosis"})
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip()

    out_path = os.path.join(DATASET_DIR, "file_labels.csv")
    df.to_csv(out_path, index=False)

    print("✅ Saved:", out_path)
    print(df["diagnosis"].value_counts().head(20))

if __name__ == "__main__":
    main()
