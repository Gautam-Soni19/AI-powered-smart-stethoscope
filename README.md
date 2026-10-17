# AI-Powered Smart Stethoscope 🩺🤖

An end-to-end deep learning and clinical decision-support pipeline for automated respiratory sound analysis using acoustic auscultation data from the **ICBHI (International Conference on Biomedical Health Informatics) 2017 Respiratory Sound Database**.

---

## 🌟 Highlights & Features

- **Acoustic Preprocessing**: Extracts log-mel spectrogram features ($64$ mel bands, $16\text{ kHz}$ sampling rate, normalized) from segmented respiratory cycles.
- **Hybrid Neural Architecture**:
  - **2D Convolutional Neural Network (CNN)** for spectral spatial pattern extraction (formants, adventitious sound bursts).
  - **Bidirectional GRU (BiGRU)** for temporal sequential dependency modeling across respiratory phases (inhalation / exhalation).
  - **Self-Attention Pooling** mechanism to dynamically weight informative acoustic segments.
- **Four-Class Adventitious Sound Event Detection**:
  - `0`: **Normal**
  - `1`: **Crackles** (discontinuous explosive sounds indicative of fluid or reopening of collapsed airways)
  - `2`: **Wheezes** (continuous musical sounds indicative of airway obstruction)
  - `3`: **Both** (co-occurrence of crackles and wheezes)
- **Clinical Rule Decision Engine**: Translates acoustic segment distributions into clinical-style diagnostic risk assessments:
  - **Normal**
  - **COPD** (Chronic Obstructive Pulmonary Disease)
  - **Asthma**
  - **Pneumonia**

---

## 🏗️ System Architecture

```
Raw Respiratory Audio (.wav)
          │
          ▼
Annotation-Guided Audio Segmentation (02_make_segments_and_mels.py)
          │
          ▼
Log-Mel Spectrogram Transformation (64 Mel bins, 16 kHz)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│               Deep Neural Network (03)                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 2D CNN Layers (Conv2D -> BatchNorm -> ReLU)      │  │
│  └────────────────────────┬─────────────────────────┘  │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Bidirectional GRU (Temporal Sequence Modeling)   │  │
│  └────────────────────────┬─────────────────────────┘  │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Self-Attention Pooling                           │  │
│  └────────────────────────┬─────────────────────────┘  │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Dense Classification Head (4 Classes)            │  │
│  └──────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
Segment Predictions: Normal / Crackles / Wheezes / Both (04, 05)
                            │
                            ▼
Clinical Rule-Based Disease Engine: Normal / COPD / Asthma / Pneumonia (06)
```

---

## 📂 Repository Structure

```text
├── Code/
│   ├── 01_build_file_labels.py           # Maps patient diagnosis CSV to file labels
│   ├── 02_make_segments_and_mels.py      # Audio segmentation & Log-Mel spectrogram generation
│   ├── 03_train_cnn_bigru_attention.py   # Model architecture & PyTorch training pipeline
│   ├── 04_predict_patient.py             # Single-file audio inference script
│   ├── 05_predict_segments_and_decide.py # Multi-segment sliding-window inference & voting
│   └── 06_disease_rules.py               # Clinical heuristics for disease classification
├── ICBHI_final_database/
│   ├── *.txt                             # Official ICBHI respiratory cycle & event annotations
│   ├── script.sh                         # Annotation normalization helper script
│   └── processed/
│       ├── best_model.pt                 # Saved PyTorch checkpoint weights
│       ├── best_model_meta.json          # Checkpoint metadata & label mapping
│       ├── segments_manifest.csv         # Segment annotations & extracted feature index
│       └── train_log.csv                 # Epoch-by-epoch training and validation metrics
├── .gitignore                            # Excludes bulky raw audio and feature arrays
└── README.md                             # Project overview and documentation
```

---

## ⚙️ Installation & Prerequisites

### 1. Clone the Repository
```bash
git clone https://github.com/Gautam-Soni19/AI-powered-smart-stethoscope.git
cd AI-powered-smart-stethoscope
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # or cpu
pip install librosa soundfile numpy pandas scikit-learn tqdm
```

---

## 🚀 Pipeline Usage Guide

### Step 1: Prepare Patient Diagnosis Labels (Optional)
```bash
python Code/01_build_file_labels.py
```

### Step 2: Extract Log-Mel Spectrogram Segments
Extracts audio slices corresponding to annotated respiratory cycles and computes normalized log-mel spectrograms:
```bash
python Code/02_make_segments_and_mels.py
```

### Step 3: Train the Deep Learning Model
Trains the CNN-BiGRU-Attention architecture on CUDA/CPU and outputs the best checkpoint:
```bash
python Code/03_train_cnn_bigru_attention.py
```

### Step 4: Run Inference on a Single Audio Recording
```bash
python Code/04_predict_patient.py --wav "path/to/patient_recording.wav"
```

### Step 5: Multi-Segment Voting & Confidence Scoring
```bash
python Code/05_predict_segments_and_decide.py --wav "path/to/patient_recording.wav" --save_json
```

### Step 6: Disease-Level Decision Engine
Feed the segment predictions into clinical rules to estimate pathology:
```bash
python Code/06_disease_rules.py --json "ICBHI_final_database/processed/sample_segment_decision.json" --save_json
```

---

## 📊 Model Training & Evaluation

The trained model checkpoint (`best_model.pt`) was trained for 20 epochs with batch size 32 using Adam optimizer ($lr=0.001$) and cross-entropy loss:

- **Input Representation**: 64-band Log-Mel Spectrogram ($1024$ FFT, $256$ Hop)
- **Trained Parameters**: 2D Convolutional Feature Extractor, 2-layer Bidirectional GRU, Self-Attention layer.
- **Validation Progress**: Loss reduced consistently from $1.13$ to $0.87$, achieving $>61.8\%$ segment-level accuracy across imbalanced respiratory sound classes.

---

## 📜 License & Acknowledgments

- **Dataset**: ICBHI 2017 Respiratory Sound Database provided by the International Conference on Biomedical Health Informatics.
- Built for AI-assisted digital stethoscope auscultation research and acoustic diagnostics.
