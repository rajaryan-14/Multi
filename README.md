# 🎵 Multi-Modal Emotion Recognition & Music Recommendation System

> A real-time affective computing system that detects your emotional state from **facial expressions** and **text**, then recommends music that matches how you feel — not what you listened to last week.

**B.Tech Final Year Project · JIIT Noida · May 2026**  
**Supervisor:** Prof. Richa Gupta, Dept. of ECE

---

## 🧠 The Problem We Solved

Modern music recommendation engines (Spotify, YouTube Music) are calibrated to your *historical* preferences — they know what you liked yesterday but have zero awareness of how you feel *right now*. A user experiencing anxiety receives the same uptempo workout playlist they always get. This is the **Affective Gap**.

This system closes that gap by inferring your present emotional state from your face and/or text, then mapping that state to emotionally coherent music in real time.

---

## ✨ Key Results

| Metric | Value |
|---|---|
| CNN Accuracy (7-class, Phase III) | **60.90%** |
| CNN Macro-F1 | **0.5748** (+28.7% over baseline) |
| Disgust Recall (baseline → final) | **0.00 → 0.48** |
| BERT Validation Accuracy | **84.2%** |
| Fusion System Accuracy | **82.7%** |
| End-to-End Latency (mean) | **178 ms** on consumer CPU |
| Privacy | Zero persistent storage of biometric data |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│   Live Webcam (30fps)  │  Static Image  │  Text Entry       │
└────────────┬────────────────────┬────────────────┬──────────┘
             │                    │                │
             ▼                    ▼                ▼
┌────────────────────┐   ┌────────────────────────────────────┐
│  VISUAL PIPELINE   │   │         TEXT PIPELINE              │
│  Haar Cascade      │   │  BERT WordPiece Tokeniser          │
│  → Grayscale       │   │  → 128-token sequence              │
│  → Bicubic 48×48   │   │  → Attention mask                  │
│  → Normalise [0,1] │   └────────────────────────────────────┘
└────────┬───────────┘                    │
         │                               │
         ▼                               ▼
┌─────────────────┐             ┌─────────────────┐
│  CNN (4-block)  │             │  BERT fine-tuned │
│  TensorFlow     │             │  GoEmotions      │
│  4.2M params    │             │  PyTorch / HF    │
└────────┬────────┘             └────────┬────────┘
         │    7-dim prob vector          │    7-dim prob vector
         └──────────────┬───────────────┘
                        ▼
           ┌────────────────────────┐
           │  DECISION-LEVEL FUSION │
           │  if max(P_bert) > 0.65 │
           │    → use BERT result   │
           │  else → use CNN result │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │  MUSIC RECOMMENDATION  │
           │  Russell Circumplex    │
           │  → KNN on Spotify      │
           │  → 17,340 tracks       │
           └────────────────────────┘
```

---

## 🔬 Three-Phase CNN Training Experiment

The core research contribution of this project is a systematic investigation of class imbalance on FER2013 and its remediation via Weighted Categorical Cross-Entropy.

### Phase I — Baseline (7-class, unweighted loss)
Standard training revealed complete **Disgust collapse**: the model never correctly predicted a single disgusted face across the entire validation set (Recall = 0.00). Overall accuracy was 56.86% but Macro-F1 only 0.4467 — a misleadingly high accuracy masking total failure on minority classes.

### Phase II — Diagnostic (6-class, Disgust removed)
Removing Disgust from training improved raw accuracy to **65.49%** and Fear Recall from 0.13 → 0.42. This proved the failure was caused by **class imbalance**, not visual ambiguity. However, a model that cannot recognise one of the seven universal emotions is not a solution.

### Phase III — Remediation (7-class, weighted loss)
Weighted Categorical Cross-Entropy with inverse-frequency class weights (Disgust weight ≈ 9.41×, Happy weight ≈ 0.46×) recovered Disgust Recall to **0.48** with a Macro-F1 of **0.5748** — a 28.7% relative improvement over Phase I on the same task.

| Phase | Classes | Loss | Accuracy | Macro-F1 | Disgust Recall |
|---|---|---|---|---|---|
| I | 7 | Standard CE | 56.86% | 0.4467 | 0.00 |
| II | 6 | Standard CE | 65.49% | 0.6363 | N/A |
| **III** | **7** | **Weighted CE** | **60.90%** | **0.5748** | **0.48** |

---

## 🧩 Tech Stack

| Component | Technology |
|---|---|
| Visual Modality (CNN) | TensorFlow / Keras, OpenCV |
| Textual Modality (BERT) | PyTorch, HuggingFace Transformers |
| Face Detection | OpenCV Haar Cascade |
| Music Recommendation | scikit-learn KMeans, Spotify Web API |
| Frontend | Streamlit |
| Datasets | FER2013 (28,709 images), GoEmotions (58,009 comments), Spotify (~17,340 tracks) |

---

## 📁 Repository Structure

```
Multi/
├── app.py                   # Streamlit entry point + UI orchestration
├── emotion_module/          # CNN inference, preprocessing, fusion logic
├── requirement.txt          # Python dependencies
└── playwright-*.png         # UI screenshots (webcam, mixed-mode, home)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A Spotify Developer account (for API credentials)

### Installation

```bash
git clone https://github.com/rajaryan-14/Multi.git
cd Multi
pip install -r requirement.txt
```

### Running the App

```bash
streamlit run app.py
```

The app runs locally at `http://localhost:8501`. No data is sent to any server — all inference runs on your machine.

> **Note:** You will need to supply your own trained model weights (`facialemotionmodel.h5`) and a fine-tuned BERT checkpoint. The Spotify `filtered_track_df.csv` dataset is also required for music recommendations.
