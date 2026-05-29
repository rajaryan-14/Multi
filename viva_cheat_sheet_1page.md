# 1-Page Viva Cheat Sheet

## Project in 2 Lines

Multi-Modal Emotion Recognition and Music Recommendation System detects user emotion from text, image, video, live camera, or mixed input, then recommends songs whose audio features best match that mood.

Core idea: emotion detection -> final mood -> target music vector -> nearest matching songs.

## 30-Second Intro

"Our project makes music recommendation emotion-aware in real time. Instead of relying only on listening history, it first detects the user's current mood using text and visual inputs, then maps that mood to song features like valence, energy, danceability, acousticness, instrumentalness, and tempo, and retrieves the closest matching songs."

## Full Pipeline

1. User selects Text / Image / Video / Live Camera / Mixed
2. Emotion is predicted from the chosen modality
3. Mixed mode uses weighted fusion
4. Final emotion is stored
5. Music module maps emotion to target audio-feature vector
6. Nearest-neighbor search returns matching songs

## Inputs and Models

- Text: text classifier pipeline
- Image: Haar cascade face detection + CNN emotion model
- Video: sampled frames + same image CNN + averaged predictions
- Live camera: browser webcam + repeated frame-wise emotion detection
- Mixed: text + visual weighted fusion

## Very Important "Safe Truth" for Viva

Conceptual/academic design in report and PPT mentions:
- BERT for text
- CNN + LSTM for video
- K-Means in some places

Current working prototype actually uses:
- Text: `CountVectorizer + LogisticRegression`
- Face/image/video: OpenCV + CNN
- Video: multi-frame sampling + averaging
- Music recommendation: `NearestNeighbors`

Safe answer:
"Our academic design explored stronger research-oriented models, but the final integrated prototype uses lighter deployable components for stable real-time performance."

## Main Results

### CNN Facial Emotion Model
- Baseline CNN:
  - Macro-F1: 0.42
  - Disgust Recall: 0.00
  - Accuracy: 58.2%
- Weighted-loss CNN:
  - Macro-F1: 0.57
  - Disgust Recall: 0.48
  - Accuracy: 60.9%

### Text Model
- Validation Accuracy: 84.2%
- Macro-F1: 0.82

### Multimodal System
- Accuracy: 78%
- Macro-F1: 0.78
- End-to-End Latency: 180 ms

## Most Important Comparisons

### 1. Baseline vs Improved CNN
- Macro-F1: 0.42 -> 0.57
- Disgust Recall: 0.00 -> 0.48
- Accuracy: 58.2% -> 60.9%

Takeaway: weighted loss fixed minority-class failure.

### 2. CNN vs Text vs Multimodal
- CNN only: 60.9%
- Text only: 84.2%
- Multimodal: 78%

Takeaway: text is strongest in accuracy, multimodal is strongest in robustness.

### 3. Fusion Threshold
- 0.50 -> 0.71
- 0.55 -> 0.74
- 0.65 -> 0.78 best
- 0.75 -> 0.76
- 0.85 -> 0.73

Takeaway: threshold 0.65 gave best balance.

### 4. GPU vs CPU
- GPU total latency: 95 ms
- CPU total latency: 166 ms
- Speedup: about 1.75x

## Latency Breakdown

- CNN: 45 ms
- Text model: 80 ms
- Preprocessing: 15 ms
- Fusion: 5 ms
- Recommendation query: 35 ms
- Total: 180 ms

Takeaway: text model is the main bottleneck.

## 6 Music Features

- Valence = positivity
- Energy = intensity
- Danceability = rhythm suitability
- Tempo = BPM
- Acousticness = organic/acoustic feel
- Instrumentalness = instrument-heavy vs vocal-heavy

## 5 High-Probability Viva Questions

### Why multimodal?
Single modalities are fragile. Combining text and visual signals improves robustness.

### Why KNN / nearest-neighbor?
Because recommendation is feature-similarity based, simple, interpretable, and works for cold-start mood queries.

### How is video different from image?
Image uses one frame. Video uses multiple sampled frames and averages predictions for more stability.

### Are you using a separate video model?
No, not in the current prototype. We reuse the image CNN on sampled video frames.

### Why weighted fusion?
Because text and face are not equally reliable in all situations, so weighted fusion gives controlled combination.

## Strengths

- End-to-end working system
- Multiple input modalities
- Explainable recommendations
- Real-time interactive demo
- Privacy-first project framing

## Limitations

- Face accuracy depends on lighting and camera quality
- Text may miss sarcasm or ambiguity
- Recommendation is feature-based, not long-term personalized
- Current video path does not use explicit temporal sequence learning

## Best Final Closing Line

"Our key contribution is not just emotion detection, but an end-to-end multimodal system that turns detected mood into explainable, real-time music recommendations with a usable interface."
