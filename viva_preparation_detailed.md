# Viva Preparation Guide

## Project Title

Multi-Modal Emotion Recognition and Music Recommendation System

## 1. One-line project summary

This project detects a user's emotion from text, image, video, or live camera input and recommends songs whose audio features best match the detected mood.

## 2. 30-second introduction

Our project is an emotion-aware music recommendation system. Instead of recommending songs only from past listening history, it first understands the user's current mood using multiple input modalities such as text, image, video, and live camera. After detecting the dominant emotion, it maps that emotion to music attributes like energy, valence, danceability, acousticness, instrumentalness, and tempo, and then retrieves the nearest matching songs from a Spotify-based dataset. The goal is to make recommendations more personal, immediate, and emotionally relevant.

## 3. 1-minute explanation of the full pipeline

1. The user opens the EXPLORE page and chooses one mode: Text, Image, Video, Live Camera, or Mixed.
2. The selected input is processed by the corresponding emotion recognition pipeline.
3. The system produces emotion probabilities across the seven Ekman-style emotions:
   Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
4. In Mixed mode, text and visual emotions are combined using weighted fusion.
5. The final dominant emotion is stored in session state and passed to the music module.
6. The music module converts that emotion into a target music feature vector.
7. A nearest-neighbor search finds songs closest to that target vector.
8. The interface shows recommendations, explanations, audio-feature charts, and track-level reasons.

## 4. Main problem statement

Traditional music recommendation systems usually rely on:

- listening history
- genre preference
- collaborative filtering

These methods do not adapt to the user's real-time emotional state. If a person is stressed, sad, excited, or calm in the present moment, conventional systems may still recommend songs based only on old preferences. Our project solves that gap by recommending music using the user's current emotion.

## 5. Why this project matters

- It improves personalization.
- It makes recommendation systems more emotionally aware.
- It can support well-being and mood regulation.
- It is a strong applied AI project because it combines computer vision, NLP, recommender systems, and UI engineering.

## 6. Core topics you should know for viva

### 6.1 Multimodal emotion recognition

Multimodal means using more than one type of input. In this project:

- text captures what the user says
- image captures facial expression from a photo
- video captures emotion across several frames
- live camera captures emotion in real time
- mixed mode combines text with a visual modality

Why multimodal:

- a face can be unclear or misleading
- text can be ambiguous on its own
- combining sources usually improves robustness

### 6.2 Emotion classes used

The system uses these emotion labels:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

These are stored in the code in `EMOTION_LABELS` inside `C:\Users\rasin\Downloads\Multimodal-Music-Recommendation-using-Emotions-main\emotion_module\emotion_app.py`.

### 6.3 Facial emotion recognition

For image and live/video facial analysis, the system:

1. detects faces using OpenCV Haar cascade classifiers
2. converts the face region to grayscale
3. resizes the crop to the CNN input size
4. runs the trained model
5. outputs class probabilities

Important points to say:

- Haar cascade is used because it is lightweight and fast.
- CNN is good for learning spatial features such as mouth shape, eyebrow movement, and eye-region changes.
- The project uses a saved Keras model file `model.h5`.

### 6.4 Text emotion recognition

For text, the running prototype uses a saved scikit-learn pipeline stored in:

`C:\Users\rasin\Downloads\Multimodal-Music-Recommendation-using-Emotions-main\emotion_module\text_emotion.pkl`

The loaded pipeline contains:

- `CountVectorizer`
- `LogisticRegression`

So the practical text path in the current deployed app is:

1. convert text into a vector representation
2. run logistic regression
3. produce class probabilities
4. map text labels like `joy` to project labels like `Happy`

This logic is implemented in `predict_text_emotion()` in `emotion_app.py`.

### 6.5 Weighted fusion in Mixed mode

When both text and visual signals are available, the system performs weighted fusion:

`final_probability = text_weight * text_probability + image_weight * image_probability`

This is implemented in:

`fuse_emotions()` in `C:\Users\rasin\Downloads\Multimodal-Music-Recommendation-using-Emotions-main\emotion_module\emotion_app.py`

Why weighted fusion is useful:

- text may better represent inner emotion
- face may better represent visible expression
- user can control the relative influence

### 6.6 Video emotion logic

For uploaded video, the current prototype:

1. reads frames using OpenCV
2. samples frames across the video
3. predicts emotions on sampled frames
4. averages the emotion probabilities
5. uses the average as the video emotion result

This is implemented in:

`predict_video_emotion()` in `emotion_app.py`

### 6.7 Live camera logic

The live camera is browser-native in the current app.

How it works:

1. browser camera opens using `getUserMedia`
2. frontend captures frames at intervals
3. frames are sent to Streamlit
4. backend runs face detection and emotion prediction
5. the result is drawn as an OpenCV-style overlay
6. when the user clicks recommend, the current live emotion is used

Important frontend file:

`C:\Users\rasin\Downloads\Multimodal-Music-Recommendation-using-Emotions-main\emotion_module\browser_webcam_frontend\main.js`

Important backend wrapper:

`C:\Users\rasin\Downloads\Multimodal-Music-Recommendation-using-Emotions-main\emotion_module\browser_webcam_component.py`

### 6.8 Music recommendation logic

The recommendation module uses six audio features:

- acousticness
- danceability
- energy
- instrumentalness
- valence
- tempo

The flow is:

1. detect emotion
2. convert emotion to a target feature vector
3. filter songs by genre and year range
4. use nearest-neighbor search
5. return the closest songs

This is implemented mainly in:

- `n_neighbors_uri_audio()`
- `render_recommendation_story()`
- `render_song_meta()`
- `build_track_reason()`

inside:

`C:\Users\rasin\Downloads\Multimodal-Music-Recommendation-using-Emotions-main\music_module\music_app.py`

### 6.9 Why KNN is suitable here

KNN is suitable because:

- there is no expensive retraining for each query
- recommendation is based directly on feature similarity
- it works well for cold-start mood-based recommendation
- the audio feature vector is low-dimensional and interpretable

### 6.10 Streamlit UI architecture

The project uses Streamlit for the interface because:

- it is fast for prototyping AI applications
- it easily supports forms, uploads, charts, and session state
- it works well with Python ML pipelines

The main app file is:

`C:\Users\rasin\Downloads\Multimodal-Music-Recommendation-using-Emotions-main\app.py`

It sets the page and applies the EXPLORE theme.

## 7. Important functions you should know

### 7.1 From `emotion_app.py`

- `load_models()`
  Loads Haar cascades, CNN model, and text model.

- `predict_text_emotion(text)`
  Runs the text classifier and returns emotion probabilities.

- `detect_faces(gray, fast=False)`
  Detects faces using OpenCV cascades.

- `predict_image_emotion(image)`
  Detects faces in an uploaded image and predicts facial emotion.

- `analyze_frame_emotions(frame, fast=False, max_faces=1)`
  Runs live-frame analysis for camera input.

- `predict_frame_emotion(frame, fast=False, max_faces=1, annotate=False)`
  Predicts frame-level emotion and optionally annotates it.

- `fuse_emotions(text_emotions, image_emotions, text_weight=0.4, image_weight=0.6)`
  Combines probabilities from two modalities.

- `average_emotions(results)`
  Averages emotion probabilities from multiple detections or frames.

- `predict_video_emotion(video_bytes, suffix)`
  Samples video frames and averages predictions.

- `render_live_detection(...)`
  Connects live webcam UI to backend inference.

- `run_text_flow()`
  Text mode UI + inference.

- `run_image_flow()`
  Image mode UI + inference.

- `run_live_camera_flow()`
  Live camera UI + inference.

- `run_video_flow()`
  Video mode UI + inference.

- `run_mixed_flow()`
  Mixed mode UI + weighted combination.

- `run()`
  Main entry point for the emotion module.

### 7.2 From `music_app.py`

- `load_data()`
  Loads and preprocesses the Spotify track dataset.

- `n_neighbors_uri_audio(genre, start_year, end_year, test_feat)`
  Finds the nearest songs to the target feature vector.

- `render_recommendation_story(...)`
  Explains why songs were recommended.

- `render_mixed_breakdown(breakdown)`
  Shows text vs visual vs final mix in Mixed mode.

- `render_song_meta(track, rank, target_profile)`
  Shows title, artist, tags, and match details.

- `build_track_reason(track, target_profile)`
  Generates the short "why it fits" explanation.

- `render_audio_chart(audio, chart_key)`
  Plots a radar chart of song features.

- `render_controls(selected_emotion)`
  Lets the user change genre and components.

- `run()`
  Main entry point for the music module.

### 7.3 From the webcam frontend

- `drawOverlay(overlay)`
  Draws face box and emotion label over the live camera feed.

- `captureFrame()`
  Captures the browser webcam frame and sends it to Streamlit.

- `ensureStream()`
  Starts webcam access through `getUserMedia`.

- `onRender(event)`
  Updates the component when Streamlit sends new data.

## 8. Libraries and why they were used

- `streamlit`
  UI framework for the web app.

- `opencv-python`
  Face detection, frame capture, image preprocessing.

- `tensorflow` and `keras`
  Loading and running the CNN emotion model.

- `numpy`
  Matrix and probability operations.

- `pandas`
  Track dataset loading and manipulation.

- `scikit-learn`
  Text classification pipeline and nearest-neighbor recommendation.

- `plotly`
  Visualizing audio features.

- `Pillow`
  Image handling and annotation.

- `joblib`
  Loading the serialized text model.

## 9. Datasets and assets

### Emotion side

- FER2013 is the standard academic dataset mentioned in your report for facial emotion recognition.
- GoEmotions is the standard academic dataset mentioned in your report for text emotion classification.

### Music side

- The app uses `filtered_track_df.csv` from the `music_module`.
- It contains Spotify-style metadata and audio features.

## 10. Strengths of the project

- Combines AI from multiple domains.
- Supports multiple input modes, not only one.
- Gives explainable recommendations.
- Has a working UI instead of only notebook results.
- Uses live camera and mixed mode, which makes the demo stronger.

## 11. Limitations you should admit honestly

- Facial recognition can be affected by lighting, face angle, and camera quality.
- Text emotion can misread sarcasm or very short input.
- Live camera performance depends on browser/device resources.
- Music recommendation is feature-based and does not yet learn long-term user taste.
- The current prototype prioritizes responsiveness and usability over the heaviest research-grade models.

## 12. Very important viva honesty note

Your PPT and report use some research-oriented language that is stronger than the current deployed prototype.

### What the current working prototype actually uses

- Text: `CountVectorizer + LogisticRegression` pipeline loaded from `text_emotion.pkl`
- Image/face: Haar cascade + CNN model in `model.h5`
- Video: sampled frame prediction + average aggregation
- Mixed: weighted probability fusion
- Music recommendation: `NearestNeighbors` search on six audio features

### Where your PPT/report language is more conceptual

- PPT mentions BERT
- PPT mentions CNN + LSTM for video
- report draft mentions K-Means in some places

### Safest viva phrasing

Say this if needed:

"Our academic design and semester study explored BERT-based text understanding and temporal video modeling ideas such as CNN with sequence reasoning. In the final working prototype we used lighter deployable components for real-time integration, including a serialized text classifier, CNN-based facial emotion prediction, weighted multimodal fusion, and nearest-neighbor music retrieval."

This answer is honest, technically safe, and still sounds strong.

## 13. Slide-by-slide viva understanding

### Slide 1: Title

Say:

- project name
- your department and institute
- team members
- supervisor

### Slide 2: Problem statement

Key point:

Existing platforms are personalized but not emotion-aware in real time.

### Slide 3: Semester 1 baseline

Key point:

You already had text and static image understanding, but frame-by-frame vision caused unstable emotion predictions over time.

### Slide 4: Work done this semester

Use this as your transition slide:

- moved from separate modules to integrated pipeline
- improved UX
- added live camera and mixed mode
- connected emotion output to song recommendation

### Slide 5: Integrated song recommendation

Explain the complete chain:

emotion -> target music vector -> nearest songs -> recommendation display

### Slide 6: Music feature parameters

Know the meaning:

- Energy = intensity
- Valence = positivity
- Danceability = rhythm suitability
- Tempo = speed in BPM
- Acousticness = organic vs electronic feel
- Instrumentalness = fewer vocals, more instruments

### Slide 7: KNN logic

Important words:

- 6-dimensional feature space
- Euclidean distance
- nearest neighbors

### Slide 8: Why KNN

Main answer:

Because it is simple, interpretable, fast enough, and suitable for mood-to-feature retrieval.

### Slide 9: Likely visual/demo slide

If asked, explain what the interface shows and how a user would interact with it.

### Slide 10: Multimodal integration

Key point:

Different streams are processed independently and then combined.

### Slide 11: Weighted decision fusion

Key equation idea:

final score = weighted sum of modality scores

### Slide 12: Likely visual/results slide

Talk about the actual user flow or demo output.

### Slide 13: Video training concept

Even if the slide says CNN + LSTM, be careful. You can say:

- CNN is for spatial frame-level features
- temporal modeling is useful because emotion changes over time
- in the working prototype, frame aggregation is used for practical real-time integration

### Slide 14: Spatial vs temporal logic

Know the difference:

- spatial = what is present in a frame
- temporal = how the expression evolves across frames

### Slide 15: Likely final system/demo slide

Use this to summarize the completed system and practical usability.

### Slide 16: Thank you

Be ready for questions on:

- why multimodal
- why KNN
- why not collaborative filtering
- limitations
- future work

## 14. Likely viva questions with strong answers

### Q1. Why did you choose this project?

We chose it because music is closely linked to emotion, but current recommendation systems usually ignore the user's present mood. This project let us combine emotion AI and recommender systems in a practical way.

### Q2. What makes your system multimodal?

It can infer emotion from more than one input type: text, image, video, and live camera. Mixed mode also combines text with a visual signal using weighted fusion.

### Q3. Why not use only text or only image?

Single modalities are fragile. Text may miss facial expression, and facial expression may not represent internal emotion. Multimodal input improves robustness and flexibility.

### Q4. Why is CNN suitable for facial emotion recognition?

CNNs automatically learn spatial patterns such as edges, shapes, and local facial configurations, which are essential for recognizing expressions like smiles, frowns, and surprise.

### Q5. What is the role of Haar cascade?

Haar cascade is used for fast face detection before emotion classification. It isolates the face region so the CNN works only on the relevant part of the image.

### Q6. Why did you use weighted fusion?

Because different modalities are not equally reliable in every case. Weighted fusion lets us tune influence and gives better control than simply choosing one input blindly.

### Q7. Why did you choose KNN for recommendation?

KNN is simple, interpretable, and effective for feature-space similarity. Since we already have a target audio vector, nearest-neighbor retrieval is a natural fit.

### Q8. What features are used for song recommendation?

Acousticness, danceability, energy, instrumentalness, valence, and tempo.

### Q9. Why is valence important?

Valence represents how positive or negative a song feels. It is one of the most direct bridges between detected emotion and music mood.

### Q10. Why not use collaborative filtering?

Collaborative filtering depends heavily on user history and similar-user patterns. Our goal is current-emotion-based recommendation, so content/audio-feature similarity is more appropriate.

### Q11. What are the biggest challenges in this project?

- real-time emotion detection
- handling noisy webcam/video input
- balancing speed and accuracy
- combining multiple modalities
- giving explainable recommendations

### Q12. What are the limitations of facial emotion recognition?

Lighting, camera angle, occlusion, expression ambiguity, and differences across users can affect prediction quality.

### Q13. How does the system handle video?

In the current prototype, it samples frames from the video, predicts frame-level emotion, and averages the results.

### Q14. What is the advantage of live camera mode?

It makes the system interactive and demonstrates real-time affect-aware recommendation rather than only offline analysis.

### Q15. What is the difference between image mode and live camera mode?

Image mode analyzes one uploaded frame. Live camera mode continuously receives frames from the browser webcam and updates the detected emotion interactively.

### Q16. What future improvements would you suggest?

- stronger temporal modeling
- personalized long-term recommendation
- multilingual text emotion support
- better fairness evaluation
- user feedback loops such as more calm or more energetic

### Q17. How is privacy handled?

The app processes input for inference and recommendation without building a persistent personal emotional profile. In the prototype, the emotion is used for the session flow and recommendation context.

### Q18. What if text and image disagree?

That is exactly why fusion is useful. The system can weight the modalities instead of forcing one to dominate every time.

## 15. Tricky questions and safe responses

### If asked: "Are you really using BERT in the final app?"

Safe answer:

"BERT was part of our academic design direction and report framing for text emotion understanding. In the current integrated prototype we deployed a lighter serialized text classifier so that the full system remains simpler and more responsive in the demo environment."

### If asked: "Where is the LSTM in the running system?"

Safe answer:

"The temporal modeling idea is captured in our presentation architecture. In the current deployable demo we use sampled-frame aggregation for video because it integrated more reliably into the working application timeline."

### If asked: "Is it KNN or K-Means?"

Safe answer:

"In the current recommendation module it is nearest-neighbor retrieval over audio features. K-Means appeared in our broader design discussion as a clustering idea, but the working app uses nearest-neighbor search for direct similarity-based recommendations."

## 16. What each team member should own in Q&A

### Shubhanshu Gupta

- overall architecture
- CNN and face pipeline
- image/video/live camera flow
- limitations of visual recognition

### Raj Aryan Singh

- text emotion analysis
- meaning of multimodal fusion
- why textual emotion matters
- conceptual BERT-related discussion

### Vaanya Sharma

- music recommendation engine
- audio feature meanings
- KNN logic
- interface integration and final user flow

## 17. Short closing statement for viva

Our project demonstrates how emotion recognition and recommendation systems can be combined into a practical user-facing application. The main contribution is not only detecting emotion, but using it immediately to produce explainable music recommendations through multiple input modes and a usable interface.

## 18. Last-night preparation checklist

- Practice the 30-second intro three times.
- Practice the 1-minute full pipeline once without looking.
- Memorize the six music features.
- Memorize why KNN was used.
- Memorize what weighted fusion means.
- Be ready with the honest explanation about BERT/LSTM/KNN vs prototype details.
- Decide who answers which technical area.
- Keep one backup example ready:
  "If the user looks sad and types a negative sentence, the system maps that to low valence and lower energy, then retrieves songs that best match that profile."
