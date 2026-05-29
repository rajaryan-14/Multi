# Multi-Modal Emotion Recognition and Music Recommendation System

A Streamlit app that detects emotion from facial expressions and text, then recommends songs that match the detected emotional state.

## Features

- Facial emotion recognition using OpenCV and a trained Keras model.
- Text emotion recognition using the text emotion model stored in `emotion_module`.
- Music recommendation using the filtered track dataset in `music_module`.
- Browser webcam support for live facial emotion input.
- Streamlit interface for combining emotion detection and recommendations.

## Repository Structure

```text
.
├── app.py
├── requirement.txt
├── emotion_module/
│   ├── browser_webcam_component.py
│   ├── browser_webcam_frontend/
│   ├── emotion_app.py
│   ├── haarcascade_frontalface_default.xml
│   ├── model.h5
│   └── text_emotion.pkl
└── music_module/
    ├── filtered_track_df.csv
    └── music_app.py
```

## Setup

```bash
git clone https://github.com/rajaryan-14/Multi.git
cd Multi
pip install -r requirement.txt
```

## Run

```bash
streamlit run app.py
```

The app runs locally at `http://localhost:8501`.

## Notes

- Runtime logs, Python caches, virtual environments, local environment files, and Playwright screenshots are ignored by Git.
- Keep model and dataset files in their existing folders unless the code paths are updated.
