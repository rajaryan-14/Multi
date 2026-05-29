import base64
import os
import tempfile
import threading
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from emotion_module.browser_webcam_component import browser_webcam
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
LIVE_FRAME_WIDTH = 480
LIVE_FRAME_HEIGHT = 360
LIVE_PREDICTION_INTERVAL = 0.45
LIVE_OVERLAY_TTL = 1.2
LIVE_JPEG_QUALITY = 68
BROWSER_WEBCAM_JPEG_QUALITY = 0.58

FACE_DETECTION_LOCK = threading.Lock()
EMOTION_MODEL_LOCK = threading.Lock()


@st.cache_resource
def load_models():
    cascade_paths = [
        'emotion_module/haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
    ]
    classifiers = []
    for cascade_path in cascade_paths:
        classifier = cv2.CascadeClassifier(cascade_path)
        if not classifier.empty():
            classifiers.append(classifier)

    image_model = load_model('emotion_module/model.h5')
    import joblib
    text_model = joblib.load('emotion_module/text_emotion.pkl')
    return classifiers, image_model, text_model


face_classifiers, image_model, text_model = load_models()


def predict_text_emotion(text):
    try:
        prediction = text_model.predict_proba([text])[0]
        text_emotions = text_model.classes_

        emotion_map = {
            'anger': 'Angry',
            'disgust': 'Disgust',
            'fear': 'Fear',
            'joy': 'Happy',
            'neutral': 'Neutral',
            'sadness': 'Sad',
            'surprise': 'Surprise',
            'shame': 'Sad',
        }

        mapped_emotions = {}
        for i, text_emotion in enumerate(text_emotions):
            mapped_label = emotion_map.get(text_emotion, text_emotion.capitalize())
            mapped_emotions[mapped_label] = mapped_emotions.get(mapped_label, 0) + prediction[i]

        return mapped_emotions
    except Exception as e:
        st.error(f"Text emotion prediction error: {str(e)}")
        return None


def detect_faces(gray, fast=False):
    if not face_classifiers:
        return []

    height, width = gray.shape[:2]
    max_dim = max(height, width)
    target_dim = 420 if fast else 900
    resize_ratio = min(target_dim / max_dim, 1.0)
    detection_gray = gray
    if resize_ratio < 1.0:
        detection_gray = cv2.resize(
            gray,
            (int(width * resize_ratio), int(height * resize_ratio)),
            interpolation=cv2.INTER_AREA,
        )

    equalized_gray = cv2.equalizeHist(detection_gray)

    if fast:
        variants = [equalized_gray, detection_gray]
        detector_settings = [
            {"scaleFactor": 1.12, "minNeighbors": 4, "minSize": (36, 36)},
            {"scaleFactor": 1.08, "minNeighbors": 3, "minSize": (30, 30)},
        ]
        classifiers = face_classifiers[:2]
    else:
        variants = [
            detection_gray,
            equalized_gray,
            cv2.GaussianBlur(equalized_gray, (3, 3), 0),
        ]
        detector_settings = [
            {"scaleFactor": 1.08, "minNeighbors": 4, "minSize": (45, 45)},
            {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (35, 35)},
            {"scaleFactor": 1.03, "minNeighbors": 3, "minSize": (28, 28)},
        ]
        classifiers = face_classifiers

    with FACE_DETECTION_LOCK:
        for classifier in classifiers:
            for variant in variants:
                for settings in detector_settings:
                    faces = classifier.detectMultiScale(variant, **settings)
                    if len(faces) == 0:
                        continue

                    mapped_faces = []
                    for x, y, w, h in faces:
                        if resize_ratio < 1.0:
                            x = int(x / resize_ratio)
                            y = int(y / resize_ratio)
                            w = int(w / resize_ratio)
                            h = int(h / resize_ratio)
                        mapped_faces.append((x, y, w, h))
                    return sorted(mapped_faces, key=lambda face: face[2] * face[3], reverse=True)

    return []


def predict_face_emotions(gray, faces, max_faces):
    results = []
    for x, y, w, h in faces[:max_faces]:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float32') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        with EMOTION_MODEL_LOCK:
            prediction = image_model.predict(roi, verbose=0)[0]

        results.append({EMOTION_LABELS[i]: prediction[i] for i in range(len(EMOTION_LABELS))})
    return results


def annotate_pil_image(image, faces, results):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for face_num, ((x, y, w, h), emotions) in enumerate(zip(faces, results), 1):
        max_emotion = max(emotions, key=emotions.get)
        confidence = emotions[max_emotion]

        draw.rectangle([(x, y), (x + w, y + h)], outline="#c9a84f", width=18)
        label = f"{face_num}. {max_emotion}: {confidence:.2f}"
        try:
            font = ImageFont.truetype("arial.ttf", 72)
        except Exception:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_top = max(0, y - text_height - 16)
        draw.rectangle([(x, label_top), (x + text_width + 24, y)], fill="#1f2716")
        draw.text((x + 12, label_top + 5), label, fill="#f8f4e8", font=font)

    return annotated


def predict_image_emotion(image):
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = detect_faces(gray, fast=False)

    if not faces:
        return None, image

    results = predict_face_emotions(gray, faces, max_faces=min(3, len(faces)))
    return results, annotate_pil_image(image, faces[:len(results)], results)


def resize_live_frame(frame):
    height, width = frame.shape[:2]
    if width <= LIVE_FRAME_WIDTH:
        return frame

    ratio = LIVE_FRAME_WIDTH / width
    return cv2.resize(
        frame,
        (LIVE_FRAME_WIDTH, int(height * ratio)),
        interpolation=cv2.INTER_AREA,
    )


def analyze_frame_emotions(frame, fast=False, max_faces=1):
    frame = resize_live_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, fast=fast)

    if not faces:
        return None, [], frame

    results = predict_face_emotions(gray, faces, max_faces=max_faces)
    mapped_faces = faces[:len(results)]
    return results, mapped_faces, frame


def predict_frame_emotion(frame, fast=False, max_faces=1, annotate=False):
    results, faces, frame = analyze_frame_emotions(frame, fast=fast, max_faces=max_faces)
    if not results:
        return None, frame

    annotated_frame = frame.copy()

    if annotate:
        for face_num, ((x, y, w, h), emotions) in enumerate(zip(faces, results), 1):
            max_emotion = max(emotions, key=emotions.get)
            confidence = emotions[max_emotion]
            label = f"{face_num}. {max_emotion}: {confidence:.2f}"

            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (79, 168, 201), 3)
            cv2.rectangle(annotated_frame, (x, max(0, y - 36)), (x + 260, y), (31, 39, 22), -1)
            cv2.putText(
                annotated_frame,
                label,
                (x + 8, max(24, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (248, 244, 232),
                2,
                cv2.LINE_AA,
            )

    return results, annotated_frame


def draw_live_overlay(frame, detection):
    if not detection:
        return frame

    x, y, w, h = detection["face"]
    label = f'{detection["emotion"]} {detection["confidence"]:.0%}'
    frame_h, frame_w = frame.shape[:2]

    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))

    cv2.rectangle(frame, (x, y), (x + w, y + h), (79, 168, 201), 2)

    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    text_width, text_height = text_size
    label_top = y - text_height - baseline - 16
    if label_top < 8:
        label_top = y + 8

    label_bottom = label_top + text_height + baseline + 12
    label_right = min(frame_w - 8, x + text_width + 24)
    cv2.rectangle(frame, (x, label_top), (label_right, label_bottom), (31, 39, 22), -1)
    cv2.putText(
        frame,
        label,
        (x + 10, label_bottom - baseline - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (248, 244, 232),
        2,
        cv2.LINE_AA,
    )

    return frame


def decode_frame_data_url(frame_data_url):
    if not frame_data_url or "," not in frame_data_url:
        return None

    try:
        encoded_frame = frame_data_url.split(",", 1)[1]
        frame_bytes = base64.b64decode(encoded_frame)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        return cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def build_live_overlay_payload(frame_shape, face, emotions):
    dominant_emotion = max(emotions, key=emotions.get)
    return {
        "face": [int(value) for value in face],
        "emotion": dominant_emotion,
        "confidence": float(emotions[dominant_emotion]),
        "source_width": int(frame_shape[1]),
        "source_height": int(frame_shape[0]),
    }


def process_browser_webcam_payload(prefix, payload):
    if not payload:
        return

    capture_id_key = f"{prefix}_capture_id"
    emotions_key = f"{prefix}_current_emotions"
    overlay_key = f"{prefix}_overlay"
    error_key = f"{prefix}_webcam_error"
    detected_at_key = f"{prefix}_detected_at"

    payload_error = payload.get("error")
    if payload_error:
        st.session_state[error_key] = payload_error
        return

    capture_id = payload.get("capture_id")
    if not capture_id or capture_id == st.session_state.get(capture_id_key):
        return

    st.session_state[capture_id_key] = capture_id
    frame = decode_frame_data_url(payload.get("frame_data_url"))
    if frame is None:
        return

    results, faces, analyzed_frame = analyze_frame_emotions(frame, fast=True, max_faces=1)
    now = time.time()

    if results and faces:
        current_emotions = average_emotions(results)
        st.session_state[emotions_key] = current_emotions
        st.session_state[overlay_key] = build_live_overlay_payload(
            analyzed_frame.shape,
            faces[0],
            current_emotions,
        )
        st.session_state[detected_at_key] = now
        st.session_state.pop(error_key, None)
    elif now - st.session_state.get(detected_at_key, 0.0) > LIVE_OVERLAY_TTL:
        st.session_state.pop(emotions_key, None)
        st.session_state.pop(overlay_key, None)


def fuse_emotions(text_emotions, image_emotions, text_weight=0.4, image_weight=0.6):
    fused = {}
    for emotion in EMOTION_LABELS:
        text_prob = text_emotions.get(emotion, 0)
        image_prob = image_emotions.get(emotion, 0)
        fused[emotion] = (text_weight * text_prob) + (image_weight * image_prob)
    return fused


def average_emotions(results):
    if not results:
        return None

    averaged = {}
    for emotion in EMOTION_LABELS:
        averaged[emotion] = float(np.mean([result.get(emotion, 0) for result in results]))
    return averaged


def predict_video_emotion(video_bytes, suffix):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return None

        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(frame_total // 24, 1)
        frame_index = 0
        sampled_frames = 0
        detections = []

        while sampled_frames < 24:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % step == 0:
                results, _ = predict_frame_emotion(frame, fast=True, max_faces=2, annotate=False)
                if results:
                    detections.extend(results)
                sampled_frames += 1

            frame_index += 1

        cap.release()
        return average_emotions(detections)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def render_explore_hero():
    st.markdown(
        """
        <section class="explore-hero">
            <div class="hero-copy">
                <div class="hero-kicker">Emotion powered music discovery</div>
                <div class="hero-title hero-title-display" id="explore">EXPLORE</div>
                <p class="hero-line">Feel music like never before</p>
            </div>
            <div class="music-scene" aria-hidden="true">
                <div class="record"></div>
                <div class="bars">
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def reset_mode_state(option):
    if st.session_state.get("emotion_mode") == option:
        return

    stop_all_live_cameras()
    for key in [
        "text_emotions",
        "live_camera_emotions",
        "video_emotions",
        "mixed_emotions",
        "mixed_live_camera_emotions",
        "mixed_breakdown",
    ]:
        st.session_state.pop(key, None)
    st.session_state.emotion_mode = option


def render_mode_selector():
    if "emotion_mode" not in st.session_state:
        st.session_state.emotion_mode = "Text"

    options = ["Text", "Image", "Video", "Live Camera", "Mixed"]
    st.markdown('<div class="mode-row">', unsafe_allow_html=True)
    columns = st.columns(5)
    for column, option in zip(columns, options):
        with column:
            if st.button(option, key=f"mode_{option}", use_container_width=True):
                reset_mode_state(option)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    return st.session_state.emotion_mode


def render_emotion_result(title, emotions):
    df = pd.DataFrame(list(emotions.items()), columns=["Emotion", "Probability"])
    df = df.sort_values("Probability", ascending=False)

    dominant_emotion = df.iloc[0]["Emotion"]
    confidence = df.iloc[0]["Probability"]

    st.subheader(title)
    metric_col, chart_col, data_col = st.columns([0.8, 1.3, 1])
    with metric_col:
        st.metric("Detected emotion", dominant_emotion, f"{confidence:.1%}")
    with chart_col:
        st.bar_chart(df.set_index("Emotion"))
    with data_col:
        st.dataframe(df, hide_index=True, use_container_width=True)

    return dominant_emotion


def summarize_text_input(text, limit=96):
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def top_emotions(emotions, limit=3):
    if not emotions:
        return []

    ranked = sorted(
        ((emotion, float(probability)) for emotion, probability in emotions.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"emotion": emotion, "probability": probability}
        for emotion, probability in ranked[:limit]
    ]


def build_emotion_summary(emotions, label=None, limit=3):
    ranked = top_emotions(emotions, limit=limit)
    if not ranked:
        return {
            "label": label,
            "dominant_emotion": "Neutral",
            "confidence": 0.0,
            "top_emotions": [],
        }

    return {
        "label": label,
        "dominant_emotion": ranked[0]["emotion"],
        "confidence": ranked[0]["probability"],
        "top_emotions": ranked,
    }


def set_recommendation_state(source, emotions, extra_context=None):
    summary = build_emotion_summary(emotions, label=source)
    st.session_state.detected_emotion = summary["dominant_emotion"]
    st.session_state.selected_genre = "Pop"
    st.session_state.start_track_i = 0
    st.session_state.page = "music"

    recommendation_context = {
        "source": source,
        "dominant_emotion": summary["dominant_emotion"],
        "confidence": summary["confidence"],
        "top_emotions": summary["top_emotions"],
    }
    if extra_context:
        recommendation_context.update(extra_context)

    st.session_state.recommendation_context = recommendation_context


def recommend_button(emotion, key, source, emotions, extra_context=None):
    if st.button("Recommend Songs", type="primary", key=key):
        stop_all_live_cameras()
        set_recommendation_state(source, emotions, extra_context=extra_context)
        st.rerun()


def open_local_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, LIVE_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 24)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


class LiveCameraBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.capture_thread = None
        self.prediction_thread = None
        self.running = False
        self.latest_raw_frame = None
        self.latest_frame_url = None
        self.latest_emotions = None
        self.latest_detection = None
        self.latest_detection_at = 0.0
        self.latest_error = None

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.latest_raw_frame = None
            self.latest_frame_url = None
            self.latest_emotions = None
            self.latest_detection = None
            self.latest_detection_at = 0.0
            self.latest_error = None
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
            self.capture_thread.start()
            self.prediction_thread.start()

    def stop(self):
        with self.lock:
            self.running = False

    def _capture_loop(self):
        cap = open_local_camera()
        if not cap.isOpened():
            with self.lock:
                self.running = False
                self.latest_error = "Could not open the camera. Close other camera apps and try again."
            return

        try:
            while True:
                with self.lock:
                    if not self.running:
                        break

                ret, frame = cap.read()
                if not ret:
                    with self.lock:
                        self.latest_error = "Could not read a frame from the camera."
                    time.sleep(0.1)
                    continue

                frame = resize_live_frame(frame)
                display_frame = frame.copy()

                with self.lock:
                    detection = self.latest_detection
                    detection_age = time.time() - self.latest_detection_at

                if detection and detection_age <= LIVE_OVERLAY_TTL:
                    display_frame = draw_live_overlay(display_frame, detection)

                frame_url = encode_frame_to_data_url(display_frame)

                with self.lock:
                    self.latest_raw_frame = frame.copy()
                    self.latest_frame_url = frame_url

                time.sleep(0.02)
        finally:
            cap.release()
            with self.lock:
                self.running = False

    def _prediction_loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break
                frame = None if self.latest_raw_frame is None else self.latest_raw_frame.copy()

            if frame is None:
                time.sleep(0.1)
                continue

            results, faces, _ = analyze_frame_emotions(frame, fast=True, max_faces=1)
            current_emotions = average_emotions(results) if results else None
            now = time.time()

            with self.lock:
                if current_emotions:
                    self.latest_emotions = current_emotions
                    dominant_emotion = max(current_emotions, key=current_emotions.get)
                    self.latest_detection = {
                        "face": tuple(int(v) for v in faces[0]),
                        "emotion": dominant_emotion,
                        "confidence": float(current_emotions[dominant_emotion]),
                    }
                    self.latest_detection_at = now
                elif now - self.latest_detection_at > LIVE_OVERLAY_TTL:
                    self.latest_detection = None

            time.sleep(LIVE_PREDICTION_INTERVAL)

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "frame_url": self.latest_frame_url,
                "emotions": None if self.latest_emotions is None else dict(self.latest_emotions),
                "error": self.latest_error,
            }

    def current_emotions(self):
        with self.lock:
            return None if self.latest_emotions is None else dict(self.latest_emotions)


@st.cache_resource
def get_live_camera_buffers(version=3):
    return {}


def get_live_camera_buffer(prefix):
    buffers = get_live_camera_buffers()
    if not isinstance(buffers.get(prefix), LiveCameraBuffer):
        buffers[prefix] = LiveCameraBuffer()
    return buffers[prefix]


def stop_all_live_cameras():
    for buffer in get_live_camera_buffers().values():
        if hasattr(buffer, "stop"):
            buffer.stop()
    for prefix in ["live_camera", "mixed_live_camera"]:
        for suffix in ["_current_emotions", "_overlay", "_capture_id", "_webcam_error", "_detected_at"]:
            st.session_state.pop(f"{prefix}{suffix}", None)


def encode_frame_to_data_url(frame_bgr):
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_JPEG_QUALITY])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def render_live_status(message, variant):
    st.markdown(
        f'<div class="live-status live-status--{variant}">{message}</div>',
        unsafe_allow_html=True,
    )


@st.fragment
def render_live_detection(prefix, title, show_recommend_button=False, source_label="Live Camera", extra_context=None):
    overlay = st.session_state.get(f"{prefix}_overlay")
    payload = browser_webcam(
        key=f"browser_webcam_{prefix}",
        overlay=overlay,
        width=LIVE_FRAME_WIDTH,
        height=LIVE_FRAME_HEIGHT,
        capture_interval_ms=int(LIVE_PREDICTION_INTERVAL * 1000),
        jpeg_quality=BROWSER_WEBCAM_JPEG_QUALITY,
    )
    process_browser_webcam_payload(prefix, payload)

    webcam_error = st.session_state.get(f"{prefix}_webcam_error")
    if webcam_error:
        render_live_status(webcam_error, "error")

    current_emotions = st.session_state.get(f"{prefix}_current_emotions")
    if current_emotions and show_recommend_button:
        if st.button("Recommend Songs", type="primary", key=f"recommend_{prefix}"):
            stop_all_live_cameras()
            set_recommendation_state(source_label, current_emotions, extra_context=extra_context)
            st.rerun()

    return current_emotions


def run_text_flow():
    text_input = st.text_area(
        "Tell EXPLORE what you are feeling",
        placeholder="Write a sentence, lyric, thought, or mood...",
    )

    if st.button("Analyze Text", type="primary") and text_input:
        with st.spinner("Analyzing text emotion..."):
            emotions = predict_text_emotion(text_input)

        if emotions:
            st.session_state.text_emotions = emotions

    if "text_emotions" in st.session_state:
        dominant_emotion = render_emotion_result(
            "Text emotion analysis",
            st.session_state.text_emotions,
        )
        recommend_button(
            dominant_emotion,
            "recommend_text",
            "Text",
            st.session_state.text_emotions,
            extra_context={"input_excerpt": summarize_text_input(text_input)},
        )


def run_image_flow():
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        return

    image = Image.open(uploaded_file).convert("RGB")
    results, image_with_boxes = predict_image_emotion(image)

    image_col, detected_col = st.columns(2)
    with image_col:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with detected_col:
        st.image(image_with_boxes, caption="Detected face map", use_container_width=True)

    if not results:
        st.error("No face detected in the image")
        return

    face_options = {}
    for i, emotions in enumerate(results):
        dominant_emotion = max(emotions, key=emotions.get)
        face_options[f"Face {i + 1} - {dominant_emotion}"] = i

    selected_face = st.selectbox(
        "Choose whose music you want to recommend",
        list(face_options.keys()),
    )
    selected_index = face_options[selected_face]
    selected_emotions = results[selected_index]
    dominant_emotion = render_emotion_result(
        "Image emotion analysis",
        selected_emotions,
    )
    recommend_button(
        dominant_emotion,
        "recommend_image",
        "Image",
        selected_emotions,
        extra_context={
            "visual_mode": "Image",
            "selected_face": selected_face,
        },
    )


def run_live_camera_flow():
    st.caption("Live Camera now uses a browser-native webcam preview with OpenCV overlay and current live emotion.")
    render_live_detection(
        "live_camera",
        "Live emotion",
        show_recommend_button=True,
        source_label="Live Camera",
    )


def run_video_flow():
    uploaded_video = st.file_uploader(
        "Choose a short video",
        type=["mp4", "mov", "avi", "mkv"],
    )
    if not uploaded_video:
        return

    video_bytes = uploaded_video.getvalue()
    st.video(video_bytes)

    if st.button("Analyze Video", type="primary"):
        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
        with st.spinner("Reading facial emotion from video frames..."):
            emotions = predict_video_emotion(video_bytes, suffix)

        if emotions:
            st.session_state.video_emotions = emotions
        else:
            st.error("No face detected in the sampled video frames")

    if "video_emotions" in st.session_state:
        dominant_emotion = render_emotion_result(
            "Video emotion analysis",
            st.session_state.video_emotions,
        )
        recommend_button(
            dominant_emotion,
            "recommend_video",
            "Video",
            st.session_state.video_emotions,
            extra_context={"visual_mode": "Video"},
        )


def run_mixed_flow():
    text_input = st.text_area(
        "Write the feeling you want to combine",
        placeholder="A line, mood, caption, or thought...",
    )
    visual_mode = st.radio(
        "Add a visual signal",
        ["Image", "Video", "Live Camera"],
        horizontal=True,
    )
    if visual_mode != "Live Camera":
        stop_all_live_cameras()
    text_weight = st.slider("Text influence", 0.0, 1.0, 0.4, 0.1)

    visual_emotions = None
    mixed_video_bytes = None
    mixed_video_suffix = ".mp4"

    if visual_mode == "Image":
        mixed_image = st.file_uploader(
            "Choose an image for Mixed",
            type=["jpg", "jpeg", "png"],
            key="mixed_image",
        )
        if mixed_image:
            image = Image.open(mixed_image).convert("RGB")
            results, image_with_boxes = predict_image_emotion(image)
            image_col, detected_col = st.columns(2)
            with image_col:
                st.image(image, caption="Mixed image input", use_container_width=True)
            with detected_col:
                st.image(image_with_boxes, caption="Mixed face map", use_container_width=True)
            visual_emotions = average_emotions(results)
    elif visual_mode == "Video":
        mixed_video = st.file_uploader(
            "Choose a video for Mixed",
            type=["mp4", "mov", "avi", "mkv"],
            key="mixed_video",
        )
        if mixed_video:
            mixed_video_bytes = mixed_video.getvalue()
            mixed_video_suffix = os.path.splitext(mixed_video.name)[1] or ".mp4"
            st.video(mixed_video_bytes)
    else:
        st.caption("Live camera runs continuously here too. Mixed uses the current live emotion from the browser preview.")
        render_live_detection("mixed_live_camera", "Mixed live emotion", source_label="Mixed Live Camera")

    if st.button("Analyze Mixed", type="primary"):
        if not text_input:
            st.warning("Add text before running Mixed analysis")
            return

        with st.spinner("Blending text and visual emotion..."):
            text_emotions = predict_text_emotion(text_input)

            if visual_mode == "Video" and mixed_video_bytes:
                visual_emotions = predict_video_emotion(mixed_video_bytes, mixed_video_suffix)
            elif visual_mode == "Live Camera":
                visual_emotions = st.session_state.get("mixed_live_camera_current_emotions")

        if not text_emotions:
            return

        if not visual_emotions:
            st.error("No face detected in the visual signal")
            return

        fused_emotions = fuse_emotions(
            text_emotions,
            visual_emotions,
            text_weight,
            1.0 - text_weight,
        )
        st.session_state.mixed_emotions = fused_emotions
        st.session_state.mixed_breakdown = {
            "visual_mode": visual_mode,
            "text_weight": float(text_weight),
            "visual_weight": float(1.0 - text_weight),
            "text_summary": build_emotion_summary(text_emotions, label="Text"),
            "visual_summary": build_emotion_summary(visual_emotions, label=visual_mode),
            "final_summary": build_emotion_summary(fused_emotions, label="Mixed"),
            "text_excerpt": summarize_text_input(text_input),
        }

    if "mixed_emotions" in st.session_state:
        dominant_emotion = render_emotion_result(
            "Mixed emotion analysis",
            st.session_state.mixed_emotions,
        )
        recommend_button(
            dominant_emotion,
            "recommend_mixed",
            "Mixed",
            st.session_state.mixed_emotions,
            extra_context={
                "input_excerpt": summarize_text_input(text_input),
                "mixed_breakdown": st.session_state.get("mixed_breakdown"),
            },
        )


def run():
    render_explore_hero()

    st.markdown(
        '<div class="section-kicker">What describes your emotion the best?</div>',
        unsafe_allow_html=True,
    )
    option = render_mode_selector()

    if option == "Text":
        run_text_flow()
    elif option == "Image":
        run_image_flow()
    elif option == "Video":
        run_video_flow()
    elif option == "Live Camera":
        run_live_camera_flow()
    elif option == "Mixed":
        run_mixed_flow()
