const shell = document.getElementById("shell");
const video = document.getElementById("video");
const overlayCanvas = document.getElementById("overlay");
const placeholder = document.getElementById("placeholder");
const captureCanvas = document.createElement("canvas");
const captureContext = captureCanvas.getContext("2d");
const overlayContext = overlayCanvas.getContext("2d");

const state = {
  args: null,
  stream: null,
  startPromise: null,
  captureTimer: null,
  captureId: 0,
  lastError: "",
  lastOverlayKey: "",
  sizeReady: false,
};

function setPlaceholder(message, hidden = false) {
  placeholder.textContent = message;
  placeholder.classList.toggle("hidden", hidden);
}

function sendValue(value) {
  Streamlit.setComponentValue(value);
}

function currentVideoWidth() {
  return video.videoWidth || state.args?.width || 480;
}

function currentVideoHeight() {
  return video.videoHeight || state.args?.height || 360;
}

function syncCanvasSize() {
  const width = currentVideoWidth();
  const height = currentVideoHeight();
  overlayCanvas.width = width;
  overlayCanvas.height = height;
  shell.style.aspectRatio = `${width} / ${height}`;
  Streamlit.setFrameHeight(shell.getBoundingClientRect().height);
}

function drawOverlay(overlay) {
  if (!overlayCanvas.width || !overlayCanvas.height) {
    syncCanvasSize();
  }

  overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (!overlay || !overlay.face) {
    return;
  }

  const [x, y, w, h] = overlay.face;
  const sourceWidth = overlay.source_width || overlayCanvas.width;
  const sourceHeight = overlay.source_height || overlayCanvas.height;
  const scaleX = overlayCanvas.width / sourceWidth;
  const scaleY = overlayCanvas.height / sourceHeight;

  const left = x * scaleX;
  const top = y * scaleY;
  const width = w * scaleX;
  const height = h * scaleY;
  const label = `${overlay.emotion} ${Math.round((overlay.confidence || 0) * 100)}%`;

  overlayContext.strokeStyle = "#4fa8c9";
  overlayContext.lineWidth = 2;
  overlayContext.strokeRect(left, top, width, height);

  overlayContext.font = "700 18px Segoe UI, sans-serif";
  const metrics = overlayContext.measureText(label);
  const textWidth = metrics.width;
  const textHeight = 20;

  let labelTop = top - textHeight - 14;
  if (labelTop < 8) {
    labelTop = top + 8;
  }

  overlayContext.fillStyle = "rgba(31, 39, 22, 0.92)";
  overlayContext.fillRect(left, labelTop, textWidth + 22, textHeight + 10);
  overlayContext.fillStyle = "#f8f4e8";
  overlayContext.fillText(label, left + 11, labelTop + 22);
}

function overlayKey(overlay) {
  if (!overlay || !overlay.face) {
    return "";
  }
  return JSON.stringify(overlay);
}

function captureFrame() {
  if (!state.stream || video.readyState < 2) {
    return;
  }

  const width = currentVideoWidth();
  const height = currentVideoHeight();
  captureCanvas.width = width;
  captureCanvas.height = height;
  captureContext.drawImage(video, 0, 0, width, height);

  state.captureId += 1;
  sendValue({
    capture_id: state.captureId,
    frame_data_url: captureCanvas.toDataURL("image/jpeg", state.args?.jpeg_quality ?? 0.6),
    width,
    height,
    ts: Date.now(),
  });
}

function startCaptureLoop() {
  const interval = state.args?.capture_interval_ms || 450;
  if (state.captureTimer) {
    window.clearInterval(state.captureTimer);
  }
  state.captureTimer = window.setInterval(captureFrame, interval);
}

async function ensureStream() {
  if (state.stream || state.startPromise) {
    return state.startPromise;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const message = "This browser does not support webcam access.";
    if (state.lastError !== message) {
      state.lastError = message;
      sendValue({ error: message, ts: Date.now() });
    }
    setPlaceholder(message, false);
    return null;
  }

  state.startPromise = navigator.mediaDevices
    .getUserMedia({
      audio: false,
      video: {
        facingMode: state.args?.facing_mode || "user",
        width: { ideal: state.args?.width || 480 },
        height: { ideal: state.args?.height || 360 },
      },
    })
    .then(async (stream) => {
      state.stream = stream;
      video.srcObject = stream;
      await video.play();
      syncCanvasSize();
      drawOverlay(state.args?.overlay);
      setPlaceholder("", true);
      startCaptureLoop();
      return stream;
    })
    .catch((error) => {
      const message = `Could not access the camera: ${error.message}`;
      if (state.lastError !== message) {
        state.lastError = message;
        sendValue({ error: message, ts: Date.now() });
      }
      setPlaceholder(message, false);
      return null;
    })
    .finally(() => {
      state.startPromise = null;
    });

  return state.startPromise;
}

function onRender(event) {
  state.args = event.detail.args;
  setPlaceholder("Allow camera access to start the live preview.", !state.stream ? false : true);

  const nextOverlayKey = overlayKey(state.args.overlay);
  if (nextOverlayKey !== state.lastOverlayKey) {
    state.lastOverlayKey = nextOverlayKey;
    drawOverlay(state.args.overlay);
  }

  ensureStream();
  Streamlit.setFrameHeight(shell.getBoundingClientRect().height);
}

video.addEventListener("loadedmetadata", () => {
  syncCanvasSize();
  drawOverlay(state.args?.overlay);
  setPlaceholder("", true);
});

window.addEventListener("resize", () => {
  syncCanvasSize();
  drawOverlay(state.args?.overlay);
});

window.addEventListener("beforeunload", () => {
  if (state.captureTimer) {
    window.clearInterval(state.captureTimer);
  }
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
  }
});

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
