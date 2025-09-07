import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model
import pickle
import gdown  # pip install gdown

# Map of models to Google Drive URLs
MODEL_FILES_URLS = {
    "staticgestures.h5": "https://drive.google.com/uc?id=1NMG6IGZ8YbWewzKJ2gUneMGPVbi_UJjc",
    "staticgestures.pkl": "https://drive.google.com/uc?id=1FypzQbs8nZrbXP8HKv_hCHaJEaRLSFW8",
    "single_hand_gesture_model.h5": "https://drive.google.com/uc?id=1bZ_H_Ye9-9l-NLlaZKFbm-rS9RiY9CyQ",
    "single_hand_label_encoder.npy": "https://drive.google.com/uc?id=17Go1UMZ6GLXRDRc0PyOPC5Y8YfNuElV8",
    "two_hand_model.h5": "https://drive.google.com/uc?id=18EAXuiV2sXWsMJ03nDW7NfsLUUjVTFjX",
    "two_hand_label.npy": "https://drive.google.com/uc?id=1SCB6ZPbjjAydl33lSjkkOC5B_hPOSsse"
}


os.makedirs("models", exist_ok=True)

for filename, url in MODEL_FILES_URLS.items():
    path = os.path.join("models", filename)
    if not os.path.exists(path):
        st.info(f"Downloading {filename}...")
        gdown.download(url, path, quiet=False)
# ==============================
# Base directory
# ==============================
BASE_DIR = os.path.dirname(__file__)  # path of current file
MODELS_DIR = os.path.join(BASE_DIR, "models")
# ==============================
# Load Models
# ==============================
# Static gesture model
STATIC_MODEL_PATH = os.path.join(MODELS_DIR, "staticgestures.h5")
STATIC_LABELS_PATH = os.path.join(MODELS_DIR, "staticgestures.pkl")
static_model = load_model(STATIC_MODEL_PATH, safe_mode=False)
with open(STATIC_LABELS_PATH, "rb") as f:
    static_le = pickle.load(f)
static_classes = list(static_le.classes_)
static_input_dim = static_model.inputs[0].shape[-1]

# Motion gesture models
SINGLE_HAND_MODEL = os.path.join(MODELS_DIR, "single_hand_gesture_model.h5")
SINGLE_HAND_LABELS = os.path.join(MODELS_DIR, "single_hand_label_encoder.npy")
TWO_HAND_MODEL = os.path.join(MODELS_DIR, "two_hand_model.h5")
TWO_HAND_LABELS = os.path.join(MODELS_DIR, "two_hand_label.npy")

single_model = load_model(SINGLE_HAND_MODEL)
single_classes = np.load(SINGLE_HAND_LABELS, allow_pickle=True)
two_model = load_model(TWO_HAND_MODEL)
two_classes = np.load(TWO_HAND_LABELS, allow_pickle=True)


# ==============================
# Mediapipe setup
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)

# ==============================
# Helpers
# ==============================
def normalize_hand(hand_landmarks):
    if hand_landmarks is None:
        return [0.0]*63
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords -= wrist
    scale = np.linalg.norm(coords[9] - coords[0])
    if scale > 0:
        coords /= scale
    return coords.flatten().tolist()

def features_from_hand(hand_landmarks):
    feats = []
    for lm in hand_landmarks.landmark:
        feats.extend([lm.x, lm.y, lm.z])
    return feats

# ==============================
# Streamlit UI
# ==============================
st.title("🖐️ FSL Real-time Recognition")
mode = st.sidebar.selectbox("Choose Mode", ["Static Gestures", "Motion Gestures"])
run = st.checkbox("Run Webcam")

FRAME_WINDOW = st.image([])
cap = None

# For motion mode
FRAMES_PER_SEQUENCE = 50
FEATURES_PER_FRAME = 126
frame_window = deque(maxlen=FRAMES_PER_SEQUENCE)

while run:
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Camera not available")
            break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display_label = "Listening..."
    display_conf = 0.0

    if results.multi_hand_landmarks:
        # Draw hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if mode == "Static Gestures":
            # Use first hand only
            feats = features_from_hand(results.multi_hand_landmarks[0])
            if len(feats) == static_input_dim:
                probs = static_model.predict(np.array(feats).reshape(1, -1), verbose=0)[0]
                idx = int(np.argmax(probs))
                display_label = static_classes[idx]
                display_conf = float(probs[idx])

        elif mode == "Motion Gestures":
            # Handle one or two hands
            coords_list = [normalize_hand(h) for h in results.multi_hand_landmarks]
            if len(coords_list) == 1:
                left_hand, right_hand = [0.0]*63, coords_list[0]
            else:
                left_hand, right_hand = coords_list[:2]
            combined = left_hand + right_hand
            frame_window.append(combined)

            if len(frame_window) == FRAMES_PER_SEQUENCE:
                X = np.array(frame_window).reshape(1, FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME).astype(np.float32)
                if len(coords_list) <= 1:
                    probs = single_model.predict(X, verbose=0)[0]
                    classes = single_classes
                else:
                    probs = two_model.predict(X, verbose=0)[0]
                    classes = two_classes

                idx = int(np.argmax(probs))
                display_label = classes[idx]
                display_conf = float(probs[idx])
                frame_window.clear()

    # Overlay
    cv2.putText(frame, f"{display_label} {display_conf*100:.1f}%",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
hands.close()
