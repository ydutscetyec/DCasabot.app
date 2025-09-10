import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque
import tensorflow as tf
import os
import pyttsx3
import traceback
from vosk import Model, KaldiRecognizer  # NEW for offline speech recognition
import pyaudio
import json

# ================= CONFIG =================
CAM_INDEX = 0
MIN_DET_CONF = 0.7
MAX_NUM_HANDS = 2

# Static gestures
SMOOTH_WINDOW_STATIC = 8
CONF_THRESHOLD_STATIC = 0.90

# Motion gestures
SMOOTH_WINDOW_DYNAMIC = 3
CONF_THRESHOLD_DYNAMIC = 0.60
FRAMES_PER_SEQUENCE = 50
FEATURES_PER_FRAME = 126  # 63 per hand x 2 hands

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================= PATHS =================
ALPHABET_STATIC_MODEL = r"C:\Users\user\PycharmProjects\PythonProject3\staticgestures_alphabets.h5"
ALPHABET_STATIC_LABELS = r"C:\Users\user\PycharmProjects\PythonProject3\staticgestures_alphabets.pkl"
NUMBER_STATIC_MODEL = r"C:\Users\user\PycharmProjects\PythonProject3\staticgestures_numbers.h5"
NUMBER_STATIC_LABELS = r"C:\Users\user\PycharmProjects\PythonProject3\staticgestures_numbers.pkl"
PHRASES_STATIC_MODEL = r"C:\Users\user\PycharmProjects\PythonProject3\staticgestures_phrases.h5"
PHRASES_STATIC_LABELS = r"C:\Users\user\PycharmProjects\PythonProject3\staticgestures_phrases.pkl"

ALPHABET_OH_MODEL = r"C:\Users\user\PycharmProjects\PythonProject3\oh_alphabets.h5"
ALPHABET_OH_LABELS = r"C:\Users\user\PycharmProjects\PythonProject3\oh_alphabets.npy"
NUMBER_OH_MODEL = r"D:\Competition\D'CASABoT\motion_gestures\oh\numbers\model.h5"
NUMBER_OH_LABELS = r"D:\Competition\D'CASABoT\motion_gestures\oh\numbers\labels.npy"
PHRASES_OH_MODEL = r"C:\Users\user\PycharmProjects\PythonProject3\oh_phrases.h5"
PHRASES_OH_LABELS = r"C:\Users\user\PycharmProjects\PythonProject3\oh_phrases.npy"

ALPHABET_TH_MODEL = r"D:\Competition\D'CASABoT\motion_gestures\th\alphabets\model.h5"
ALPHABET_TH_LABELS = r"D:\Competition\D'CASABoT\motion_gestures\th\alphabets\labels.npy"
NUMBER_TH_MODEL = r"D:\Competition\D'CASABoT\motion_gestures\th\numbers\model.h5"
NUMBER_TH_LABELS = r"D:\Competition\D'CASABoT\motion_gestures\th\numbers\labels.npy"
PHRASES_TH_MODEL = r"C:\Users\user\PycharmProjects\PythonProject3\th_phrases.h5"
PHRASES_TH_LABELS = r"C:\Users\user\PycharmProjects\PythonProject3\th_phrases.npy"

# ================= NEW: SPEECH-TO-SIGN CONFIG (VOSK) =================
VOSK_MODEL_PATH = r"C:\Users\user\PycharmProjects\PythonProject3\vosk-model-small-en-us-0.15"
SIGN_FOLDER = r"D:\Competition\D'CASABoT\speech to sign"

vosk_model = Model(VOSK_MODEL_PATH)
rec = KaldiRecognizer(vosk_model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=8192)
stream.start_stream()

def recognize_offline():
    print("[LISTENING] Speak now...")
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if len(data) == 0:
            continue
        if rec.AcceptWaveform(data):
            result = rec.Result()
            text = json.loads(result)["text"]
            if text:
                print(f"[RECOGNIZED] {text}")
                return text.lower()
            else:
                return ""

def play_sign(name):
    """Return a frame (image or video frame) if available, else None"""
    video_path = os.path.join(SIGN_FOLDER, f"{name}.mp4")
    img_path_png = os.path.join(SIGN_FOLDER, f"{name}.png")
    img_path_jpg = os.path.join(SIGN_FOLDER, f"{name}.jpg")

    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
    elif os.path.exists(img_path_png) or os.path.exists(img_path_jpg):
        img_path = img_path_png if os.path.exists(img_path_png) else img_path_jpg
        frame = cv2.imread(img_path)
        return frame
    return None

def speech_to_sign():
    text = recognize_offline()
    if not text:
        print("[EMPTY] No speech detected.")
        return

    text = text.lower().strip()

    # --- Try phrase first ---
    phrase_frame = play_sign(text)
    if phrase_frame is not None:
        cv2.imshow("Speech-to-Sign", phrase_frame)
        cv2.waitKey(2000)
        return

    # --- Phrase missing: try word by word ---
    words = text.split()
    frames_to_show = []

    for word in words:
        f = play_sign(word)
        if f is not None:
            frames_to_show.append(f)
        else:
            # --- Word missing: fallback to letters ---
            for letter in word:
                lf = play_sign(letter)
                if lf is not None:
                    frames_to_show.append(lf)

    if frames_to_show:
        # show all frames side by side
        max_height = max(f.shape[0] for f in frames_to_show)
        resized_frames = [cv2.resize(f, (int(f.shape[1]*max_height/f.shape[0]), max_height))
                          for f in frames_to_show]
        final_frame = cv2.hconcat(resized_frames)
        cv2.imshow("Speech-to-Sign", final_frame)
        cv2.waitKey(2000)
    else:
        print("[MISSING] No signs available for phrase, words, or letters.")



    words = text.lower().split()  # split phrase into words
    play_sign(words)

    # Normalize filename (remove newlines, lowercase)
    phrase_filename = text.replace("\n", " ").strip().lower()
    phrase_video = os.path.join(SIGN_FOLDER, f"{phrase_filename}.mp4")

    # 1. Try full phrase first
    if os.path.exists(phrase_video):
        print(f"[FOUND] Playing phrase video: {phrase_filename}")
        play_sign(phrase_filename)
        return

    # 2. Otherwise, try word by word
    words = text.split()
    for word in words:
        word_filename = word.strip().lower()
        play_sign(word_filename)


# ================= LOAD MODELS =================
def load_static_model(model_path, labels_path):
    model, labels = None, None
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"[OK] Loaded static model: {model_path}")
        else:
            print(f"[MISSING] Static model file not found: {model_path}")
    except Exception as e:
        print(f"[WARNING] Error loading static model {model_path}: {e}")

    try:
        if os.path.exists(labels_path):
            with open(labels_path, "rb") as f:
                loaded = pickle.load(f)
                if hasattr(loaded, "classes_"):
                    labels = list(loaded.classes_)
                elif isinstance(loaded, (list, tuple, np.ndarray)):
                    labels = list(loaded)
                else:
                    labels = list(loaded)
            print(f"[OK] Loaded static labels: {labels_path}")
        else:
            print(f"[MISSING] Static labels file not found: {labels_path}")
    except Exception as e:
        print(f"[WARNING] Error loading static labels {labels_path}: {e}")

    return model, labels

def load_motion_model(model_path, labels_path):
    model, labels = None, None
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"[OK] Loaded motion model: {model_path}")
        else:
            print(f"[MISSING] Motion model file not found: {model_path}")
    except Exception as e:
        print(f"[WARNING] Error loading motion model {model_path}: {e}")

    try:
        if os.path.exists(labels_path):
            labels = np.load(labels_path, allow_pickle=True)
            labels = list(labels)
            print(f"[OK] Loaded motion labels: {labels_path}")
        else:
            print(f"[MISSING] Motion labels file not found: {labels_path}")
    except Exception as e:
        print(f"[WARNING] Error loading motion labels {labels_path}: {e}")

    return model, labels

# Load (will print status)
alphabet_model_static, alphabet_classes_static = load_static_model(ALPHABET_STATIC_MODEL, ALPHABET_STATIC_LABELS)
number_model_static, number_classes_static = load_static_model(NUMBER_STATIC_MODEL, NUMBER_STATIC_LABELS)
phrases_model_static, phrases_classes_static = load_static_model(PHRASES_STATIC_MODEL, PHRASES_STATIC_LABELS)

alphabet_model_oh, alphabet_classes_oh = load_motion_model(ALPHABET_OH_MODEL, ALPHABET_OH_LABELS)
number_model_oh, number_classes_oh     = load_motion_model(NUMBER_OH_MODEL, NUMBER_OH_LABELS)
phrases_model_oh, phrases_classes_oh   = load_motion_model(PHRASES_OH_MODEL, PHRASES_OH_LABELS)

alphabet_model_th, alphabet_classes_th = load_motion_model(ALPHABET_TH_MODEL, ALPHABET_TH_LABELS)
number_model_th, number_classes_th     = load_motion_model(NUMBER_TH_MODEL, NUMBER_TH_LABELS)
phrases_model_th, phrases_classes_th   = load_motion_model(PHRASES_TH_MODEL, PHRASES_TH_LABELS)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_DET_CONF
)

# ================= FEATURE FUNCTIONS =================
def features_from_hand(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

def normalize_hand(hand_landmarks):
    if hand_landmarks is None:
        return [0.0]*63
    coords = np.array([[lm.x,lm.y,lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords -= wrist
    scale = np.linalg.norm(coords[9]-coords[0])
    if scale>0: coords/=scale
    return coords.flatten().tolist()

def get_ordered_hands(coords_list, labels_list, single_hand_training='Right'):
    left_hand,right_hand = [0.0]*63,[0.0]*63
    for i,label in enumerate(labels_list):
        if label=='Left': left_hand=coords_list[i]
        elif label=='Right': right_hand=coords_list[i]
    if len(coords_list)==1:
        if single_hand_training=='Right' and right_hand==[0.0]*63:
            right_hand=left_hand; left_hand=[0.0]*63
        elif single_hand_training=='Left' and left_hand==[0.0]*63:
            left_hand=right_hand; right_hand=[0.0]*63
    return left_hand,right_hand

# ================= SMOOTHING =================
alphabet_queue = deque(maxlen=SMOOTH_WINDOW_STATIC)
number_queue   = deque(maxlen=SMOOTH_WINDOW_STATIC)
phrases_queue  = deque(maxlen=SMOOTH_WINDOW_STATIC)
dynamic_queue  = deque(maxlen=SMOOTH_WINDOW_DYNAMIC)
frame_window   = deque(maxlen=FRAMES_PER_SEQUENCE)

def smooth_probs(new_probs, queue):
    queue.append(new_probs)
    return np.mean(queue,axis=0)

# ================= UNIFIED MODE =================
current_mode     = "static"    # static / dynamic / speech2sign
current_source   = "static"    # static / oh / th
current_category = "alphabets" # alphabets / numbers / phrases
await_confirmation = False

# ================= SENTENCE BUFFER =================
sentence = []
engine = pyttsx3.init()
last_confirmed = None

# ================= VIDEO CAPTURE =================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print(f"[ERROR] Camera index {CAM_INDEX} not available. Try different index (0,1,2...)")
else:
    print(f"[OK] Camera {CAM_INDEX} opened.")

prev_time=0
fps=0.0

cv2.namedWindow("Unified Gesture Recognition", cv2.WINDOW_NORMAL)

print("[INFO] Controls: M=static, 1=OH, 2=TH | 0=STS, 9=Speech2Sign | A/N/P categories | C=confirm | Q=quit")
print("[INFO] Special gestures: S=space | T=newline | D=done/action")

# ================= MAIN LOOP =================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            blank = np.zeros((480,640,3), dtype=np.uint8)
            cv2.putText(blank, "No camera frame (ret=False)", (10,240), FONT, 0.7, (0,0,255), 2)
            cv2.imshow("Unified Gesture Recognition", blank)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            continue

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = hands.process(rgb)
        except Exception as e:
            print("[ERROR] Mediapipe processing error:", e)
            traceback.print_exc()
            results = None

        coords_list,labels_list = [],[]
        if results and getattr(results, "multi_hand_landmarks", None):
            for i,hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords_list.append(normalize_hand(hand_landmarks))
                labels_list.append(results.multi_handedness[i].classification[0].label)

        display_label = "Waiting..."
        display_conf  = 0.0

        # ---------- SIGN TO SPEECH (original code, unchanged) ----------
        if current_mode=="static" and results and getattr(results, "multi_hand_landmarks", None):
            try:
                feats = features_from_hand(results.multi_hand_landmarks[0])
                if current_category=="alphabets" and alphabet_model_static:
                    probs = alphabet_model_static.predict(np.array(feats).reshape(1,-1),verbose=0)[0]
                    probs = smooth_probs(probs, alphabet_queue)
                    idx = int(np.argmax(probs))
                    if alphabet_classes_static:
                        display_label = alphabet_classes_static[idx]
                    display_conf = float(probs[idx])
                elif current_category=="numbers" and number_model_static:
                    probs = number_model_static.predict(np.array(feats).reshape(1,-1),verbose=0)[0]
                    probs = smooth_probs(probs, number_queue)
                    idx = int(np.argmax(probs))
                    if number_classes_static:
                        display_label = number_classes_static[idx]
                    display_conf = float(probs[idx])
                elif current_category=="phrases" and phrases_model_static:
                    probs = phrases_model_static.predict(np.array(feats).reshape(1,-1),verbose=0)[0]
                    probs = smooth_probs(probs, phrases_queue)
                    idx = int(np.argmax(probs))
                    if phrases_classes_static:
                        display_label = phrases_classes_static[idx]
                    display_conf = float(probs[idx])
                if display_conf<CONF_THRESHOLD_STATIC:
                    display_label="Listening..."
            except Exception as e:
                print("[ERROR] static prediction:", e)
                traceback.print_exc()
                display_label = "prediction error"

        # ---------- DYNAMIC MODE ----------
        if current_mode=="dynamic":
            left_hand,right_hand = get_ordered_hands(coords_list,labels_list)
            combined = left_hand+right_hand
            frame_window.append(combined)
            if len(frame_window)==FRAMES_PER_SEQUENCE and current_source in ["oh","th"]:
                try:
                    if current_category=="alphabets":
                        model = alphabet_model_oh if current_source=="oh" else alphabet_model_th
                        labels = alphabet_classes_oh if current_source=="oh" else alphabet_classes_th
                    elif current_category=="numbers":
                        model = number_model_oh if current_source=="oh" else number_model_th
                        labels = number_classes_oh if current_source=="oh" else number_classes_th
                    elif current_category=="phrases":
                        model = phrases_model_oh if current_source=="oh" else phrases_model_th
                        labels = phrases_classes_oh if current_source=="oh" else phrases_classes_th

                    if model is None or labels is None:
                        display_label="model/labels missing"
                    else:
                        X = np.array(frame_window).reshape(1,FRAMES_PER_SEQUENCE,FEATURES_PER_FRAME).astype(np.float32)
                        probs = model.predict(X,verbose=0)[0]
                        dynamic_queue.append(probs)
                        avg_probs = np.mean(dynamic_queue,axis=0)
                        idx = int(np.argmax(avg_probs))
                        display_label = labels[idx]
                        display_conf = float(avg_probs[idx])
                        if display_conf>=CONF_THRESHOLD_DYNAMIC:
                            await_confirmation=True
                        else:
                            display_label="Listening..."
                except Exception as e:
                    print("[ERROR] dynamic prediction:", e)
                    traceback.print_exc()
                    display_label="prediction error"

        # ---------- FPS ----------
        curr_time = time.time()
        fps = 0.9*fps + 0.1*(1.0/max(curr_time-prev_time,1e-6))
        prev_time = curr_time

        # ---------- DISPLAY ----------
        h,w = frame.shape[:2]
        cv2.rectangle(frame,(0,0),(w,140),(0,0,0),-1)
        cv2.putText(frame,f"Mode:{current_mode.upper()} | Source:{current_source.upper()} | Cat:{current_category.upper()}",
                    (10,25),FONT,0.6,(255,255,255),2)
        cv2.putText(frame,f"Prediction:{display_label} ({display_conf*100:.1f}%)",(10,55),FONT,0.8,(0,255,0),2)
        cv2.putText(frame,f"FPS:{fps:.1f}",(10,85),FONT,0.7,(200,200,0),2)
        if current_mode=="dynamic":
            cv2.putText(frame,"Press 'C' to confirm",(10,110),FONT,0.6,(0,255,255),2)

        y_offset = 130
        for i, line in enumerate("".join(sentence).split("\n")):
            cv2.putText(frame, ("Sentence: " if i == 0 else "          ") + line,
                        (10, y_offset + i*25), FONT, 0.7, (0,200,255), 2)

        cv2.imshow("Unified Gesture Recognition",frame)

        # ---------- KEY HANDLING ----------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            current_mode="static"; current_source="static"
            frame_window.clear(); dynamic_queue.clear(); await_confirmation=False
        elif key == ord('1'):
            current_mode="dynamic"; current_source="oh"
            frame_window.clear(); dynamic_queue.clear(); await_confirmation=False
        elif key == ord('2'):
            current_mode="dynamic"; current_source="th"
            frame_window.clear(); dynamic_queue.clear(); await_confirmation=False
        elif chr(key).upper() in ['A','N','P']:
            selected_category = {'A':'alphabets','N':'numbers','P':'phrases'}[chr(key).upper()]
            current_category = selected_category
            frame_window.clear(); dynamic_queue.clear(); await_confirmation=False

        # ---------- MODE SWITCH ----------
        elif key == ord('0'):
            current_mode = "static"
            print("[MODE] Switched to Sign-to-Speech")
        elif key == ord('9'):
            current_mode = "speech2sign"
            print("[MODE] Switched to Speech-to-Sign")

        # ---------- TEXT BUILD ----------
        elif key == ord('c'):
            if current_mode=="dynamic" and await_confirmation and display_conf>=CONF_THRESHOLD_DYNAMIC:
                sentence.append(display_label); await_confirmation=False; last_confirmed=display_label
            elif current_mode=="static" and display_conf>=CONF_THRESHOLD_STATIC:
                sentence.append(display_label); last_confirmed=display_label
                # ---------- UNDO LAST CONFIRMED ----------
        elif key == ord('x'):
                if sentence:
                    removed = sentence.pop()
                    removed = sentence.pop()
                    print(f"[UNDO] Removed: {removed}")
                    print("👉 Current sentence:", "".join(sentence).replace("\n", " ⏎ "))
                elif current_mode == "speech2sign" and sentence:
                    removed = sentence.pop()
                    print(f"[UNDO] Removed last entry: {removed}")
                else:
                    print("[UNDO] Nothing to remove (sentence empty)")

            # ---------- SPACE (S) ----------
        elif key == ord('s'):
                sentence.append(" ")
                print("[SPACE] Added space")
                print("👉 Current sentence:", "".join(sentence).replace("\n", " ⏎ "))

            # ---------- NEW SENTENCE (T) ----------
        elif key == ord('t'):
                if sentence:
                    sentence.clear()
                    print("[NEW] Previous sentence cleared.")
                else:
                    print("[NEW] Nothing to clear (already empty).")
        elif key == ord('d'):
            if current_mode=="speech2sign":
                speech_to_sign()
            else:
                if sentence:
                    full_sentence="".join(sentence).replace("\n"," ")
                    print(f"[DONE] Speaking sentence: {full_sentence}")
                    import subprocess
                    cmd = f'powershell -Command "Add-Type –AssemblyName System.speech; ' \
                          f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; ' \
                          f'$speak.Speak(\'{full_sentence}\');"'
                    subprocess.Popen(cmd,shell=True)
                else:
                    print("[DONE] No sentence to speak.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    p.terminate()
