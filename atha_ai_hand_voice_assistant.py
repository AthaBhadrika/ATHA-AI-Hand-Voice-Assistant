"""
ATHA AI HAND VOICE ASSISTANT
Real-time Hand Gesture Detection + Text-to-Speech
Compatible: MediaPipe 0.10+, Python 3.11

Install:
    py -3.11 -m pip install opencv-python mediapipe numpy pyttsx3

Run:
    py -3.11 atha_ai_hand_voice_assistant.py

Controls:
    Q - Quit

NOTE: Saat pertama dijalankan, program otomatis download model
      MediaPipe sekitar 5MB. Pastikan ada koneksi internet.
"""

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import collections
import urllib.request
import os

# ─────────────────────────────────────────────────────────────────
# 0. AUTO-DOWNLOAD MODEL MEDIAPIPE (sekali saja ~5MB)
# ─────────────────────────────────────────────────────────────────

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    """Download model jika belum ada di folder yang sama."""
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Mendownload model MediaPipe... (sekali saja ~5MB)")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[INFO] Model berhasil didownload.")
        except Exception as e:
            print(f"[ERROR] Gagal download model: {e}")
            print(f"  Download manual: {MODEL_URL}")
            print(f"  Simpan sebagai : {MODEL_PATH}")
            exit(1)


# ─────────────────────────────────────────────────────────────────
# 1. TTS ENGINE – thread terpisah agar kamera tidak freeze
# ─────────────────────────────────────────────────────────────────

class TTSEngine:
    """Wrapper pyttsx3 thread-safe."""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 155)
        self.engine.setProperty('volume', 1.0)
        self._lock = threading.Lock()
        self._busy = False

    def speak(self, text: str):
        def _run():
            with self._lock:
                self._busy = True
                self.engine.say(text)
                self.engine.runAndWait()
                self._busy = False
        threading.Thread(target=_run, daemon=True).start()

    @property
    def is_busy(self) -> bool:
        return self._busy


# ─────────────────────────────────────────────────────────────────
# 2. GESTURE DETECTOR – hitung jari dari landmark
# ─────────────────────────────────────────────────────────────────

class GestureDetector:
    """Hitung jari terangkat dari landmark MediaPipe Tasks."""

    TIPS   = [4,  8, 12, 16, 20]  # ujung jari
    JOINTS = [3,  6, 10, 14, 18]  # sendi kedua

    def count_fingers(self, landmarks) -> int:
        fingers = 0
        # Ibu jari: bandingkan sumbu X
        if landmarks[self.TIPS[0]].x < landmarks[self.JOINTS[0]].x:
            fingers += 1
        # 4 jari lain: ujung lebih tinggi dari sendi kedua
        for tip, joint in zip(self.TIPS[1:], self.JOINTS[1:]):
            if landmarks[tip].y < landmarks[joint].y:
                fingers += 1
        return fingers

    def get_wrist_x(self, landmarks) -> float:
        return landmarks[0].x


# ─────────────────────────────────────────────────────────────────
# 3. WAVE DETECTOR – deteksi gerakan dadah kiri-kanan
# ─────────────────────────────────────────────────────────────────

class WaveDetector:
    """Deteksi wave dari riwayat posisi X wrist."""

    WINDOW_SIZE        = 30
    REVERSAL_THRESHOLD = 3
    MIN_DELTA          = 0.015
    TIME_WINDOW        = 1.8

    def __init__(self):
        self.positions  = collections.deque(maxlen=self.WINDOW_SIZE)
        self.timestamps = collections.deque(maxlen=self.WINDOW_SIZE)

    def update(self, x: float) -> bool:
        now = time.time()
        self.positions.append(x)
        self.timestamps.append(now)

        cutoff = now - self.TIME_WINDOW
        recent = [p for p, t in zip(self.positions, self.timestamps)
                  if t >= cutoff]

        if len(recent) < 6:
            return False

        reversals, prev_dir = 0, 0
        for i in range(1, len(recent)):
            delta = recent[i] - recent[i - 1]
            if abs(delta) < self.MIN_DELTA:
                continue
            cur_dir = 1 if delta > 0 else -1
            if prev_dir != 0 and cur_dir != prev_dir:
                reversals += 1
            prev_dir = cur_dir

        return reversals >= self.REVERSAL_THRESHOLD

    def reset(self):
        self.positions.clear()
        self.timestamps.clear()


# ─────────────────────────────────────────────────────────────────
# 4. UI RENDERER – overlay neon hijau futuristik
# ─────────────────────────────────────────────────────────────────

class UIRenderer:
    NEON_GREEN = (57, 255, 20)
    NEON_CYAN  = (255, 255, 0)
    NEON_WHITE = (220, 255, 220)
    DIM_GREEN  = (0, 120, 0)
    ACCENT     = (0, 220, 100)
    ORANGE     = (0, 140, 255)
    FONT       = cv2.FONT_HERSHEY_SIMPLEX

    def _panel(self, frame, x, y, w, h, alpha=0.55):
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x+w, y+h), (10, 20, 10), -1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x+w, y+h), self.NEON_GREEN, 1)

    def draw_title(self, frame):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 48), (5, 15, 5), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
        title = "ATHA AI HAND VOICE ASSISTANT"
        tw = cv2.getTextSize(title, self.FONT, 0.65, 2)[0][0]
        cv2.putText(frame, title, ((w-tw)//2, 32),
                    self.FONT, 0.65, self.NEON_GREEN, 2, cv2.LINE_AA)
        cv2.line(frame, (0, 48), (w, 48), self.NEON_GREEN, 1)

    def draw_scanline(self, frame, fc):
        h, w = frame.shape[:2]
        y = int((fc * 3) % h)
        cv2.line(frame, (0, y), (w, y), (0, 60, 0), 1)

    def draw_corners(self, frame):
        h, w = frame.shape[:2]
        s, t, c = 20, 2, self.NEON_GREEN
        cv2.line(frame, (10, 55),    (10+s, 55),    c, t)
        cv2.line(frame, (10, 55),    (10, 55+s),    c, t)
        cv2.line(frame, (w-10, 55),  (w-10-s, 55),  c, t)
        cv2.line(frame, (w-10, 55),  (w-10, 55+s),  c, t)
        cv2.line(frame, (10, h-10),  (10+s, h-10),  c, t)
        cv2.line(frame, (10, h-10),  (10, h-10-s),  c, t)
        cv2.line(frame, (w-10, h-10),(w-10-s, h-10),c, t)
        cv2.line(frame, (w-10, h-10),(w-10, h-10-s),c, t)

    def draw_info_panel(self, frame, fingers, gesture, ai_text, fps):
        h, w = frame.shape[:2]
        px, py, pw, ph = 10, h-175, 315, 165
        self._panel(frame, px, py, pw, ph)
        rows = [
            ("Camera Status",    "Active",                          self.NEON_GREEN),
            ("Fingers Detected", str(fingers),                      self.NEON_CYAN),
            ("Gesture Detected", gesture if gesture else "---",     self.ACCENT),
            ("AI Response",
             (ai_text[:28]+"...") if len(ai_text)>28 else ai_text, self.ORANGE),
            ("FPS",              f"{fps:.1f}",                      self.NEON_WHITE),
        ]
        for i, (label, value, color) in enumerate(rows):
            ry = py + 26 + i*30
            cv2.putText(frame, f"{label}:", (px+8, ry),
                        self.FONT, 0.42, self.DIM_GREEN, 1, cv2.LINE_AA)
            cv2.putText(frame, value, (px+165, ry),
                        self.FONT, 0.44, color, 1, cv2.LINE_AA)

    def draw_finger_count(self, frame, fingers):
        h, w = frame.shape[:2]
        cv2.putText(frame, str(fingers), (w-90, 120),
                    self.FONT, 3.5, self.NEON_GREEN, 6, cv2.LINE_AA)
        cv2.putText(frame, "fingers", (w-100, 148),
                    self.FONT, 0.5, self.DIM_GREEN, 1, cv2.LINE_AA)

    def draw_gesture_badge(self, frame, gesture):
        if not gesture:
            return
        h, w = frame.shape[:2]
        text = f"[ {gesture.upper()} ]"
        tw = cv2.getTextSize(text, self.FONT, 0.8, 2)[0][0]
        tx, ty = (w-tw)//2, h-20
        ov = frame.copy()
        cv2.rectangle(ov, (tx-10, ty-28), (tx+tw+10, ty+8), (5,30,5), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, text, (tx, ty),
                    self.FONT, 0.8, self.NEON_GREEN, 2, cv2.LINE_AA)

    def draw_landmarks(self, frame, landmarks):
        """Gambar titik landmark & koneksi neon hijau."""
        h, w = frame.shape[:2]
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        for a, b in connections:
            x1,y1 = int(landmarks[a].x*w), int(landmarks[a].y*h)
            x2,y2 = int(landmarks[b].x*w), int(landmarks[b].y*h)
            cv2.line(frame, (x1,y1), (x2,y2), (0,200,80), 2)

        tips = {4, 8, 12, 16, 20}
        for i, pt in enumerate(landmarks):
            cx, cy = int(pt.x*w), int(pt.y*h)
            if i in tips:
                cv2.circle(frame, (cx,cy), 7, (255,255,0), -1)
                cv2.circle(frame, (cx,cy), 9, (0,200,80), 1)
            else:
                cv2.circle(frame, (cx,cy), 4, (0,255,100), -1)


# ─────────────────────────────────────────────────────────────────
# 5. MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────

class HandVoiceApp:
    GESTURES = {
        "WAVE":  {"text": "Hello"},
        "PEACE": {"text": "Saya Atha"},
        "THREE": {"text": (
            "seorang Software Engineer dan Vibe Coder "
            "yang senang menciptakan inovasi digital."
        )},
    }
    COOLDOWN = 3.0

    def __init__(self):
        # Import MediaPipe Tasks API (0.10+)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)

        self.detector        = GestureDetector()
        self.wave_det        = WaveDetector()
        self.tts             = TTSEngine()
        self.ui              = UIRenderer()

        self.last_gesture    = ""
        self.last_speak_time = 0.0
        self.ai_text         = ""
        self.frame_count     = 0
        self.fps             = 0.0
        self._fps_timer      = time.time()
        self._fps_frames     = 0

    def _update_fps(self):
        self._fps_frames += 1
        now = time.time()
        if now - self._fps_timer >= 0.5:
            self.fps = self._fps_frames / (now - self._fps_timer)
            self._fps_frames = 0
            self._fps_timer  = now

    def _handle_gesture(self, gesture_name: str):
        now = time.time()
        if (gesture_name
                and gesture_name != self.last_gesture
                and now - self.last_speak_time >= self.COOLDOWN
                and not self.tts.is_busy):
            info = self.GESTURES.get(gesture_name)
            if info:
                self.ai_text = info["text"]
                self.tts.speak(self.ai_text)
                self.last_speak_time = now
        self.last_gesture = gesture_name

    def run(self):
        from mediapipe import Image, ImageFormat

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Webcam tidak ditemukan.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=" * 60)
        print("  ATHA AI HAND VOICE ASSISTANT  —  Tekan Q untuk keluar")
        print("=" * 60)
        print("  Gesture:")
        print("    5 jari + dadah  ->  Hello")
        print("    2 jari (peace)  ->  Saya Atha")
        print("    3 jari          ->  deskripsi profil")
        print("=" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            self.frame_count += 1
            self._update_fps()

            # Konversi ke MediaPipe Image
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms    = int(time.time() * 1000)
            result   = self.landmarker.detect_for_video(mp_image, ts_ms)

            fingers      = 0
            gesture_name = ""

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                self.ui.draw_landmarks(frame, lm)

                fingers = self.detector.count_fingers(lm)
                wrist_x = self.detector.get_wrist_x(lm)

                if fingers == 5:
                    if self.wave_det.update(wrist_x):
                        gesture_name = "WAVE"
                else:
                    self.wave_det.reset()
                    if fingers == 2:
                        gesture_name = "PEACE"
                    elif fingers == 3:
                        gesture_name = "THREE"

                self._handle_gesture(gesture_name)
            else:
                self.wave_det.reset()
                self.last_gesture = ""

            # Render UI
            self.ui.draw_scanline(frame, self.frame_count)
            self.ui.draw_title(frame)
            self.ui.draw_corners(frame)
            self.ui.draw_finger_count(frame, fingers)
            self.ui.draw_info_panel(frame, fingers, gesture_name,
                                    self.ai_text, self.fps)
            self.ui.draw_gesture_badge(frame, gesture_name)

            cv2.imshow("ATHA AI HAND VOICE ASSISTANT", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Keluar dari aplikasi.")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_model()   # Download model jika belum ada
    app = HandVoiceApp()
    app.run()
