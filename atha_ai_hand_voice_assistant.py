"""
╔══════════════════════════════════════════════════════════════════╗
║           ATHA AI HAND VOICE ASSISTANT                          ║
║  Real-time Hand Gesture Detection + Text-to-Speech              ║
║  Dependencies: opencv-python, mediapipe, numpy, pyttsx3         ║
║                                                                  ║
║  Install:                                                        ║
║    pip install opencv-python mediapipe numpy pyttsx3            ║
║                                                                  ║
║  Run:                                                            ║
║    python atha_ai_hand_voice_assistant.py                       ║
║                                                                  ║
║  Controls:                                                       ║
║    Q  - Quit application                                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import collections

# ─────────────────────────────────────────────────────────────────
# 1. TTS ENGINE – dijalankan di thread terpisah agar tidak blocking
# ─────────────────────────────────────────────────────────────────

class TTSEngine:
    """Wrapper pyttsx3 yang aman dipakai multi-thread."""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 155)       # kecepatan bicara
        self.engine.setProperty('volume', 1.0)     # volume penuh
        self._lock = threading.Lock()
        self._busy = False

    def speak(self, text: str):
        """Jalankan TTS di thread terpisah."""
        def _run():
            with self._lock:
                self._busy = True
                self.engine.say(text)
                self.engine.runAndWait()
                self._busy = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    @property
    def is_busy(self) -> bool:
        return self._busy


# ─────────────────────────────────────────────────────────────────
# 2. GESTURE DETECTOR
# ─────────────────────────────────────────────────────────────────

class GestureDetector:
    """
    Menentukan gesture dari landmark MediaPipe Hands.

    Landmark indeks ujung jari:
        Ibu jari  = 4
        Telunjuk  = 8
        Tengah    = 12
        Manis     = 16
        Kelingking= 20

    Landmark sendi kedua (MCP / PIP):
        Ibu jari  = 3
        Telunjuk  = 6
        Tengah    = 10
        Manis     = 14
        Kelingking= 18
    """

    # ── Finger-tip & second-joint index ──────────────────────────
    TIPS   = [4,  8, 12, 16, 20]
    JOINTS = [3,  6, 10, 14, 18]

    def count_fingers(self, landmarks) -> int:
        """Hitung berapa jari yang terangkat."""
        lm = landmarks.landmark
        fingers = 0

        # Ibu jari: bandingkan sumbu X (kiri/kanan tangan)
        # Jika ujung ibu jari lebih ke pinggir dari sendi keduanya → terangkat
        if lm[self.TIPS[0]].x < lm[self.JOINTS[0]].x:
            fingers += 1

        # Jari lainnya: ujung lebih tinggi (y lebih kecil) dari sendi kedua
        for tip, joint in zip(self.TIPS[1:], self.JOINTS[1:]):
            if lm[tip].y < lm[joint].y:
                fingers += 1

        return fingers

    def get_wrist_x(self, landmarks) -> float:
        """Kembalikan posisi X wrist (titik pangkal telapak)."""
        return landmarks.landmark[0].x


# ─────────────────────────────────────────────────────────────────
# 3. WAVE DETECTOR – deteksi gerakan dadah (kiri-kanan)
# ─────────────────────────────────────────────────────────────────

class WaveDetector:
    """
    Deteksi gerakan dadah dengan mencatat riwayat posisi X tangan.

    Algoritma:
      - Simpan posisi X wrist dalam deque berukuran WINDOW_SIZE.
      - Hitung jumlah kali arah gerakan berbalik (kiri→kanan atau kanan→kiri).
      - Jika ≥ REVERSAL_THRESHOLD berbalik dalam TIME_WINDOW detik → wave!
    """

    WINDOW_SIZE        = 30    # jumlah frame yang disimpan
    REVERSAL_THRESHOLD = 3     # minimal berapa kali balik arah
    MIN_DELTA          = 0.015 # minimum perubahan X yang dianggap gerakan
    TIME_WINDOW        = 1.8   # detik evaluasi

    def __init__(self):
        self.positions = collections.deque(maxlen=self.WINDOW_SIZE)
        self.timestamps = collections.deque(maxlen=self.WINDOW_SIZE)

    def update(self, x: float) -> bool:
        """
        Tambahkan posisi baru.
        Kembalikan True jika terdeteksi gerakan dadah.
        """
        now = time.time()
        self.positions.append(x)
        self.timestamps.append(now)

        # Hanya evaluasi frame dalam TIME_WINDOW terakhir
        cutoff = now - self.TIME_WINDOW
        recent_pos = [p for p, t in zip(self.positions, self.timestamps) if t >= cutoff]

        if len(recent_pos) < 6:
            return False

        # Hitung jumlah pembalikan arah
        reversals = 0
        prev_dir = 0
        for i in range(1, len(recent_pos)):
            delta = recent_pos[i] - recent_pos[i - 1]
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
# 4. UI RENDERER – gambar overlay futuristik di atas frame kamera
# ─────────────────────────────────────────────────────────────────

class UIRenderer:
    """Render overlay UI gaya futuristik / neon hijau."""

    # Palet warna (BGR)
    NEON_GREEN  = (57, 255, 20)
    NEON_CYAN   = (255, 255, 0)
    NEON_WHITE  = (220, 255, 220)
    DIM_GREEN   = (0, 100, 0)
    DARK_BG     = (10, 15, 10)
    ACCENT      = (0, 220, 100)
    ORANGE      = (0, 140, 255)

    FONT        = cv2.FONT_HERSHEY_SIMPLEX
    FONT_MONO   = cv2.FONT_HERSHEY_PLAIN

    def draw_panel(self, frame, x: int, y: int, w: int, h: int,
                   alpha: float = 0.55):
        """Gambar kotak semi-transparan sebagai panel info."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (10, 20, 10), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.NEON_GREEN, 1)

    def draw_title(self, frame):
        """Judul aplikasi di bagian atas layar."""
        h, w = frame.shape[:2]

        # Background strip
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), (5, 15, 5), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        title = "ATHA AI HAND VOICE ASSISTANT"
        tw, _ = cv2.getTextSize(title, self.FONT, 0.65, 2)[0:2]
        tx = (w - tw[0]) // 2
        cv2.putText(frame, title, (tx, 32),
                    self.FONT, 0.65, self.NEON_GREEN, 2, cv2.LINE_AA)

        # Garis dekoratif
        cv2.line(frame, (0, 48), (w, 48), self.NEON_GREEN, 1)

    def draw_scanline(self, frame, frame_count: int):
        """Efek scanline animasi ringan."""
        h, w = frame.shape[:2]
        y = int((frame_count * 3) % h)
        cv2.line(frame, (0, y), (w, y), (0, 60, 0), 1)

    def draw_corner_brackets(self, frame):
        """Bracket sudut layar gaya HUD."""
        h, w = frame.shape[:2]
        s, t = 20, 2  # ukuran bracket, ketebalan
        col = self.NEON_GREEN
        # Kiri atas
        cv2.line(frame, (10, 55),    (10 + s, 55),    col, t)
        cv2.line(frame, (10, 55),    (10, 55 + s),     col, t)
        # Kanan atas
        cv2.line(frame, (w - 10, 55),  (w - 10 - s, 55), col, t)
        cv2.line(frame, (w - 10, 55),  (w - 10, 55 + s), col, t)
        # Kiri bawah
        cv2.line(frame, (10, h - 10),  (10 + s, h - 10), col, t)
        cv2.line(frame, (10, h - 10),  (10, h - 10 - s), col, t)
        # Kanan bawah
        cv2.line(frame, (w - 10, h - 10), (w - 10 - s, h - 10), col, t)
        cv2.line(frame, (w - 10, h - 10), (w - 10, h - 10 - s), col, t)

    def draw_info_panel(self, frame, fingers: int, gesture: str,
                        ai_text: str, fps: float):
        """Panel info di kiri bawah."""
        h, w = frame.shape[:2]
        px, py, pw, ph = 10, h - 170, 310, 160
        self.draw_panel(frame, px, py, pw, ph)

        rows = [
            ("Camera Status", "Active", self.NEON_GREEN),
            ("Fingers Detected", str(fingers), self.NEON_CYAN),
            ("Gesture Detected", gesture if gesture else "---", self.ACCENT),
            ("AI Response",
             (ai_text[:28] + "…") if len(ai_text) > 28 else ai_text,
             self.ORANGE),
            ("FPS", f"{fps:.1f}", self.NEON_WHITE),
        ]

        for i, (label, value, color) in enumerate(rows):
            ry = py + 24 + i * 28
            cv2.putText(frame, f"{label}:", (px + 8, ry),
                        self.FONT, 0.42, self.DIM_GREEN, 1, cv2.LINE_AA)
            cv2.putText(frame, value, (px + 160, ry),
                        self.FONT, 0.45, color, 1, cv2.LINE_AA)

    def draw_finger_count_large(self, frame, fingers: int):
        """Tampilkan angka jari besar di pojok kanan atas."""
        h, w = frame.shape[:2]
        num_str = str(fingers)
        cv2.putText(frame, num_str, (w - 90, 120),
                    self.FONT, 3.5, self.NEON_GREEN, 6, cv2.LINE_AA)
        cv2.putText(frame, "fingers", (w - 100, 148),
                    self.FONT, 0.5, self.DIM_GREEN, 1, cv2.LINE_AA)

    def draw_gesture_badge(self, frame, gesture: str):
        """Badge gesture di bagian tengah bawah."""
        if not gesture:
            return
        h, w = frame.shape[:2]
        text = f"[ {gesture.upper()} ]"
        tw = cv2.getTextSize(text, self.FONT, 0.8, 2)[0][0]
        tx = (w - tw) // 2
        ty = h - 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (tx - 10, ty - 28), (tx + tw + 10, ty + 8),
                      (5, 30, 5), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, text, (tx, ty),
                    self.FONT, 0.8, self.NEON_GREEN, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
# 5. MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────

class HandVoiceApp:
    """Aplikasi utama: loop kamera + deteksi gesture + TTS."""

    # ── Gesture definitions ──────────────────────────────────────
    GESTURES = {
        "WAVE":    {"fingers": 5,  "text": "Hello"},
        "PEACE":   {"fingers": 2,  "text": "Saya Atha"},
        "THREE":   {"fingers": 3,
                    "text": "seorang Software Engineer dan Vibe Coder "
                            "yang senang menciptakan inovasi digital."},
    }

    COOLDOWN = 3.0  # detik jeda antar ucapan

    def __init__(self):
        # MediaPipe
        self.mp_hands    = mp.solutions.hands
        self.mp_draw     = mp.solutions.drawing_utils
        self.mp_styles   = mp.solutions.drawing_styles
        self.hands_model = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        self.detector  = GestureDetector()
        self.wave_det  = WaveDetector()
        self.tts       = TTSEngine()
        self.ui        = UIRenderer()

        # State
        self.last_gesture    = ""
        self.last_speak_time = 0.0
        self.ai_text         = ""
        self.frame_count     = 0
        self.fps             = 0.0
        self._fps_timer      = time.time()
        self._fps_frames     = 0

    # ── Landmark drawing ─────────────────────────────────────────

    def _draw_landmarks(self, frame, landmarks):
        """Gambar landmark dan koneksi jari dengan warna neon hijau."""
        h, w = frame.shape[:2]
        lm = landmarks.landmark

        # Koneksi antar landmark (garis)
        for conn in self.mp_hands.HAND_CONNECTIONS:
            a, b = conn
            x1, y1 = int(lm[a].x * w), int(lm[a].y * h)
            x2, y2 = int(lm[b].x * w), int(lm[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 80), 2)

        # Titik landmark
        for i, pt in enumerate(lm):
            cx, cy = int(pt.x * w), int(pt.y * h)
            # Ujung jari lebih besar & berwarna cyan
            if i in GestureDetector.TIPS:
                cv2.circle(frame, (cx, cy), 7, (255, 255, 0), -1)
                cv2.circle(frame, (cx, cy), 9, (0, 200, 80), 1)
            else:
                cv2.circle(frame, (cx, cy), 4, (0, 255, 100), -1)

    # ── FPS counter ──────────────────────────────────────────────

    def _update_fps(self):
        self._fps_frames += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 0.5:
            self.fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_timer = now

    # ── Gesture → TTS logic ──────────────────────────────────────

    def _handle_gesture(self, gesture_name: str):
        """
        Ucapkan teks jika:
          - gesture baru (berbeda dari sebelumnya), DAN
          - cooldown sudah lewat, DAN
          - TTS tidak sedang sibuk.
        """
        now = time.time()
        if (gesture_name != self.last_gesture
                and now - self.last_speak_time >= self.COOLDOWN
                and not self.tts.is_busy):

            info = self.GESTURES.get(gesture_name)
            if info:
                self.ai_text = info["text"]
                self.tts.speak(self.ai_text)
                self.last_speak_time = now

        self.last_gesture = gesture_name

    # ── Main loop ────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Webcam tidak ditemukan. Pastikan kamera terhubung.")
            return

        # Resolusi kamera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=" * 60)
        print("  ATHA AI HAND VOICE ASSISTANT  —  Tekan Q untuk keluar")
        print("=" * 60)
        print("  Gesture:")
        print("    5 jari + dadah  → 'Hello'")
        print("    2 jari (peace)  → 'Saya Atha'")
        print("    3 jari          → deskripsi profil")
        print("=" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame tidak terbaca, mencoba lagi...")
                continue

            # Mirror (natural selfie view)
            frame = cv2.flip(frame, 1)
            self.frame_count += 1
            self._update_fps()

            # ── Deteksi tangan ───────────────────────────────────
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands_model.process(rgb)

            fingers      = 0
            gesture_name = ""

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]

                # Gambar landmark
                self._draw_landmarks(frame, landmarks)

                # Hitung jari
                fingers = self.detector.count_fingers(landmarks)

                # Posisi wrist untuk deteksi dadah
                wrist_x = self.detector.get_wrist_x(landmarks)

                if fingers == 5:
                    # Cek apakah sedang dadah
                    is_waving = self.wave_det.update(wrist_x)
                    if is_waving:
                        gesture_name = "WAVE"
                    else:
                        gesture_name = ""   # 5 jari diam, tidak ada gesture
                else:
                    # Reset wave buffer jika tangan tidak menunjuk 5 jari
                    self.wave_det.reset()

                    if fingers == 2:
                        gesture_name = "PEACE"
                    elif fingers == 3:
                        gesture_name = "THREE"

                # Mainkan TTS sesuai gesture
                self._handle_gesture(gesture_name)

            else:
                # Tidak ada tangan → reset state
                self.wave_det.reset()
                if self.last_gesture:
                    self.last_gesture = ""

            # ── Render UI ────────────────────────────────────────
            self.ui.draw_scanline(frame, self.frame_count)
            self.ui.draw_title(frame)
            self.ui.draw_corner_brackets(frame)
            self.ui.draw_finger_count_large(frame, fingers)
            self.ui.draw_info_panel(
                frame,
                fingers,
                gesture_name,
                self.ai_text,
                self.fps
            )
            self.ui.draw_gesture_badge(frame, gesture_name)

            cv2.imshow("ATHA AI HAND VOICE ASSISTANT", frame)

            # Tekan Q untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Keluar dari aplikasi...")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands_model.close()


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = HandVoiceApp()
    app.run()
