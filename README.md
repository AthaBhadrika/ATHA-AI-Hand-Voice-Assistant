# 🤖 ATHA AI Hand Voice Assistant

> Real-time hand gesture detection with Text-to-Speech using OpenCV, MediaPipe, and pyttsx3.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Fitur

- 🖐️ Deteksi tangan real-time via webcam
- 🔢 Hitung jumlah jari yang terangkat
- 👋 Deteksi gesture dadah (wave)
- 🔊 Text-to-Speech otomatis (offline, tanpa internet)
- 🎨 UI overlay futuristik neon hijau
- 📊 Tampilan FPS live
- ⚡ Cooldown 3 detik agar suara tidak berulang

---

## 🎮 Gesture & Suara

| Gesture | Deskripsi | AI Mengucapkan |
|---|---|---|
| 👋 **Wave** | 5 jari terbuka + goyang kiri-kanan | *"Hello"* |
| ✌️ **Peace** | 2 jari (telunjuk + tengah) | *"Saya Atha"* |
| 🤟 **Three** | 3 jari terbuka | *"seorang Software Engineer dan Vibe Coder yang senang menciptakan inovasi digital."* |

---

## 🖥️ Cara Menjalankan di Windows / Linux (Laptop / PC)

### 1. Pastikan Python 3.10+ sudah terinstall

```bash
python --version
# atau
python3 --version
```

Jika belum, download di → https://www.python.org/downloads/

---

### 2. Clone repository ini

```bash
git clone https://github.com/username/atha-ai-hand-voice-assistant.git
cd atha-ai-hand-voice-assistant
```

---

### 3. (Opsional) Buat virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install semua dependensi

```bash
pip install opencv-python mediapipe numpy pyttsx3
```

**Khusus Linux**, install juga dependensi sistem untuk audio TTS:
```bash
sudo apt update
sudo apt install espeak ffmpeg libespeak1 -y
```

---

### 5. Jalankan program

**Windows:**
```bash
python atha_ai_hand_voice_assistant.py
```

**Linux:**
```bash
python3 atha_ai_hand_voice_assistant.py
```

---

### 6. Kontrol aplikasi

| Tombol | Fungsi |
|---|---|
| `Q` | Keluar dari aplikasi |

---

## 📱 Cara Menjalankan di HP Android (Termux)

> ⚠️ **Catatan Penting:** MediaPipe dan OpenCV dengan GUI (jendela kamera) **tidak bisa berjalan langsung di Termux** karena Termux tidak punya akses display dan kamera secara native. Gunakan salah satu solusi berikut:

---

### ✅ Solusi A — Termux + UserLAnd (Recommended)

UserLAnd adalah app untuk menjalankan Linux lengkap di Android.

**Langkah:**

1. Install **Termux** dari F-Droid → https://f-droid.org/packages/com.termux/
2. Install **UserLAnd** dari Play Store → https://play.google.com/store/apps/details?id=tech.ula
3. Di UserLAnd, pilih distro **Ubuntu**
4. Setelah masuk ke Ubuntu, jalankan:

```bash
# Update sistem
sudo apt update && sudo apt upgrade -y

# Install Python dan dependensi sistem
sudo apt install python3 python3-pip python3-venv \
  espeak libespeak1 ffmpeg \
  libgl1 libglib2.0-0 -y

# Install dependensi Python
pip3 install opencv-python mediapipe numpy pyttsx3

# Clone repo
git clone https://github.com/username/atha-ai-hand-voice-assistant.git
cd atha-ai-hand-voice-assistant

# Jalankan
python3 atha_ai_hand_voice_assistant.py
```

> Untuk tampilan GUI di Android, install juga **VNC Viewer** dan set up XFCE desktop di UserLAnd.

---

### ✅ Solusi B — Termux Murni (Tanpa GUI / Headless Mode)

Jika hanya ingin menjalankan logika gesture tanpa tampilan kamera di Termux:

```bash
# Update Termux
pkg update && pkg upgrade -y

# Install Python
pkg install python python-pip -y

# Install dependensi (versi headless, tanpa GUI)
pip install opencv-python-headless numpy

# Catatan: mediapipe dan pyttsx3 mungkin tidak tersedia
# di Termux ARM. Gunakan solusi A untuk pengalaman penuh.
```

> ⚠️ Untuk fitur lengkap (kamera + TTS + MediaPipe), **Solusi A (UserLAnd)** jauh lebih disarankan.

---

### ✅ Solusi C — Remote ke PC lewat SSH (Termux sebagai Terminal)

Jalankan program di PC/laptop, kontrol dari HP via SSH:

**Di PC (server):**
```bash
# Install SSH server (Linux)
sudo apt install openssh-server -y
sudo systemctl start ssh

# Jalankan program di background
python3 atha_ai_hand_voice_assistant.py
```

**Di HP (Termux sebagai client):**
```bash
pkg install openssh -y
ssh username@IP_PC_KAMU
# lalu jalankan program dari sana
```

---

## 📦 Dependensi

| Library | Versi | Fungsi |
|---|---|---|
| `opencv-python` | ≥ 4.8 | Capture kamera & render frame |
| `mediapipe` | ≥ 0.10 | Deteksi landmark tangan |
| `numpy` | ≥ 1.24 | Operasi array frame |
| `pyttsx3` | ≥ 2.90 | Text-to-Speech offline |

---

## 🗂️ Struktur Kode

```
atha_ai_hand_voice_assistant.py
│
├── TTSEngine          # TTS pyttsx3 thread-safe
├── GestureDetector    # Hitung jari dari landmark MediaPipe
├── WaveDetector       # Deteksi gerakan dadah (kiri-kanan)
├── UIRenderer         # Overlay UI neon hijau futuristik
└── HandVoiceApp       # Main app loop
```

---

## ❓ Troubleshooting

**Webcam tidak terdeteksi:**
```bash
# Ganti index kamera (0, 1, 2, dst)
# Edit baris ini di kode:
cap = cv2.VideoCapture(0)  # coba ganti 0 → 1
```

**TTS tidak bersuara di Linux:**
```bash
sudo apt install espeak libespeak1 -y
```

**Error `mediapipe` tidak ditemukan:**
```bash
pip install --upgrade mediapipe
```

**FPS terlalu rendah:**
- Turunkan resolusi kamera di kode:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # dari 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # dari 720
```

---

## 👨‍💻 Author

**Atha** — Software Engineer & Vibe Coder  
*"Senang menciptakan inovasi digital."*

---

## 📄 License

MIT License — bebas digunakan dan dimodifikasi.
