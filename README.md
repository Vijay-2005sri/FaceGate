# 🔐 FaceGate — AI-Powered Face Authentication Lock Screen

FaceGate is a Windows lock-screen application that uses real-time face recognition to authenticate users. It features an Iron Man–inspired HUD overlay, voice feedback via SAPI5 TTS, and OS-level keyboard blocking to prevent unauthorized access.

---

## ✨ Features

- **Real-time face recognition** using `face_recognition` + `dlib`
- **Iron Man HUD overlay** built with PyQt6 — rotating rings, scanning grid, nebula background
- **Voice feedback** — SAPI5 text-to-speech confirms identity on successful verification
- **Full lock-screen mode** — frameless, always-on-top, fullscreen with hidden cursor
- **OS-level keyboard hook** — blocks Alt+F4, Alt+Tab, Win key, Ctrl+Esc while locked
- **Emergency termination** — press `Ctrl+Shift+Q` to force-quit the app at any time
- **Multi-face registration** — register multiple face samples for improved accuracy
- **PyInstaller-ready** — ships with `.spec` files for building standalone `.exe` files

---

## 📁 Project Structure

```
FaceGate/
├── iron_hud.py              # Main HUD lock-screen application
├── face_recognition_thread.py # Background face recognition worker
├── register1.py             # Face registration script (capture samples)
├── verify.py                # Standalone face verification (no HUD)
├── iron_hud.spec            # PyInstaller spec for iron_hud.exe
├── register1.spec           # PyInstaller spec for register1.exe
├── verify.spec              # PyInstaller spec for verify.exe
├── face_data.pkl            # Stored face encodings (generated at runtime)
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (Windows)
- **Webcam** connected and accessible
- Required packages:

```bash
pip install opencv-python face_recognition dlib numpy pyttsx3 PyQt6
```

> **Note:** `dlib` may require Visual Studio Build Tools. See [dlib installation guide](http://dlib.net/compile.html).

### 1. Register Your Face

Run the registration script to capture face samples:

```bash
python register1.py
```

- Press **S** to capture a sample (3 samples required)
- Press **Q** to quit early
- Ensure only **one face** is visible when capturing

### 2. Launch the Lock Screen

```bash
python iron_hud.py
```

The HUD will go fullscreen and lock your system. It will:
1. Scan for faces via your webcam
2. Compare against registered encodings
3. On match → speak "Identity verified. Welcome back sir." → auto-close

### 3. Standalone Verification (No HUD)

```bash
python verify.py
```

A simpler OpenCV window that verifies faces without the HUD overlay.

---

## ⌨️ Key Bindings

| Key Combo | Action |
|---|---|
| `Ctrl+Shift+Q` | **Force-terminate** the app (bypasses all blocks) |
| Alt+F4 | ❌ Blocked while locked |
| Alt+Tab | ❌ Blocked while locked |
| Win key | ❌ Blocked while locked |
| Ctrl+Esc | ❌ Blocked while locked |

---

## 🔧 Building Executables

Use PyInstaller with the included `.spec` files:

```bash
pyinstaller iron_hud.spec --noconfirm
pyinstaller register1.spec --noconfirm
pyinstaller verify.spec --noconfirm
```

Executables will be output to the `dist/` folder.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              iron_hud.py (Main)             │
│                                             │
│  ┌─────────────┐    ┌────────────────────┐  │
│  │  PyQt6 HUD  │◄───│  FaceRecognition   │  │
│  │  (Fullscreen │    │  Thread (Background)│  │
│  │   Overlay)  │    └────────────────────┘  │
│  └─────────────┘                            │
│                                             │
│  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Keyboard Hook   │  │ SAPI5 TTS Voice │   │
│  │ (Low-level, OS) │  │ (Threaded)      │   │
│  └─────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────┘
```

- **IronHUD** — Main PyQt6 window handling rendering, camera feed, and state management
- **FaceRecognitionThread** — QThread that runs face detection/comparison off the main thread
- **Keyboard Hook** — Windows low-level hook (`WH_KEYBOARD_LL`) to intercept and block system keys
- **TTS Engine** — `pyttsx3` with SAPI5 backend, runs in a daemon thread to avoid blocking the UI

---

## 📝 Recent Changes

### Termination Key (Ctrl+Shift+Q)
- Added `Ctrl+Shift+Q` as an emergency termination key combo
- Works at the OS-level keyboard hook (bypasses all blocks)
- Also handled at the Qt `keyPressEvent` level as a fallback
- Sets state to `TERMINATED` which allows `closeEvent` to proceed with graceful cleanup

---

## 📄 License

This project is for educational and personal use.

---

## 👤 Author

**Vijay** — [GitHub](https://github.com/Vijay-2005sri)
