# FaceGate — AI-Powered Face Authentication Lock Screen

FaceGate is a secure, Windows-based lock screen application that utilizes real-time facial recognition to authenticate users. Designed as a seamless biometric security layer, it prevents unauthorized access by employing OS-level keyboard blocking and a full-screen, always-on-top UI overlay, unlocking only when a registered face is detected.

---

## 🚀 Key Features

- **Real-Time Facial Recognition**: Uses `face_recognition` and `dlib` for robust and accurate biometric identification.
- **Biometric Registration**: Supports registering multiple facial samples for improved accuracy and consistency.
- **OS-Level Keyboard Hook**: Intercepts and blocks critical system shortcuts (e.g., `Alt+F4`, `Alt+Tab`, `Win` key, `Ctrl+Esc`) to prevent unauthorized bypass of the lock screen.
- **Seamless UI Overlay**: Built with PyQt6, providing a frameless, always-on-top interface that completely conceals the underlying desktop environment.
- **Voice Feedback**: Integrates SAPI5 text-to-speech for audible confirmation upon successful authentication.
- **Standalone Executables**: Configured with PyInstaller for building ready-to-use `.exe` files.

---

## 🛠️ Technology Stack

- **Language**: Python 3.11+
- **Computer Vision**: OpenCV (`opencv-python`)
- **Machine Learning / Biometrics**: `face_recognition`, `dlib`, `numpy`
- **Graphical User Interface**: PyQt6
- **System Integration**: Low-level Windows API (for keyboard hooks), `pyttsx3` (TTS)
- **Deployment**: PyInstaller

---

## ⚙️ How It Works

1. **Registration**: The user runs the registration script to capture several face samples via the webcam. These samples are converted into mathematical facial encodings and securely stored.
2. **Locking the System**: Upon launching the main application, a fullscreen overlay is displayed. A background keyboard hook is established to ignore common escape keystrokes.
3. **Continuous Scanning**: A background thread continuously captures frames from the webcam and scans for faces.
4. **Authentication**: Detected faces are compared against the stored encodings. If a match meets the predefined confidence threshold, the application provides voice confirmation ("Identity verified"), lifts the keyboard hooks, and closes the overlay, granting access to the system.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (Windows)
- **Webcam** connected and accessible

Install required dependencies:

```bash
pip install opencv-python face_recognition dlib numpy pyttsx3 PyQt6
```

> **Note:** `dlib` may require Visual Studio Build Tools to compile successfully.

### 1. Register Your Face

Run the registration script to capture your facial encodings:

```bash
python register1.py
```

- Press **S** to capture a sample (at least 3 recommended).
- Ensure only your face is visible in the frame.
- Press **Q** to complete registration.

### 2. Launch the Lock Screen

Run the main application to lock your device:

```bash
python iron_hud.py
```

The application will go fullscreen and secure your system. It will automatically unlock once it recognizes your face.

*(Emergency Override: Press `Alt+T` to force-terminate the application if necessary.)*

---

## 🏗️ Project Architecture

- **Main UI Process (`iron_hud.py`)**: Handles the PyQt6 fullscreen window, camera feed rendering, and application state.
- **Background Recognition Thread (`face_recognition_thread.py`)**: A `QThread` dedicated to running heavy face detection and comparison algorithms off the main UI thread, ensuring a smooth visual experience.
- **Keyboard Hook**: A low-level Windows hook (`WH_KEYBOARD_LL`) that effectively sandboxes the user by blocking system-level navigation keys.

---

## 📄 License

This project is open-source and intended for educational purposes and portfolio demonstration.
