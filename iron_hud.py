import os
import sys
import cv2
import math
import random
import pickle
import threading
import ctypes
import ctypes.wintypes
import pyttsx3

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QFont,
    QImage, QPixmap, QPainterPath, QRadialGradient
)

from face_recognition_thread import FaceRecognitionThread

# Resolve path relative to exe directory (for PyInstaller builds)
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_DATA_PATH = os.path.join(BASE_DIR, "face_data.pkl")

# ── Windows Low-Level Keyboard Hook ──────────────────────────────────
# Blocks: Alt+F4, Alt+Tab, Win key (L/R), Ctrl+Esc
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
VK_TAB = 0x09
VK_ESCAPE = 0x1B
VK_F4 = 0x73
VK_LWIN = 0x5B
VK_RWIN = 0x5C

LLKHF_ALTDOWN = 0x20

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

HOOKPROC = ctypes.CFUNCTYPE(
    ctypes.c_long,
    ctypes.c_int,
    ctypes.wintypes.WPARAM,
    ctypes.wintypes.LPARAM
)

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", ctypes.wintypes.DWORD),
        ("scanCode", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

_hook_handle = None
_hook_active = True

def _keyboard_hook_proc(nCode, wParam, lParam):
    global _hook_active
    if not _hook_active:
        return user32.CallNextHookEx(_hook_handle, nCode, wParam, lParam)

    if nCode >= 0:
        kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
        vk = kb.vkCode
        alt_down = kb.flags & LLKHF_ALTDOWN

        # Block Win keys
        if vk in (VK_LWIN, VK_RWIN):
            return 1

        # Block Alt+Tab
        if vk == VK_TAB and alt_down:
            return 1

        # Block Alt+F4
        if vk == VK_F4 and alt_down:
            return 1

        # Block Ctrl+Esc (Start menu)
        if vk == VK_ESCAPE and (user32.GetAsyncKeyState(0x11) & 0x8000):
            return 1

    return user32.CallNextHookEx(_hook_handle, nCode, wParam, lParam)

# Must keep a reference so it doesn't get garbage collected
_callback = HOOKPROC(_keyboard_hook_proc)

def install_keyboard_hook():
    global _hook_handle
    _hook_handle = user32.SetWindowsHookExW(
        WH_KEYBOARD_LL, _callback,
        kernel32.GetModuleHandleW(None), 0
    )

def uninstall_keyboard_hook():
    global _hook_handle, _hook_active
    _hook_active = False
    if _hook_handle:
        user32.UnhookWindowsHookEx(_hook_handle)
        _hook_handle = None
# ─────────────────────────────────────────────────────────────────────


class IronHUD(QMainWindow):

    def __init__(self):
        super().__init__()

        self.outer_angle = 0
        self.inner_angle = 0
        self.sweep_angle = 0
        self.node_phase = 0

        self.grid_offset = 0

        self.current_frame = None
        self.state = "SCANNING"

        self.voice_ready = True

        # 🔥 Faster camera start
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        with open(FACE_DATA_PATH, "rb") as f:
            self.known_encodings = pickle.load(f)

        self.face_box = None
        self.access_granted = False

        self.recognition_thread = FaceRecognitionThread(self.known_encodings)
        self.recognition_thread.result.connect(self.update_recognition)
        self.recognition_thread.start()

        # stars
        self.stars = []
        for i in range(200):
            self.stars.append({
                "x": random.randint(0, 1920),
                "y": random.randint(0, 1080),
                "speed": random.uniform(0.05, 0.3)
            })

        # nebula
        self.nebula = []
        for i in range(6):
            self.nebula.append({
                "x": random.randint(0, 1920),
                "y": random.randint(0, 1080),
                "radius": random.randint(200, 400),
                "color": random.choice([
                    QColor(80,120,255,40),
                    QColor(120,60,255,40),
                    QColor(40,180,255,30)
                ])
            })

        # 🔒 Lock screen behavior
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )

        self.setCursor(Qt.CursorShape.BlankCursor)

        self.showFullScreen()

        # 🔒 Install OS-level keyboard hook
        install_keyboard_hook()

        # 🔒 Re-grab focus if window loses it
        self.focus_timer = QTimer(self)
        self.focus_timer.timeout.connect(self._force_focus)
        self.focus_timer.start(500)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

    def _force_focus(self):
        """Re-activate and raise window if it loses focus."""
        if self.state != "ACCESS GRANTED":
            self.raise_()
            self.activateWindow()
            self.showFullScreen()

    def speak(self, text, on_done=None):

        if not self.voice_ready:
            return

        self.voice_ready = False

        def run():
            try:
                engine = pyttsx3.init('sapi5')
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[0].id)
                engine.setProperty('rate', 180)

                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"TTS error: {e}")
            finally:
                self.voice_ready = True
                if on_done:
                    # Schedule the callback on the main thread via QTimer
                    QTimer.singleShot(0, on_done)

        threading.Thread(target=run, daemon=False).start()

    def update_frame(self):

        self.outer_angle += 0.8
        self.inner_angle -= 1.2
        self.sweep_angle += 3
        self.node_phase += 0.08

        self.grid_offset += 4
        if self.grid_offset > 40:
            self.grid_offset = 0

        if self.sweep_angle >= 360:
            self.sweep_angle = 0

        ret, frame = self.cap.read()

        if ret:
            self.current_frame = frame
            self.recognition_thread.update_frame(frame)

        for star in self.stars:
            star["y"] += star["speed"]
            if star["y"] > self.height():
                star["y"] = 0
                star["x"] = random.randint(0, self.width())

        self.update()

    def update_recognition(self, face_box, access):

        self.face_box = face_box
        self.access_granted = access

        if access:
            if self.state != "ACCESS GRANTED":
                self.state = "ACCESS GRANTED"
                # 🔓 Speak first, then close after TTS finishes
                self.speak("Identity verified. Welcome back sir.", on_done=self.close)

        elif face_box is not None:
            self.state = "UNKNOWN FACE"

        else:
            self.state = "SCANNING"

    def get_ring_color(self):

        if self.state == "ACCESS GRANTED":
            return QColor(0,255,120)

        if self.state == "UNKNOWN FACE":
            return QColor(255,80,80)

        return QColor(0,255,255)

    def draw_arc(self, painter, radius, start, span, color):

        pen = QPen(color)
        pen.setWidth(3)
        painter.setPen(pen)

        painter.drawArc(
            -radius, -radius,
            radius*2, radius*2,
            int(start*16),
            int(span*16)
        )

    def draw_face_brackets(self, painter, x1, y1, x2, y2, color):

        size = 40

        pen = QPen(color,4)
        painter.setPen(pen)

        painter.drawLine(x1,y1,x1+size,y1)
        painter.drawLine(x1,y1,x1,y1+size)

        painter.drawLine(x2,y1,x2-size,y1)
        painter.drawLine(x2,y1,x2,y1+size)

        painter.drawLine(x1,y2,x1+size,y2)
        painter.drawLine(x1,y2,x1,y2-size)

        painter.drawLine(x2,y2,x2-size,y2)
        painter.drawLine(x2,y2,x2,y2-size)

    def draw_face_grid(self, painter, x1, y1, x2, y2, color=None):

        grid_color = QColor(color) if color else QColor(0,255,255,120)
        grid_color.setAlpha(120)
        painter.setPen(QPen(grid_color, 1))

        step = 20

        for x in range(x1, x2, step):
            painter.drawLine(x, y1, x, y2)

        for y in range(y1 + self.grid_offset, y2, step):
            painter.drawLine(x1, y, x2, y)

    def paintEvent(self, event):

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(5,10,31))

        for cloud in self.nebula:

            gradient = QRadialGradient(
                cloud["x"], cloud["y"], cloud["radius"]
            )

            gradient.setColorAt(0, cloud["color"])
            gradient.setColorAt(1, QColor(0,0,0,0))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)

            painter.drawEllipse(
                cloud["x"]-cloud["radius"],
                cloud["y"]-cloud["radius"],
                cloud["radius"]*2,
                cloud["radius"]*2
            )

        for star in self.stars:
            painter.setPen(QPen(QColor(200,220,255,200)))
            painter.drawPoint(int(star["x"]), int(star["y"]))

        cx = self.width()//2
        cy = self.height()//2
        lens_radius = 260

        ring_color = self.get_ring_color()

        if self.current_frame is not None:

            frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            h,w,ch = frame.shape

            qimg = QImage(frame.data,w,h,ch*w,QImage.Format.Format_RGB888)

            painter.save()

            mask = QPainterPath()
            mask.addEllipse(cx-lens_radius, cy-lens_radius, lens_radius*2, lens_radius*2)
            painter.setClipPath(mask)

            pix = QPixmap.fromImage(qimg).scaled(
                lens_radius*2,
                lens_radius*2,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )

            painter.drawPixmap(cx-lens_radius, cy-lens_radius, pix)
            painter.restore()

        painter.save()
        painter.translate(cx,cy)
        painter.rotate(self.outer_angle)

        for i in range(8):
            self.draw_arc(painter,330,i*(360/8),20,ring_color)

        painter.restore()

        painter.save()
        painter.translate(cx,cy)
        painter.rotate(self.inner_angle)

        for i in range(6):
            self.draw_arc(painter,240,i*(360/6),25,ring_color)

        painter.restore()

        painter.save()
        painter.translate(cx,cy)
        painter.rotate(self.sweep_angle)

        pen = QPen(ring_color,3)
        painter.setPen(pen)
        painter.drawLine(0,0,0,-lens_radius)

        painter.restore()

        orbit = 200
        r = 12 + math.sin(self.node_phase)*3

        for i in range(3):

            angle = math.radians(self.outer_angle + i*120)

            x = cx + orbit*math.cos(angle)
            y = cy + orbit*math.sin(angle)

            painter.setPen(QPen(ring_color,2))
            painter.drawEllipse(int(x-r), int(y-r), int(r*2), int(r*2))

        if self.face_box is not None and self.current_frame is not None:

            top,right,bottom,left = self.face_box

            frame_h, frame_w = self.current_frame.shape[:2]

            lens_size = lens_radius*2

            scale_x = lens_size / frame_w
            scale_y = lens_size / frame_h

            x1 = int(cx - lens_radius + left * scale_x)
            y1 = int(cy - lens_radius + top * scale_y)
            x2 = int(cx - lens_radius + right * scale_x)
            y2 = int(cy - lens_radius + bottom * scale_y)

            # Use ring color for consistent HUD look:
            # cyan=scanning, green=access granted, red=unknown
            color = self.get_ring_color()

            self.draw_face_grid(painter, x1, y1, x2, y2, color)
            self.draw_face_brackets(painter, x1, y1, x2, y2, color)

        painter.setPen(QColor(0,255,255))
        painter.setFont(QFont("Consolas",18))
        painter.drawText(50,self.height()-70,f"STATUS : {self.state}")

    # 🔒 Block keyboard
    def keyPressEvent(self, event):
        pass

    # 🔒 Prevent closing unless unlocked
    def closeEvent(self,event):

        if self.state != "ACCESS GRANTED":
            event.ignore()
            return

        # 🔓 Unhook keyboard and stop focus grabbing
        uninstall_keyboard_hook()
        self.focus_timer.stop()

        if self.cap:
            self.cap.release()

        self.recognition_thread.running=False
        self.recognition_thread.quit()
        self.recognition_thread.wait()

        event.accept()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = IronHUD()

    sys.exit(app.exec())