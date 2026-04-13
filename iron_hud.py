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
    QImage, QPixmap, QPainterPath,
    QRadialGradient, QLinearGradient
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
VK_Q = 0x51

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
_terminate_requested = False

def _keyboard_hook_proc(nCode, wParam, lParam):
    global _hook_active, _terminate_requested
    if not _hook_active:
        return user32.CallNextHookEx(_hook_handle, nCode, wParam, lParam)

    if nCode >= 0:
        kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
        vk = kb.vkCode
        alt_down = kb.flags & LLKHF_ALTDOWN

        # ✅ Ctrl+Shift+Q = Termination key (bypasses all blocks)
        ctrl_down = user32.GetAsyncKeyState(0x11) & 0x8000
        shift_down = user32.GetAsyncKeyState(0x10) & 0x8000
        if vk == VK_Q and ctrl_down and shift_down:
            _terminate_requested = True
            return user32.CallNextHookEx(_hook_handle, nCode, wParam, lParam)

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

        # ── Core state ─────────────────────────────────────────────
        self.tick = 0
        self.grid_offset = 0
        self.current_frame = None
        self.state = "SCANNING"
        self.voice_ready = True

        # ── Camera ─────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ── Face data ──────────────────────────────────────────────
        with open(FACE_DATA_PATH, "rb") as f:
            self.known_encodings = pickle.load(f)

        self.face_box = None
        self.access_granted = False

        self.recognition_thread = FaceRecognitionThread(self.known_encodings)
        self.recognition_thread.result.connect(self.update_recognition)
        self.recognition_thread.start()

        # ═══════════════════════════════════════════════════════════
        # ══  PROCEDURAL VISUAL SYSTEMS  ═══════════════════════════
        # ═══════════════════════════════════════════════════════════

        # ── Procedural nebula clouds (slowly drifting, breathing) ──
        self.nebula_clouds = []
        for _ in range(9):
            self.nebula_clouds.append({
                "x": random.uniform(-200, 2120),
                "y": random.uniform(-200, 1280),
                "vx": random.uniform(-0.06, 0.06),
                "vy": random.uniform(-0.04, 0.04),
                "radius": random.randint(280, 580),
                "color": random.choice([
                    QColor(15, 35, 110, 22),
                    QColor(50, 15, 90, 18),
                    QColor(10, 55, 95, 20),
                    QColor(35, 8, 70, 15),
                    QColor(12, 45, 75, 18),
                    QColor(60, 25, 80, 16),
                ]),
                "phase": random.uniform(0, math.pi * 2),
                "breathe_speed": random.uniform(0.003, 0.008),
            })

        # ── Parallax starfield (3 depth layers) ───────────────────
        self.star_layers = []
        layer_cfgs = [
            # (count, min_spd, max_spd, max_size, min_alpha, max_alpha)
            (200, 0.01, 0.05, 1, 40, 120),   # distant
            (110, 0.03, 0.10, 2, 80, 180),   # mid-range
            (45,  0.06, 0.18, 3, 140, 255),  # close
        ]
        for count, min_s, max_s, max_sz, min_a, max_a in layer_cfgs:
            layer = []
            for _ in range(count):
                drift = random.uniform(0, math.pi * 2)
                spd = random.uniform(min_s, max_s)
                layer.append({
                    "x": random.uniform(0, 1920),
                    "y": random.uniform(0, 1080),
                    "vx": math.cos(drift) * spd,
                    "vy": math.sin(drift) * spd,
                    "size": random.randint(1, max_sz),
                    "alpha": random.randint(min_a, max_a),
                    "tw_phase": random.uniform(0, math.pi * 2),
                    "tw_speed": random.uniform(0.012, 0.06),
                })
            self.star_layers.append(layer)

        # ── Zero-gravity floating particles ───────────────────────
        self.particles = []
        for _ in range(60):
            a = random.uniform(0, math.pi * 2)
            s = random.uniform(0.08, 0.4)
            self.particles.append({
                "x": random.uniform(0, 1920),
                "y": random.uniform(0, 1080),
                "vx": math.cos(a) * s,
                "vy": math.sin(a) * s,
                "size": random.uniform(1.5, 4.0),
                "alpha": random.randint(20, 80),
                "color": random.choice([
                    QColor(0, 200, 255),
                    QColor(80, 140, 255),
                    QColor(180, 200, 255),
                    QColor(140, 90, 255),
                ]),
                "trail": [],
                "trail_max": random.randint(12, 28),
            })

        # ── Advanced orbital structures (3D-projected, Keplerian) ─
        self.orbitals = []
        orbital_defs = [
            # (semi_major, semi_minor, tilt°, speed, rotation_offset°, segments, nodes)
            (300, 300,   0,    0.22,   0,  10, 4),
            (340, 185,  30,   -0.16,  42,   8, 3),
            (255, 155, -22,    0.30,  75,  14, 3),
            (385, 245,  18,   -0.11,  18,   6, 4),
            (430, 310, -38,    0.07, 105,   5, 2),
            (195, 115,  42,   -0.38,  58,  18, 2),
        ]
        for sm, sn, tilt, spd, rot, n_seg, n_nod in orbital_defs:
            ecc = max(0.0, 1.0 - sn / sm) if sm > 0 else 0.0
            nodes = []
            for i in range(n_nod):
                nodes.append({
                    "phase": (i / n_nod) * 360 + random.uniform(-12, 12),
                    "size": random.uniform(3, 6),
                    "pulse_phase": random.uniform(0, math.pi * 2),
                    "pulse_speed": random.uniform(0.04, 0.09),
                })
            segments = []
            seg_span = 360 / n_seg
            for i in range(n_seg):
                if random.random() > 0.22:
                    segments.append({
                        "start": i * seg_span,
                        "span": seg_span * random.uniform(0.45, 0.82),
                        "alpha": random.randint(45, 150),
                    })
            self.orbitals.append({
                "semi_major": sm, "semi_minor": sn,
                "tilt": tilt, "rotation": rot, "speed": spd,
                "eccentricity": ecc,
                "segments": segments, "nodes": nodes,
                "breathe_phase": random.uniform(0, math.pi * 2),
                "breathe_speed": random.uniform(0.008, 0.022),
            })

        # ── Scan pulse rings ──────────────────────────────────────
        self.scan_pulses = []
        self.scan_pulse_timer = 0

        # ── Window / lock-screen setup ────────────────────────────
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

        # ✅ Poll for Ctrl+Shift+Q termination request
        self.kill_timer = QTimer(self)
        self.kill_timer.timeout.connect(self._check_terminate)
        self.kill_timer.start(200)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

    # ── Focus enforcement ─────────────────────────────────────────
    def _force_focus(self):
        """Re-activate and raise window if it loses focus."""
        if self.state != "ACCESS GRANTED":
            self.raise_()
            self.activateWindow()
            self.showFullScreen()

    # ── Text-to-speech ────────────────────────────────────────────
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

    # ══════════════════════════════════════════════════════════════
    # ══  PER-FRAME PHYSICS UPDATE  ═══════════════════════════════
    # ══════════════════════════════════════════════════════════════

    def update_frame(self):
        self.tick += 1

        self.grid_offset += 4
        if self.grid_offset > 40:
            self.grid_offset = 0

        # ── Camera ──
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.recognition_thread.update_frame(frame)

        w = self.width()
        h = self.height()
        cx = w * 0.5
        cy = h * 0.5

        # ── Nebula drift ──
        for cloud in self.nebula_clouds:
            cloud["x"] += cloud["vx"]
            cloud["y"] += cloud["vy"]
            cloud["phase"] += cloud["breathe_speed"]
            r = cloud["radius"]
            if cloud["x"] < -r:   cloud["x"] = w + r
            if cloud["x"] > w+r:  cloud["x"] = -r
            if cloud["y"] < -r:   cloud["y"] = h + r
            if cloud["y"] > h+r:  cloud["y"] = -r

        # ── Starfield drift ──
        for layer in self.star_layers:
            for star in layer:
                star["x"] += star["vx"]
                star["y"] += star["vy"]
                star["tw_phase"] += star["tw_speed"]
                if star["x"] < -5:    star["x"] = w + 5
                if star["x"] > w+5:   star["x"] = -5
                if star["y"] < -5:    star["y"] = h + 5
                if star["y"] > h+5:   star["y"] = -5

        # ── Particle zero-gravity physics ──
        for p in self.particles:
            dx = cx - p["x"]
            dy = cy - p["y"]
            dist = math.sqrt(dx * dx + dy * dy) + 1.0
            grav = 0.004 / (1.0 + dist * 0.002)
            p["vx"] += grav * dx / dist + random.uniform(-0.008, 0.008)
            p["vy"] += grav * dy / dist + random.uniform(-0.008, 0.008)
            p["vx"] *= 0.9985
            p["vy"] *= 0.9985
            spd = math.sqrt(p["vx"] ** 2 + p["vy"] ** 2)
            if spd > 1.5:
                p["vx"] *= 1.5 / spd
                p["vy"] *= 1.5 / spd
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["trail"].append((p["x"], p["y"]))
            if len(p["trail"]) > p["trail_max"]:
                p["trail"].pop(0)
            margin = 60
            if p["x"] < margin:      p["vx"] += 0.05
            elif p["x"] > w-margin:  p["vx"] -= 0.05
            if p["y"] < margin:      p["vy"] += 0.05
            elif p["y"] > h-margin:  p["vy"] -= 0.05

        # ── Orbital Keplerian updates ──
        for orb in self.orbitals:
            orb["rotation"] += orb["speed"]
            orb["breathe_phase"] += orb["breathe_speed"]
            ecc = orb["eccentricity"]
            for node in orb["nodes"]:
                speed_mod = (1.0 + ecc * math.cos(math.radians(node["phase"]))) ** 1.3
                node["phase"] += orb["speed"] * 2.5 * speed_mod
                node["phase"] %= 360
                node["pulse_phase"] += node["pulse_speed"]

        # ── Scan pulse spawning ──
        self.scan_pulse_timer += 1
        if self.scan_pulse_timer >= 100:
            self.scan_pulse_timer = 0
            self.scan_pulses.append({"radius": 40.0, "alpha": 180.0, "speed": 2.5})
        for pulse in self.scan_pulses:
            pulse["radius"] += pulse["speed"]
            pulse["alpha"] -= 1.5
            pulse["speed"] *= 0.995
        self.scan_pulses = [p for p in self.scan_pulses if p["alpha"] > 0]

        self.update()

    # ══════════════════════════════════════════════════════════════
    # ══  RECOGNITION CALLBACK  ═══════════════════════════════════
    # ══════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════
    # ══  COLOR SCHEME  ═══════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════

    def get_ring_color(self):

        if self.state == "ACCESS GRANTED":
            return QColor(0, 255, 120)

        if self.state == "UNKNOWN FACE":
            return QColor(255, 80, 80)

        return QColor(0, 255, 255)

    # ══════════════════════════════════════════════════════════════
    # ══  ORBITAL GEOMETRY  ═══════════════════════════════════════
    # ══════════════════════════════════════════════════════════════

    def _orbital_point(self, semi_major, semi_minor,
                       angle_deg, tilt_deg, rotation_deg, cx, cy):
        """Compute 2D screen position of a point on a 3D-projected
        elliptical orbit via tilt (X-axis rotation) then 2D rotation."""
        a = math.radians(angle_deg)
        t = math.radians(tilt_deg)
        r = math.radians(rotation_deg)
        px = semi_major * math.cos(a)
        py = semi_minor * math.sin(a)
        py_t = py * math.cos(t)          # 3D tilt projection
        rx = px * math.cos(r) - py_t * math.sin(r)
        ry = px * math.sin(r) + py_t * math.cos(r)
        return cx + rx, cy + ry

    # ══════════════════════════════════════════════════════════════
    # ══  FACE DETECTION OVERLAYS  ════════════════════════════════
    # ══════════════════════════════════════════════════════════════

    def draw_face_brackets(self, painter, x1, y1, x2, y2, color):

        size = 40

        pen = QPen(color, 4)
        painter.setPen(pen)

        painter.drawLine(x1, y1, x1+size, y1)
        painter.drawLine(x1, y1, x1, y1+size)

        painter.drawLine(x2, y1, x2-size, y1)
        painter.drawLine(x2, y1, x2, y1+size)

        painter.drawLine(x1, y2, x1+size, y2)
        painter.drawLine(x1, y2, x1, y2-size)

        painter.drawLine(x2, y2, x2-size, y2)
        painter.drawLine(x2, y2, x2, y2-size)

    def draw_face_grid(self, painter, x1, y1, x2, y2, color=None):

        grid_color = QColor(color) if color else QColor(0, 255, 255, 120)
        grid_color.setAlpha(120)
        painter.setPen(QPen(grid_color, 1))

        step = 20

        for x in range(x1, x2, step):
            painter.drawLine(x, y1, x, y2)

        for y in range(y1 + self.grid_offset, y2, step):
            painter.drawLine(x1, y, x2, y)

    # ══════════════════════════════════════════════════════════════
    # ══  MAIN RENDER  ════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════

    def paintEvent(self, event):

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w // 2
        cy = h // 2
        lens_radius = 260
        ring_color = self.get_ring_color()

        # ─── 1. DEEP-SPACE GRADIENT BACKGROUND ───────────────────
        bg = QLinearGradient(0, 0, w, h)
        bg.setColorAt(0.0,  QColor(2, 3, 15))
        bg.setColorAt(0.3,  QColor(5, 8, 28))
        bg.setColorAt(0.55, QColor(8, 4, 22))
        bg.setColorAt(0.8,  QColor(4, 6, 20))
        bg.setColorAt(1.0,  QColor(3, 5, 18))
        painter.fillRect(self.rect(), bg)

        # ─── 2. PROCEDURAL NEBULA ─────────────────────────────────
        painter.setPen(Qt.PenStyle.NoPen)
        for cloud in self.nebula_clouds:
            breathe = 1.0 + 0.08 * math.sin(cloud["phase"])
            r = int(cloud["radius"] * breathe)
            grad = QRadialGradient(cloud["x"], cloud["y"], r)
            grad.setColorAt(0, cloud["color"])
            grad.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setBrush(grad)
            painter.drawEllipse(
                int(cloud["x"] - r), int(cloud["y"] - r),
                r * 2, r * 2
            )

        # ─── 3. PARALLAX STARFIELD ────────────────────────────────
        for layer in self.star_layers:
            for star in layer:
                alpha = star["alpha"] * (0.4 + 0.6 * math.sin(star["tw_phase"]))
                alpha = max(0, min(255, int(alpha)))
                c = QColor(200, 215, 255, alpha)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(c)
                sz = star["size"]
                painter.drawEllipse(
                    int(star["x"]), int(star["y"]), sz, sz
                )
                # Bloom on large bright stars
                if sz >= 2 and alpha > 160:
                    glow = QColor(180, 200, 255, int(alpha * 0.18))
                    painter.setBrush(glow)
                    painter.drawEllipse(
                        int(star["x"]) - sz, int(star["y"]) - sz,
                        sz * 3, sz * 3
                    )

        # ─── 4. FLOATING PARTICLES WITH TRAILS ───────────────────
        for p in self.particles:
            base_c = p["color"]
            trail = p["trail"]
            t_len = len(trail)

            # Trail as fading connected lines
            if t_len > 1:
                for i in range(t_len - 1):
                    progress = (i + 1) / t_len
                    tc = QColor(base_c)
                    tc.setAlpha(int(p["alpha"] * progress * 0.4))
                    pw = max(0.5, p["size"] * progress * 0.5)
                    painter.setPen(QPen(tc, pw))
                    x1t, y1t = trail[i]
                    x2t, y2t = trail[i + 1]
                    painter.drawLine(
                        int(x1t), int(y1t), int(x2t), int(y2t)
                    )

            # Particle body
            pc = QColor(base_c)
            pc.setAlpha(p["alpha"])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(pc)
            sz = int(p["size"])
            painter.drawEllipse(int(p["x"]), int(p["y"]), sz, sz)

            # Glow halo
            gc = QColor(base_c)
            gc.setAlpha(int(p["alpha"] * 0.25))
            painter.setBrush(gc)
            painter.drawEllipse(
                int(p["x"]) - sz, int(p["y"]) - sz,
                sz * 3, sz * 3
            )

        # ─── 5. ORBITAL STRUCTURES ───────────────────────────────
        all_node_pos = []

        for orb in self.orbitals:
            sm = orb["semi_major"]
            sn = orb["semi_minor"]
            tilt = orb["tilt"]
            rot = orb["rotation"]
            breathe = 0.7 + 0.3 * math.sin(orb["breathe_phase"])

            # Segmented arcs
            for seg in orb["segments"]:
                seg_alpha = int(seg["alpha"] * breathe)
                seg_alpha = max(0, min(255, seg_alpha))
                sc = QColor(ring_color)
                sc.setAlpha(seg_alpha)
                painter.setPen(QPen(sc, 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)

                start = seg["start"]
                span = seg["span"]
                steps = max(int(span / 2.5), 8)
                prev = None
                for i in range(steps + 1):
                    angle = start + span * (i / steps)
                    px, py = self._orbital_point(
                        sm, sn, angle, tilt, rot, cx, cy
                    )
                    if prev:
                        painter.drawLine(
                            int(prev[0]), int(prev[1]),
                            int(px), int(py)
                        )
                    prev = (px, py)

            # Keplerian nodes
            for node in orb["nodes"]:
                nx, ny = self._orbital_point(
                    sm, sn, node["phase"], tilt, rot, cx, cy
                )
                all_node_pos.append((nx, ny))

                pulse = 0.55 + 0.45 * math.sin(node["pulse_phase"])
                n_sz = node["size"] * pulse

                # Core dot
                nc = QColor(ring_color)
                nc.setAlpha(int(220 * pulse))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(nc)
                painter.drawEllipse(
                    int(nx - n_sz), int(ny - n_sz),
                    int(n_sz * 2), int(n_sz * 2)
                )
                # Glow
                gc = QColor(ring_color)
                gc.setAlpha(int(55 * pulse))
                painter.setBrush(gc)
                g = n_sz * 3
                painter.drawEllipse(
                    int(nx - g), int(ny - g),
                    int(g * 2), int(g * 2)
                )

        # Energy web between nearby orbital nodes
        for i in range(len(all_node_pos)):
            for j in range(i + 1, len(all_node_pos)):
                ax, ay = all_node_pos[i]
                bx, by = all_node_pos[j]
                d = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
                if d < 140:
                    alpha = int(50 * (1 - d / 140))
                    lc = QColor(ring_color)
                    lc.setAlpha(alpha)
                    painter.setPen(QPen(lc, 0.5))
                    painter.drawLine(int(ax), int(ay), int(bx), int(by))

        # ─── 6. SCAN PULSE RINGS ─────────────────────────────────
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for pulse in self.scan_pulses:
            r = int(pulse["radius"])
            a = max(0, min(255, int(pulse["alpha"])))
            pc = QColor(ring_color)
            pc.setAlpha(a)
            painter.setPen(QPen(pc, 1))
            painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # ─── 7. CAMERA CIRCULAR VIEWPORT ─────────────────────────
        if self.current_frame is not None:

            frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            fh, fw, fc = frame.shape

            qimg = QImage(
                frame.data, fw, fh, fc * fw,
                QImage.Format.Format_RGB888
            )

            painter.save()

            mask = QPainterPath()
            mask.addEllipse(
                cx - lens_radius, cy - lens_radius,
                lens_radius * 2, lens_radius * 2
            )
            painter.setClipPath(mask)

            pix = QPixmap.fromImage(qimg).scaled(
                lens_radius * 2,
                lens_radius * 2,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )

            painter.drawPixmap(cx - lens_radius, cy - lens_radius, pix)
            painter.restore()

        # Viewport border ring
        vb = QColor(ring_color)
        vb.setAlpha(120)
        painter.setPen(QPen(vb, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(
            cx - lens_radius, cy - lens_radius,
            lens_radius * 2, lens_radius * 2
        )

        # Subtle crosshair overlay
        ch = QColor(ring_color)
        ch.setAlpha(35)
        painter.setPen(QPen(ch, 1))
        painter.drawLine(cx - lens_radius, cy, cx + lens_radius, cy)
        painter.drawLine(cx, cy - lens_radius, cx, cy + lens_radius)
        ch.setAlpha(90)
        painter.setPen(QPen(ch, 1))
        painter.drawLine(cx - 20, cy, cx + 20, cy)
        painter.drawLine(cx, cy - 20, cx, cy + 20)

        # ─── 8. FACE DETECTION OVERLAY ───────────────────────────
        if self.face_box is not None and self.current_frame is not None:

            top, right, bottom, left = self.face_box

            frame_h, frame_w = self.current_frame.shape[:2]

            lens_size = lens_radius * 2

            scale_x = lens_size / frame_w
            scale_y = lens_size / frame_h

            fx1 = int(cx - lens_radius + left * scale_x)
            fy1 = int(cy - lens_radius + top * scale_y)
            fx2 = int(cx - lens_radius + right * scale_x)
            fy2 = int(cy - lens_radius + bottom * scale_y)

            # Use ring color for consistent HUD look:
            # cyan=scanning, green=access granted, red=unknown
            color = self.get_ring_color()

            self.draw_face_grid(painter, fx1, fy1, fx2, fy2, color)
            self.draw_face_brackets(painter, fx1, fy1, fx2, fy2, color)

        # ─── 9. HUD TEXT OVERLAYS ─────────────────────────────────
        painter.setPen(QColor(0, 255, 255))
        painter.setFont(QFont("Consolas", 18))
        painter.drawText(50, h - 70, f"STATUS : {self.state}")

        # Decorative system readouts (left)
        painter.setPen(QColor(0, 255, 255, 75))
        painter.setFont(QFont("Consolas", 8))
        left_text = [
            "SYS.CORE :: ONLINE",
            "FIELD INTEGRITY : 99.2%",
            "ORBITAL SYNC : NOMINAL",
            f"FRAME : {int(self.tick)}",
        ]
        for i, txt in enumerate(left_text):
            painter.drawText(20, 22 + i * 15, txt)

        # Decorative system readouts (right)
        right_text = [
            "BIOMETRIC.SCAN : ACTIVE",
            "NEURAL.NET : ENGAGED",
            "THREAT LEVEL : NONE",
        ]
        for i, txt in enumerate(right_text):
            painter.drawText(w - 250, 22 + i * 15, txt)

    # ══════════════════════════════════════════════════════════════
    # ══  INPUT HANDLING & LIFECYCLE  ═════════════════════════════
    # ══════════════════════════════════════════════════════════════

    # 🔒 Block keyboard (except Ctrl+Shift+Q termination)
    def keyPressEvent(self, event):
        if (event.modifiers() == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
                and event.key() == Qt.Key.Key_Q):
            self._force_terminate()
            return
        # Block everything else
        pass

    def _check_terminate(self):
        """Poll the global termination flag set by Ctrl+Shift+Q."""
        global _terminate_requested
        if _terminate_requested:
            _terminate_requested = False
            self._force_terminate()

    def _force_terminate(self):
        """Force-close the app, bypassing the access-granted check."""
        self.state = "TERMINATED"  # Allow closeEvent to proceed
        self.close()

    # 🔒 Prevent closing unless unlocked or force-terminated
    def closeEvent(self, event):

        if self.state not in ("ACCESS GRANTED", "TERMINATED"):
            event.ignore()
            return

        # 🔓 Unhook keyboard and stop all timers
        uninstall_keyboard_hook()
        self.focus_timer.stop()
        self.kill_timer.stop()

        if self.cap:
            self.cap.release()

        self.recognition_thread.running = False
        self.recognition_thread.quit()
        self.recognition_thread.wait()

        event.accept()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = IronHUD()

    sys.exit(app.exec())