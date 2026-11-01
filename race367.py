# FaceRacer_Combined.py
# Complete Version: Face Detection + Car Selection + Multi-Track Leaderboard
# + Dual IMU/Compass Controls: 
#     - PKS Compass (Steering Wheel via Yaw Gyro Z + Accel Roll)
#     - Non-PKS Compass/MPU6050 (Throttle Pedal via Z-Axis Gravity + Road Friction Deceleration)
# + UI Serial Log Console & Terminal Print Debugging
# + Line Buffer Stream Stitching for Fragmented Serial Data

import pygame
import pygame.midi
import random
import sys
import math
import array
import json
import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import threading
import re
import time
from collections import deque

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("⚠️  pyserial not installed. Serial features disabled. Install with: pip install pyserial")

import cv2
import numpy as np

pygame.init()
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()

# ============================================================
# SERIAL CONFIGURATION & CONTROL STATE
# ============================================================
SERIAL_PORT   = "COM24"     # Set to your active serial port
BAUD          = 115200

ser_conn = None

serial_steer           = 0.0     # -1.0 (Full Left) .. +1.0 (Full Right)
serial_throttle        = 0.0     #  0.0 (Idle) .. 1.0 (Full Throttle)
serial_steer_active    = False
serial_throttle_active = False

# Calibration Settings for Controls
OFFSET_THR_Z       = 9.81    # Standard resting gravity on Z-axis (m/s^2)
FLIPPED_THROTTLE_Z = -1.0    # Invert sign if stepping down decreases Z value
THR_SENSITIVITY    = 4.0     # Gravity deflection range for 100% throttle
STEER_SENSITIVITY  = 40.0    # Degrees / Gyro accumulation sensitivity scale

steer_yaw = 0.0
last_sensor_time = time.time()

# UI Serial Console Buffer (keeps latest 5 logs)
serial_log_buffer = deque(maxlen=5)

def add_serial_log(text):
    """Pushes a timestamped string to the UI console log buffer."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    serial_log_buffer.append(f"[{timestamp}] {text}")

def _open_serial(port):
    if not port:
        return None
    try:
        return serial.Serial(port, BAUD, timeout=0.05)
    except Exception:
        return None

def _auto_serial_port():
    if not HAS_SERIAL:
        return None
    for p in serial.tools.list_ports.comports():
        if any(x in (p.description or "").lower() for x in
               ["arduino", "ch340", "cp210", "usb serial", "peanut", "acm", "usbmodem"]):
            return p.device
    return None

# Auto-calibration baseline for steering
steer_center_angle = None
STEER_RANGE = 35.0  # Degrees of wheel turn for full lock (-1.0 to +1.0)

# ============================================================
# COMPASS & THROTTLE CONFIGURATION
# ============================================================
pks_center_heading     = None   # Auto-calibrates on first compass packet
STEER_COMPASS_RANGE    = 45.0   # Turn angle degrees for full steer lock (-1.0 to +1.0)
INVERT_STEER_DIRECTION = False  # Set to True if turning right steers left

OFFSET_THR_Z           = 9.81   # Earth gravity baseline on Z-axis (m/s^2)
FLIPPED_THROTTLE_Z     = 1.0    # Changed to 1.0: Stepping DOWN accelerates, releasing decelerates
THR_SENSITIVITY        = 4.0    # Deflection amount for full throttle

def parse_sensor_data(line):
    """
    Bulletproof parser for dual IMU streams:
    - Any line containing 'compass' -> Steering Wheel (Extracts heading angle)
    - Non-PKS 'Acceleration' lines -> Throttle Pedal (Extracts Z-gravity shift)
    """
    global serial_steer, serial_throttle, serial_steer_active, serial_throttle_active
    global pks_center_heading

    if not line:
        return

    line_lower = line.lower().strip()

    # ============================================================
    # 1. STEERING: Parse compass line (pks_compass:187, compass:187, etc.)
    # ============================================================
    if "compass" in line_lower:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if nums:
            heading = float(nums[-1])  # Extract heading angle

            # Auto-calibrate baseline center heading on first packet
            if pks_center_heading is None:
                pks_center_heading = heading
                print(f"🎯 [Compass Calibrated] Center angle set to: {pks_center_heading:.1f}°")

            # Calculate deflection angle from center
            delta = heading - pks_center_heading

            # Correct 0° / 360° compass boundary wraparound
            if delta > 180.0:
                delta -= 360.0
            elif delta < -180.0:
                delta += 360.0

            if INVERT_STEER_DIRECTION:
                delta = -delta

            serial_steer = max(-1.0, min(1.0, delta / STEER_COMPASS_RANGE))
            serial_steer_active = True
            return

    # ============================================================
    # 2. THROTTLE: Parse non-PKS Accelerometer lines
    # ============================================================
    if "acceleration" in line_lower and not line_lower.startswith("pks"):
        nums = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
        if len(nums) >= 3:
            az = nums[2]
            
            # Measure Z-axis deflection relative to gravity baseline
            # FLIPPED_THROTTLE_Z = 1.0 ensures stepping down produces positive z_delta
            z_delta = (az - OFFSET_THR_Z) * FLIPPED_THROTTLE_Z
            
            if z_delta > 0.4:
                # Step DOWN -> Accelerate (0.0 to 1.0)
                serial_throttle = max(0.0, min(1.0, z_delta / THR_SENSITIVITY))
            else:
                # Step UP / Released -> 0.0 (Road friction decelerates car)
                serial_throttle = 0.0
                
            serial_throttle_active = True

def serial_reader():
    """Background thread with isolated exception handling to prevent port drops."""
    global ser_conn, serial_steer_active, serial_throttle_active
    logged_connection = False
    line_buffer = ""

    while True:
        try:
            if ser_conn is None or not ser_conn.is_open:
                port_to_try = SERIAL_PORT or _auto_serial_port()
                if port_to_try:
                    ser_conn = _open_serial(port_to_try)
                
                if ser_conn and ser_conn.is_open:
                    ser_conn.reset_input_buffer()
                    line_buffer = ""
                    if not logged_connection:
                        msg = f"Connected on {ser_conn.port}"
                        print(f"✅ [Serial] {msg}")
                        add_serial_log(msg)
                        logged_connection = True
                else:
                    serial_steer_active = False
                    serial_throttle_active = False
                    time.sleep(1.0)
                    continue

            waiting_bytes = ser_conn.in_waiting
            if waiting_bytes > 0:
                raw_bytes = ser_conn.read(waiting_bytes)
                raw_text = raw_bytes.decode("utf-8", errors="ignore")
                line_buffer += raw_text
                
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    clean_line = line.replace("\r", "").strip()
                    
                    if clean_line:
                        print(f"[SERIAL LOG]: {clean_line}")
                        display_line = clean_line if len(clean_line) <= 32 else clean_line[:29] + "..."
                        add_serial_log(display_line)
                        
                        # Protected parse call (prevents bad lines from closing the port)
                        try:
                            parse_sensor_data(clean_line)
                        except Exception as pe:
                            print(f"⚠️ [Parse Error]: {pe}")
            else:
                time.sleep(0.005)

        except Exception as e:
            print(f"⚠️ [Serial Disconnect]: {e}")
            logged_connection = False
            serial_steer_active = False
            serial_throttle_active = False
            line_buffer = ""
            if ser_conn:
                try:
                    ser_conn.close()
                except Exception:
                    pass
                ser_conn = None
            time.sleep(1.0)

if HAS_SERIAL:
    threading.Thread(target=serial_reader, daemon=True).start()

# ============================================================
# MIDI INPUT DISCOVERY
# ============================================================
MIDI_INPUT = None

def get_default_input_device_id():
    default_id = pygame.midi.get_default_input_id()
    if default_id != -1:
        info = pygame.midi.get_device_info(default_id)
        if info and info[2] == 1:
            return default_id
    for i in range(pygame.midi.get_count()):
        info = pygame.midi.get_device_info(i)
        if info and info[2] == 1:
            return i
    return None

try:
    pygame.midi.init()
    midi_id = get_default_input_device_id()
    if midi_id is not None:
        MIDI_INPUT = pygame.midi.Input(midi_id)
        info = pygame.midi.get_device_info(midi_id)
        print(f"✅ MIDI initialized (ID: {midi_id}, Name: {info[1].decode()})")
    else:
        print("⚠️ No MIDI input devices found")
except Exception as e:
    print(f"⚠️ MIDI initialization failed: {e}")
    MIDI_INPUT = None

# ============================================================
# DISPLAY SETUP
# ============================================================
info = pygame.display.Info()
SCREEN_W, SCREEN_H = info.current_w, info.current_h
FULLSCREEN = True
WIDTH, HEIGHT = (SCREEN_W, SCREEN_H) if FULLSCREEN else (1080, 1400)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN if FULLSCREEN else 0)

pygame.display.set_caption("Face Racer - Leaderboard Edition")
clock = pygame.time.Clock()
FPS = 60

SCALE = min(WIDTH / 1000, HEIGHT / 900)
CAR_W_BASE = int(72 * SCALE)
CAR_H_BASE = int(124 * SCALE)

# ============================================================
# COLOR DEFINITIONS
# ============================================================
WHITE = (255, 255, 255)
ROAD = (34, 34, 40)
LANE = (220, 220, 170)
GRASS = (24, 100, 40)
GRASS_DARK = (14, 70, 28)
YELLOW = (255, 215, 40)
RED = (235, 40, 40)
ORANGE = (255, 120, 20)
PURPLE = (160, 50, 200)
CYAN = (30, 190, 220)
GREEN = (40, 220, 100)
GRAY = (70, 70, 80)
GOLD = (255, 215, 0)
CARD_BG = (22, 29, 45)
CARD_HOVER = (34, 45, 66)
CARD_SEL = (28, 54, 80)
STAR_ON = (255, 210, 60)
STAR_OFF = (100, 100, 100)
PED_COLOR = (255, 200, 120)
PED_CLOTHES = (80, 180, 120)
PED_PANTS = (40, 60, 120)
COIN_COLOR = (255, 215, 0)
COIN_EDGE = (200, 150, 0)
TRACK_SEL = (30, 74, 106)
TRACK_NORMAL = (26, 34, 52)
TRACK_HOVER = (36, 48, 70)
BGM_SEL = (92, 52, 126)
UPLOAD_NORMAL = (80, 120, 200)
UPLOAD_HOVER = (110, 150, 230)
NITRO_COLOR = (80, 200, 255)
SHIELD_COLOR = (120, 255, 180)
LASER_COLOR = (220, 120, 255)
WINDOW_COLOR = (180, 220, 255)
SHADOW_COLOR = (0, 0, 0)

# ============================================================
# VEHICLE PRESETS
# ============================================================
CARS = [
    {"name": "Velocity Viper", "desc": "Extremely fast • Low grip • Drifts hard",
     "color": (0, 200, 255), "accent": (255, 255, 255),
     "accel": 1.4, "grip": 0.72, "max_steer": 0.85,
     "size": 1.0, "lean": 1.5, "road_speed": 1.25,
     "stars_speed": 5, "stars_handling": 2},
    {"name": "Titan Tank", "desc": "Heavy • Rock solid grip • Slow",
     "color": (180, 55, 35), "accent": (30, 30, 30),
     "accel": 0.7, "grip": 1.35, "max_steer": 1.2,
     "size": 1.28, "lean": 0.55, "road_speed": 0.8,
     "stars_speed": 2, "stars_handling": 5},
    {"name": "Shadow Stinger", "desc": "Balanced • Great all-rounder",
     "color": (35, 35, 45), "accent": (200, 200, 220),
     "accel": 1.05, "grip": 1.1, "max_steer": 1.1,
     "size": 1.0, "lean": 1.0, "road_speed": 1.05,
     "stars_speed": 4, "stars_handling": 4},
    {"name": "Maxx", "desc": "Insane speed • Very loose • Tiny",
     "color": (255, 45, 170), "accent": (255, 255, 200),
     "accel": 1.55, "grip": 0.6, "max_steer": 0.7,
     "size": 0.82, "lean": 1.7, "road_speed": 1.35,
     "stars_speed": 5, "stars_handling": 1},
    {"name": "Tiger", "desc": "Heavy & planted • Super stable",
     "color": (25, 155, 65), "accent": (255, 255, 200),
     "accel": 0.85, "grip": 1.4, "max_steer": 1.25,
     "size": 1.22, "lean": 0.5, "road_speed": 0.9,
     "stars_speed": 3, "stars_handling": 5},
    {"name": "Phantom Pulse", "desc": "Agile • High grip • Small hitbox",
     "color": (140, 70, 255), "accent": (255, 255, 255),
     "accel": 1.2, "grip": 1.25, "max_steer": 1.3,
     "size": 0.78, "lean": 1.15, "road_speed": 1.15,
     "stars_speed": 4, "stars_handling": 4}
]

TRACKS = [
    {"name": "NARROW", "road_pct": 0.55, "player_mult": 0.85, "enemy_mult": 1.25, "spawn_mult": 0.85,
     "desc": "Narrow road, fast traffic", "stars": 5},
    {"name": "MEDIUM", "road_pct": 0.70, "player_mult": 1.0, "enemy_mult": 1.0, "spawn_mult": 1.0,
     "desc": "Standard road, balanced", "stars": 3},
    {"name": "WIDE", "road_pct": 0.85, "player_mult": 1.15, "enemy_mult": 0.85, "spawn_mult": 1.25,
     "desc": "Wide road, easier dodging", "stars": 2},
]

TRACK_KEYS = {pygame.K_q: 0, pygame.K_w: 1, pygame.K_e: 2}

# ============================================================
# LEADERBOARD DATA
# ============================================================
LEADERBOARD_FILE = "leaderboard.json"
LEADERBOARD_KEEP = 200

def _empty_boards():
    return {t["name"]: [] for t in TRACKS}

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                boards = _empty_boards()
                for name in boards:
                    boards[name] = data.get(name, [])[:LEADERBOARD_KEEP]
                return boards
            if isinstance(data, list):
                boards = _empty_boards()
                boards["MEDIUM"] = sorted(data, key=lambda x: x.get("time", 0), reverse=True)[:LEADERBOARD_KEEP]
                return boards
        except Exception:
            pass
    return _empty_boards()

def save_leaderboard(data):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def add_to_leaderboard(username, survival_time, score, car_name, track_name):
    boards = load_leaderboard()
    board = boards.get(track_name, [])
    entry = {
        "username": username,
        "time": round(survival_time, 2),
        "score": score,
        "car": car_name,
        "track": track_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    board.append(entry)
    board = sorted(board, key=lambda x: x["time"], reverse=True)[:LEADERBOARD_KEEP]
    boards[track_name] = board
    save_leaderboard(boards)

    player_rank = None
    for i, e in enumerate(board):
        if e["username"] == username and abs(e["time"] - entry["time"]) < 0.15:
            player_rank = i + 1
            entry = e
            break
    if player_rank is None:
        player_rank = len(board)
    return board, player_rank, entry

# ============================================================
# SOUND EFFECTS
# ============================================================
def make_noise(duration, volume=0.5, sample_rate=44100):
    n = int(sample_rate * duration)
    buf = array.array('h')
    for i in range(n):
        fade = 1.0 - (i / n)
        val = int(volume * fade * 32767 * (random.random() * 2 - 1))
        buf.append(val)
        buf.append(val)
    return pygame.mixer.Sound(buffer=buf)

crash_sound = make_noise(0.6, 0.55)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_sound(filename, volume):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        snd = pygame.mixer.Sound(path)
        snd.set_volume(volume)
        return snd
    except Exception:
        return None

engine_sound = load_sound("f1sound.wav", 0.5)
crash_sound_file = load_sound("fahh.wav", 0.8)
turn_sound = load_sound("turn_sound.wav", 0.45)

TURN_CHANNEL = pygame.mixer.Channel(3)
turn_playing = False

# ============================================================
# BACKGROUND MUSIC
# ============================================================
BGM_LIST = [
    {"file": "Odd Mob, FAST BOY - Super High (Extended Mix).mp3", "name": "Super High"},
    {"file": "Skrillex & Flava D ft. TAICHU - ID.mp3", "name": "ID"},
    {"file": "F1.mp3", "name": "F1"},
    {"file": "Öwnboss - Pop This (Extended Mix).mp3", "name": "Pop This"},
    {"file": "Disco Lines & Tinashe - No Broke Boys (Extended Mix).mp3", "name": "No Broke Boys"},
    {"file": "aespa - Whiplash (lime Flip).mp3", "name": "Whiplash"},
    {"file": "BABYMONSTER - SHEESH (MUNCHIE Trap Edit).mp3", "name": "SHEESH"},
    {"file": "Soma.mp3", "name": "Soma"},
    {"file": "Section.mp3", "name": "Section"},
    {"file": "SPIRAL.mp3", "name": "SPIRAL"},
]
DEFAULT_BGM = 0
bgm_playing = False

def play_bgm(index=None):
    global bgm_playing
    if index is None:
        index = DEFAULT_BGM
    for offset in range(len(BGM_LIST)):
        i = (index + offset) % len(BGM_LIST)
        path = os.path.join(BASE_DIR, BGM_LIST[i]["file"])
        if not os.path.exists(path):
            continue
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(0.35)
            pygame.mixer.music.play(-1)
            bgm_playing = True
            return True
        except Exception:
            continue
    bgm_playing = False
    return False

def stop_bgm():
    global bgm_playing
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
    bgm_playing = False

# ============================================================
# FACE DETECTION AND IMAGE PROCESSING
# ============================================================
def _apply_rounded_corners(surf, radius=8):
    w, h = surf.get_size()
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=radius)
    out = surf.copy()
    out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return out

def load_face_image(target_w, target_h):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select photo (Face detection auto-crops photo)",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")]
    )
    root.destroy()
    if not path:
        return None
    try:
        img_cv = cv2.imread(path)
        if img_cv is None:
            return None
        try:
            from PIL import Image, ImageOps
            img_cv = cv2.cvtColor(np.array(ImageOps.exif_transpose(Image.open(path))), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    except Exception as e:
        print("❌ Image read error:", e)
        return None
    return image_to_face_surface(img_cv, target_w, target_h)

def image_to_face_surface(img_cv, target_w, target_h):
    try:
        h_img, w_img = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        cascades = [
            ("frontalface_default", "haarcascade_frontalface_default.xml"),
            ("frontalface_alt", "haarcascade_frontalface_alt2.xml"),
            ("profileface", "haarcascade_profileface.xml"),
        ]
        best_face = None
        best_area = 0
        for name, xml in cascades:
            try:
                cascade_path = cv2.data.haarcascades + xml
                if not os.path.exists(cascade_path):
                    continue
                cascade = cv2.CascadeClassifier(cascade_path)
                faces = cascade.detectMultiScale(gray_eq, scaleFactor=1.05, minNeighbors=4,
                                                 minSize=(max(30, int(min(h_img, w_img) * 0.05)),) * 2)
                for (x, y, w, h) in faces:
                    area = w * h
                    if area > best_area:
                        best_area = area
                        best_face = (x, y, w, h, name)
            except Exception:
                pass

        if best_face is not None:
            x, y, w, h, src = best_face
            pad_x = int(w * 0.45)
            pad_top = int(h * 0.55)
            pad_bot = int(h * 0.55)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_top)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_bot)
            cw = x2 - x1
            ch = y2 - y1
            if cw < int(w * 1.2) or ch < int(h * 1.2):
                cx = x + w // 2
                cy = y + h // 2
                side = int(max(w, h) * 2.2)
                half = side // 2
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(w_img, cx + half)
                y2 = min(h_img, cy + half)
            face_img = img_cv[y1:y2, x1:x2]
        else:
            size = min(w_img, h_img)
            x1 = (w_img - size) // 2
            y1 = (h_img - size) // 2
            face_img = img_cv[y1:y1 + size, x1:x1 + size]

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        fh, fw = face_img.shape[:2]
        target_h_inner = target_h - 55
        target_w_inner = target_w - 22
        ratio = min(target_w_inner / fw, target_h_inner / fh)
        new_w = max(1, int(fw * ratio))
        new_h = max(1, int(fh * ratio))
        face_img = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        surf = pygame.surfarray.make_surface(np.transpose(face_img, (1, 0, 2)))
        surf = surf.convert_alpha()
        surf = _apply_rounded_corners(surf, radius=int(min(new_w, new_h) * 0.12))
        return surf
    except Exception as e:
        print("❌ Face extraction error:", e)
        return None

def pick_face():
    global player_face
    face = load_face_image(CAR_W_BASE, CAR_H_BASE)
    if face:
        player_face = face

def _open_camera():
    backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if os.name == "nt" else (cv2.CAP_ANY,)
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None

def _camera_hint():
    for _ in range(60):
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                return
        screen.fill((12, 14, 22))
        blit_cx(big_font.render("CAMERA NOT FOUND", True, RED), WIDTH // 2, HEIGHT // 2 - int(36 * SCALE))
        blit_cx(small_font.render("Upload an image instead   |   any key to close", True, (170, 185, 205)),
                WIDTH // 2, HEIGHT // 2 + int(29 * SCALE))
        pygame.display.flip()
        pygame.time.delay(33)

def capture_face(target_w, target_h):
    global capture_active
    cap = _open_camera()
    if cap is None:
        _camera_hint()
        return None

    view = pygame.Rect(0, 0, int(WIDTH * 0.62), int(HEIGHT * 0.68))
    view.center = (WIDTH // 2, HEIGHT // 2)
    guide_w = int(view.h * 0.5)
    guide = pygame.Rect(0, 0, guide_w, int(guide_w * 4 / 3))
    guide.center = view.center

    shot, fails = None, 0
    capture_active = True
    try:
        while shot is None:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 60:
                    return None
                pygame.time.delay(33)
                continue
            fails = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None
                if (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE) or \
                        (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
                    shot = frame

            screen.fill((10, 12, 18))
            fh, fw = frame.shape[:2]
            k = min(view.w / fw, view.h / fh)
            small = cv2.resize(frame, (max(1, int(fw * k)), max(1, int(fh * k))), interpolation=cv2.INTER_AREA)
            pic = pygame.surfarray.make_surface(np.transpose(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), (1, 0, 2)))
            pic_rect = pic.get_rect(center=view.center)
            screen.blit(pic, pic_rect)
            pygame.draw.rect(screen, (70, 210, 240), pic_rect.inflate(6, 6), max(1, int(2 * SCALE)),
                             border_radius=int(10 * SCALE))
            pygame.draw.rect(screen, (70, 210, 240), guide, max(1, int(3 * SCALE)), border_radius=int(70 * SCALE))
            blit_cx(big_font.render("TAKE A PHOTO", True, WHITE), WIDTH // 2, view.y - int(90 * SCALE))
            blit_cx(small_font.render("Put your face in the box   |   SPACE = shoot   |   ESC = cancel",
                                      True, (170, 190, 215)), WIDTH // 2, view.bottom + int(30 * SCALE))
            pygame.display.flip()
            pygame.time.delay(33)
    finally:
        capture_active = False
        cap.release()

    screen.fill(WHITE)
    pygame.display.flip()
    pygame.time.delay(90)
    return image_to_face_surface(shot, target_w, target_h)

face_pick_open = False
capture_active = False

FACE_PICK_W = int(600 * SCALE)
FACE_PICK_H = int(320 * SCALE)
FACE_PICK_RECT = pygame.Rect(0, 0, FACE_PICK_W, FACE_PICK_H)
FACE_PICK_RECT.center = (WIDTH // 2, HEIGHT // 2)
FACE_PICK_UPLOAD_RECT = pygame.Rect(FACE_PICK_RECT.x + int(48 * SCALE), FACE_PICK_RECT.y + int(104 * SCALE),
                                    FACE_PICK_W - int(96 * SCALE), int(70 * SCALE))
FACE_PICK_CAMERA_RECT = FACE_PICK_UPLOAD_RECT.move(0, int(90 * SCALE))

def open_face_picker():
    global face_pick_open
    face_pick_open = True

def close_face_picker():
    global face_pick_open
    face_pick_open = False

def shoot_face():
    global player_face
    face = capture_face(CAR_W_BASE, CAR_H_BASE)
    if face:
        player_face = face

def draw_face_picker(mouse_pos):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 175))
    screen.blit(overlay, (0, 0))

    panel = FACE_PICK_RECT
    pygame.draw.rect(screen, (0, 0, 0), panel.move(0, int(8 * SCALE)), border_radius=int(22 * SCALE))
    pygame.draw.rect(screen, (20, 27, 43), panel, border_radius=int(22 * SCALE))
    pygame.draw.rect(screen, (66, 200, 235), panel, max(1, int(2 * SCALE)), border_radius=int(22 * SCALE))
    blit_cx(mid_font.render("SET YOUR PHOTO", True, WHITE), panel.centerx, panel.y + int(34 * SCALE))

    for rect, label, tint in ((FACE_PICK_UPLOAD_RECT, "Upload image    (1)", UPLOAD_NORMAL),
                              (FACE_PICK_CAMERA_RECT, "Live camera    (2)", BGM_SEL)):
        fill = shade(tint, 1.3) if rect.collidepoint(mouse_pos) else tint
        button(rect, fill, WHITE, radius=int(14 * SCALE), shadow=int(4 * SCALE), border_w=int(2 * SCALE))
        blit_center(font.render(label, True, WHITE), *rect.center)

    blit_cx(tiny_font.render("ESC to cancel", True, (140, 158, 186)),
            panel.centerx, panel.bottom - int(38 * SCALE))

# ============================================================
# FONTS
# ============================================================
font = pygame.font.Font(None, int(28 * SCALE))
big_font = pygame.font.Font(None, int(56 * SCALE))
title_font = pygame.font.Font(None, int(68 * SCALE))
small_font = pygame.font.Font(None, int(20 * SCALE))
tiny_font = pygame.font.Font(None, int(16 * SCALE))
mid_font = pygame.font.Font(None, int(32 * SCALE))
btn_font = pygame.font.Font(None, int(36 * SCALE))
skill_font = pygame.font.Font(None, int(22 * SCALE))
log_font = pygame.font.SysFont("monospace", int(13 * SCALE))

# ============================================================
# GAME STATE CONTROL
# ============================================================
STATE_USERNAME = "username"
STATE_SELECT = "select"
STATE_PLAY = "play"
STATE_GAMEOVER = "gameover"

state = STATE_USERNAME
username = ""
input_active = True
player_face = None
selected_car_idx = 0
selected_track = 1
current_car = None
leaderboard = load_leaderboard()
player_rank = None
player_entry = None
last_survival_time = 0.0

car_x = 0.0
car_y = 0.0
car_vx = 0.0
car_w = CAR_W_BASE
car_h = CAR_H_BASE
steer_input = 0.0
drift_amount = 0.0
drift_angle = 0.0
obstacles = []
pedestrians = []
coins = []
obstacle_speed = 0.0
spawn_timer = 0.0
ped_timer = 0
coin_timer = 0
base_spawn_delay = 48.0
score = 0
high_score = 0
combo = 0
multiplier = 1
flash = 0
shake = 0
particles = []
speed_lines = []
skid_particles = []
smoke_particles = []
near_miss_timer = 0
game_time = 0.0

ROAD_LEFT = 0
ROAD_RIGHT = 0
ROAD_WIDTH = 0

MIN_SPEED = 3.0 * SCALE
MAX_SPEED = 17.0 * SCALE
SPEED_CHANGE_RATE = 6.0 * SCALE

GRIP_BOOST = 1.35

START_LIVES = 4
lives = START_LIVES
INVULN_MS = 2000
invuln_until_ms = 0

SPEED_LOCK_AT = 67.0
speed_locked = False

death_trap_active = False

# MIDI State
throttle_held = False
throttle_speed = 0.0
THROTTLE_MAX = 6.0
THROTTLE_ACCEL = 0.08
THROTTLE_DECAY = 0.04
THROTTLE_NOTE = 56

drift_held = False
DRIFT_MAX_ANGLE = 12
DRIFT_ANGLE_STEP = 4.5
DRIFT_SPEED_MULT = 1.5
DRIFT_NOTE = 52

MIDI_NOTE_NITRO = 53
MIDI_NOTE_SHIELD = 54
MIDI_NOTE_LASER = 55
MIDI_NOTE_PHOTO = 58
MIDI_NOTE_CAMERA = 59
MIDI_NOTE_DEATH = 57

NITRO_DURATION = 2000
NITRO_CD = 20000
SHIELD_DURATION = 4000
SHIELD_CD = 30000
LASER_DURATION = 300
LASER_CD = 25000
nitro_active = False
nitro_end_ms = 0
nitro_cd_end_ms = 0
shield_active = False
shield_end_ms = 0
shield_cd_end_ms = 0
laser_active = False
laser_end_ms = 0
laser_cd_end_ms = 0
laser_x = 0

THROTTLE_ENEMY_BOOST = 0.88

def get_now_ms():
    return pygame.time.get_ticks()

def poll_midi():
    global throttle_held, drift_held, death_trap_active
    if MIDI_INPUT is None:
        return
    while MIDI_INPUT.poll():
        for message, _ in MIDI_INPUT.read(10):
            status, data1, data2 = message[0], message[1], message[2]
            note_on = 0x90 <= status <= 0x9F and data2 > 0
            note_off = 0x80 <= status <= 0x8F or (0x90 <= status <= 0x9F and data2 == 0)
            if face_pick_open:
                if note_on and data1 == MIDI_NOTE_PHOTO:
                    close_face_picker()
                    pick_face()
                elif note_on and data1 == MIDI_NOTE_CAMERA:
                    close_face_picker()
                    shoot_face()
            elif note_on and state == STATE_PLAY:
                if data1 == MIDI_NOTE_NITRO:
                    try_activate_nitro()
                elif data1 == MIDI_NOTE_SHIELD:
                    try_activate_shield()
                elif data1 == MIDI_NOTE_LASER:
                    try_activate_laser()
                elif data1 == THROTTLE_NOTE:
                    throttle_held = True
                elif data1 == DRIFT_NOTE:
                    drift_held = True
                elif data1 == MIDI_NOTE_DEATH:
                    death_trap_active = True
            elif note_on and data1 == MIDI_NOTE_CAMERA and state in (STATE_USERNAME, STATE_SELECT):
                shoot_face()
            elif note_on and data1 == MIDI_NOTE_PHOTO and state in (STATE_USERNAME, STATE_SELECT):
                open_face_picker()
            elif note_off:
                if data1 == THROTTLE_NOTE:
                    throttle_held = False
                elif data1 == DRIFT_NOTE:
                    drift_held = False

def try_activate_nitro():
    global nitro_active, nitro_end_ms, nitro_cd_end_ms
    now = get_now_ms()
    if nitro_active or now < nitro_cd_end_ms:
        return False
    nitro_active = True
    nitro_end_ms = now + NITRO_DURATION
    nitro_cd_end_ms = now + NITRO_DURATION + NITRO_CD
    return True

def try_activate_shield():
    global shield_active, shield_end_ms, shield_cd_end_ms
    now = get_now_ms()
    if shield_active or now < shield_cd_end_ms:
        return False
    shield_active = True
    shield_end_ms = now + SHIELD_DURATION
    shield_cd_end_ms = now + SHIELD_DURATION + SHIELD_CD
    return True

def try_activate_laser():
    global laser_active, laser_end_ms, laser_cd_end_ms, laser_x, score
    now = get_now_ms()
    if laser_active or now < laser_cd_end_ms:
        return False
    laser_active = True
    laser_end_ms = now + LASER_DURATION
    laser_cd_end_ms = now + LASER_DURATION + LASER_CD
    laser_x = car_x + car_w // 2
    lw = int(40 * SCALE)
    lb = laser_x - lw // 2
    rb = laser_x + lw // 2
    for e in obstacles[:]:
        if e["kind"] == "car":
            if not (e["x"] + e["w"] < lb or e["x"] > rb):
                obstacles.remove(e)
                score += 1
    return True

def update_skills():
    global nitro_active, shield_active, laser_active
    now = get_now_ms()
    if nitro_active and now >= nitro_end_ms:
        nitro_active = False
    if shield_active and now >= shield_end_ms:
        shield_active = False
    if laser_active and now >= laser_end_ms:
        laser_active = False

def update_track_params():
    global ROAD_LEFT, ROAD_RIGHT, ROAD_WIDTH
    t = TRACKS[selected_track]
    road_w = max(int(WIDTH * t["road_pct"]), int(WIDTH * 0.4))
    ROAD_LEFT = (WIDTH - road_w) // 2
    ROAD_RIGHT = ROAD_LEFT + road_w
    ROAD_WIDTH = road_w

update_track_params()

def apply_car_stats(car):
    global car_w, car_h
    car_w = int(CAR_W_BASE * car["size"])
    car_h = int(CAR_H_BASE * car["size"])

# ============================================================
# SMOKE PARTICLES
# ============================================================
class SmokeParticle:
    def __init__(self, x, y, vx, vy, color=(200, 200, 200), grow_mul=1.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = random.uniform(3 * SCALE, 7 * SCALE)
        self.life = 1.0
        self.decay = random.uniform(0.02, 0.04)
        self.grow = random.uniform(0.1, 0.25) * SCALE * grow_mul
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.03 * SCALE
        self.vx *= 0.97
        self.radius += self.grow
        self.life -= self.decay

    def draw(self, surface):
        if self.life <= 0:
            return
        alpha = int(180 * self.life)
        r, g, b = self.color
        surf = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
        pygame.draw.circle(surf, (r, g, b, alpha),
                           (int(self.radius), int(self.radius)), int(self.radius))
        surface.blit(surf, (self.x - self.radius, self.y - self.radius))

def spawn_smoke(direction, px, py, cw, ch):
    rear_left_x = px + int(2 * SCALE)
    rear_right_x = px + cw - int(2 * SCALE)
    rear_y = py + ch - int(6 * SCALE)
    if drift_held and direction != 0:
        count_range = (4, 6)
        color = (240, 240, 240)
        grow = 2.0
    elif nitro_active or throttle_speed > 2.0:
        count_range = (3, 5)
        color = (120, 200, 255) if nitro_active else (255, 180, 100)
        grow = 1.6
    else:
        count_range = (1, 2)
        color = (200, 200, 200)
        grow = 1.0
    for wheel_x in (rear_left_x, rear_right_x):
        for _ in range(random.randint(*count_range)):
            vx = -direction * random.uniform(0.8, 2.0) * SCALE + random.uniform(-0.5, 0.5) * SCALE
            vy = random.uniform(-1.5, -0.4) * SCALE
            px_ = wheel_x + random.uniform(-3, 3) * SCALE
            py_ = rear_y + random.uniform(-3, 3) * SCALE
            smoke_particles.append(SmokeParticle(px_, py_, vx, vy, color, grow))

def update_smoke():
    global smoke_particles
    smoke_particles = [p for p in smoke_particles if p.life > 0]
    for p in smoke_particles:
        p.update()

def draw_smoke():
    for p in smoke_particles:
        p.draw(screen)

# ============================================================
# SOUND MANAGEMENT
# ============================================================
engine_channel = None
engine_playing = False

def start_engine_sound():
    global engine_channel, engine_playing
    if engine_sound and not engine_playing:
        engine_channel = engine_sound.play(loops=-1)
        if engine_channel:
            engine_channel.set_volume(0.5)
            engine_playing = True

def stop_engine_sound():
    global engine_channel, engine_playing
    if engine_channel and engine_playing:
        engine_channel.fadeout(120)
        engine_playing = False
        engine_channel = None

def stop_all_sounds():
    global engine_channel, engine_playing, turn_playing
    if engine_channel:
        try:
            engine_channel.stop()
        except:
            pass
    engine_playing = False
    engine_channel = None
    try:
        TURN_CHANNEL.stop()
    except:
        pass
    turn_playing = False

def play_turn_sound(direction):
    global turn_playing
    if turn_sound is None:
        return
    if direction == 0:
        if turn_playing:
            TURN_CHANNEL.fadeout(80)
            turn_playing = False
        return
    if turn_playing and TURN_CHANNEL.get_busy():
        return
    try:
        TURN_CHANNEL.play(turn_sound, loops=-1)
        if direction < 0:
            TURN_CHANNEL.set_volume(0.55, 0.15)
        else:
            TURN_CHANNEL.set_volume(0.15, 0.55)
        turn_playing = True
    except Exception:
        pass

# ============================================================
# GAME RUN LOOP LOGIC
# ============================================================
def start_game():
    global state, car_x, car_y, car_vx, obstacles, pedestrians, coins
    global score, combo, multiplier, obstacle_speed, spawn_timer, ped_timer, coin_timer
    global particles, speed_lines, skid_particles, smoke_particles
    global flash, shake, steer_input, drift_amount, drift_angle, game_time
    global nitro_active, shield_active, laser_active
    global nitro_end_ms, nitro_cd_end_ms, shield_end_ms, shield_cd_end_ms
    global laser_end_ms, laser_cd_end_ms, laser_x
    global throttle_speed, throttle_held, drift_held
    global lives, invuln_until_ms, speed_locked, death_trap_active

    update_track_params()
    current_car_local = CARS[selected_car_idx]
    apply_car_stats(current_car_local)
    car_x = WIDTH // 2 - car_w // 2
    car_y = HEIGHT - car_h - int(70 * SCALE)
    car_vx = 0.0
    obstacles = []
    pedestrians = []
    coins = []
    score = 0
    combo = 0
    multiplier = 1
    obstacle_speed = 4.5 * SCALE * current_car_local["road_speed"]
    spawn_timer = 0.0
    ped_timer = 0
    coin_timer = 0
    particles = []
    speed_lines = []
    skid_particles = []
    smoke_particles = []
    flash = 0
    shake = 0
    steer_input = 0.0
    drift_amount = 0.0
    drift_angle = 0.0
    game_time = 0.0
    throttle_speed = 0.0
    throttle_held = False
    drift_held = False
    nitro_active = shield_active = laser_active = False
    nitro_end_ms = nitro_cd_end_ms = 0
    shield_end_ms = shield_cd_end_ms = 0
    laser_end_ms = laser_cd_end_ms = laser_x = 0
    lives = START_LIVES
    invuln_until_ms = 0
    speed_locked = False
    death_trap_active = False
    state = STATE_PLAY

    start_engine_sound()
    stop_bgm()
    play_bgm()

def is_invulnerable():
    return get_now_ms() < invuln_until_ms

def lose_life(reset_combo=False):
    global lives, invuln_until_ms, flash, shake, combo, multiplier, car_x, car_vx
    global obstacles, pedestrians

    if is_invulnerable() or shield_active:
        return

    lives -= 1
    create_particles(car_x, car_y, 30)
    flash = 18
    shake = 22

    if reset_combo:
        combo = 0
        multiplier = 1

    if lives <= 0:
        end_run(reset_combo=False)
        return

    crash_sound.play()
    if crash_sound_file:
        crash_sound_file.play()
    invuln_until_ms = get_now_ms() + INVULN_MS
    car_x = WIDTH // 2 - car_w // 2
    car_vx = 0.0
    obstacles = [o for o in obstacles if o["y"] + o["h"] < car_y - 80]
    pedestrians = [p for p in pedestrians if p[1] + int(48 * SCALE) < car_y - 80]

def full_reset():
    global state, username, player_face, selected_car_idx, player_rank, player_entry
    state = STATE_USERNAME
    username = ""
    player_face = None
    selected_car_idx = 0
    player_rank = None
    player_entry = None
    stop_all_sounds()
    stop_bgm()

def end_run(reset_combo=False):
    global state, last_survival_time, leaderboard, player_rank, player_entry, high_score
    global flash, shake, combo, multiplier
    last_survival_time = game_time
    track_name = TRACKS[selected_track]["name"]
    full_board, player_rank, player_entry = add_to_leaderboard(
        username, game_time, score, current_car["name"], track_name
    )
    leaderboard = full_board
    high_score = max(high_score, score)
    state = STATE_GAMEOVER
    crash_sound.play()
    if crash_sound_file:
        crash_sound_file.play()
    stop_all_sounds()
    stop_bgm()
    create_particles(car_x, car_y, 45)
    flash = 22
    shake = 30
    if reset_combo:
        combo = 0
        multiplier = 1

def spawn_obstacle():
    global death_trap_active
    if death_trap_active:
        num_cars = random.randint(2, 3)
        for _ in range(num_cars):
            color = random.choice([RED, ORANGE, PURPLE, CYAN, (220, 70, 150)])
            w = int(CAR_W_BASE * random.uniform(0.9, 1.1))
            h = int(CAR_H_BASE * random.uniform(0.9, 1.1))
            x = random.randint(ROAD_LEFT + 10, ROAD_RIGHT - w - 10)
            overlap = False
            for ob in obstacles:
                if ob["y"] < 0 and abs(ob["x"] - x) < (w + int(20 * SCALE)):
                    overlap = True
                    break
            if overlap:
                continue
            obstacles.append({
                "x": x, "y": -h, "w": w, "h": h,
                "kind": "car", "color": color, "passed": False
            })
        return

    kind = random.choices(["car", "car", "car", "barrier"], weights=[3, 3, 2, 2])[0]
    if kind == "car":
        color = random.choice([RED, ORANGE, PURPLE, CYAN, (220, 70, 150)])
        w = int(CAR_W_BASE * random.uniform(0.9, 1.1))
        h = int(CAR_H_BASE * random.uniform(0.9, 1.1))
    else:
        color = None
        w = random.randint(int(100 * SCALE), int(200 * SCALE))
        h = int(36 * SCALE)
    x = random.randint(ROAD_LEFT + 40, ROAD_RIGHT - w - 40)
    obstacles.append({"x": x, "y": -h, "w": w, "h": h, "kind": kind, "color": color, "passed": False})

def spawn_pedestrian():
    x = random.randint(ROAD_LEFT + 40, ROAD_RIGHT - int(26 * SCALE) - 40)
    pedestrians.append([x, -int(48 * SCALE) - 10, random.choice([-1, 1])])

def spawn_coin():
    x = random.randint(ROAD_LEFT + 40, ROAD_RIGHT - int(24 * SCALE) - 40)
    coins.append([x, -int(24 * SCALE) - 10])

def create_particles(x, y, amount=35):
    for _ in range(amount):
        particles.append({
            "x": x + car_w // 2, "y": y + car_h // 2,
            "vx": random.uniform(-9, 9), "vy": random.uniform(-12, 5),
            "life": random.randint(25, 50),
            "color": random.choice([RED, ORANGE, YELLOW, WHITE, (255, 100, 50)])
        })

def rect_collide(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

# ============================================================
# RENDERING: ROAD & ENVIRONMENT
# ============================================================
_ROAD_NOISE_SURFACE = None

def _init_road_noise():
    global _ROAD_NOISE_SURFACE
    if _ROAD_NOISE_SURFACE is not None:
        return
    surf = pygame.Surface((ROAD_WIDTH, HEIGHT), pygame.SRCALPHA)
    for _ in range(2000):
        nx = random.randint(0, ROAD_WIDTH - 1)
        ny = random.randint(0, HEIGHT - 1)
        gray = random.randint(50, 90)
        alpha = random.randint(30, 80)
        pygame.draw.circle(surf, (gray, gray, gray, alpha), (nx, ny), random.choice([1, 1, 1, 2]))
    _ROAD_NOISE_SURFACE = surf

def draw_road(offset, shake_x=0, shake_y=0):
    screen.fill(GRASS)
    pygame.draw.rect(screen, GRASS_DARK, (ROAD_LEFT - 14, 0, 16, HEIGHT))
    pygame.draw.rect(screen, GRASS_DARK, (ROAD_RIGHT - 2, 0, 16, HEIGHT))
    pygame.draw.rect(screen, ROAD, (ROAD_LEFT, 0, ROAD_WIDTH, HEIGHT))
    _init_road_noise()
    noise_y = int(offset) % HEIGHT
    screen.blit(_ROAD_NOISE_SURFACE, (ROAD_LEFT, noise_y - HEIGHT))
    screen.blit(_ROAD_NOISE_SURFACE, (ROAD_LEFT, noise_y))
    pygame.draw.rect(screen, YELLOW, (ROAD_LEFT + 7, 0, 9, HEIGHT))
    pygame.draw.rect(screen, YELLOW, (ROAD_RIGHT - 16, 0, 9, HEIGHT))
    dash_h = int(32 * SCALE)
    gap = int(30 * SCALE)
    dash_total = dash_h + gap
    y = offset % dash_total - dash_total
    while y < HEIGHT:
        pygame.draw.rect(screen, LANE, (WIDTH // 2 - 5 + shake_x, y + shake_y, 10, dash_h))
        y += dash_total

def draw_barrier(surf, x, y, w, h):
    pygame.draw.ellipse(surf, (0, 0, 0, 80), (x - 5, y + h - 8, w + 10, 16))
    pygame.draw.rect(surf, (170, 25, 25), (x, y, w, h), border_radius=7)
    for i in range(0, w, 18):
        pygame.draw.rect(surf, YELLOW, (x + i, y + 5, 9, h - 10))

# ============================================================
# RENDERING: CARS & PEDESTRIANS
# ============================================================
def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)

def draw_car_shape(surf, x, y, car, cw, ch, face_img=None, is_player=False):
    color = car["color"]
    accent = car.get("accent", (255, 255, 255))
    color_dark = shade(color, 0.55)
    color_light = shade(color, 1.35)

    if is_player and shield_active:
        now = get_now_ms()
        pulse = int(15 * abs((now % 600) / 300 - 1))
        for i in range(3, 0, -1):
            alpha = 100 - (i - 1) * 25
            rex = i * int(8 * SCALE) + pulse
            sf = pygame.Surface((cw + rex * 2, ch + rex * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(sf, (120, 255, 180, alpha), (0, 0, cw + rex * 2, ch + rex * 2), 4)
            surf.blit(sf, (x - rex, y - rex))

    if is_player and nitro_active:
        now = get_now_ms()
        pulse = 6 + int(6 * abs((now % 400) / 200 - 1))
        sf = pygame.Surface((cw + pulse * 2, ch + pulse * 2), pygame.SRCALPHA)
        pygame.draw.rect(sf, (80, 200, 255, 100), (0, 0, cw + pulse * 2, ch + pulse * 2),
                         border_radius=int(14 * SCALE))
        surf.blit(sf, (x - pulse, y - pulse))

    pygame.draw.rect(surf, SHADOW_COLOR, (x + int(4 * SCALE), y + int(4 * SCALE), cw, ch),
                     border_radius=int(10 * SCALE))

    w_w = int(9 * SCALE)
    w_h = int(20 * SCALE)
    for wx, wy in [(x - int(3 * SCALE), y + int(12 * SCALE)),
                   (x + cw - int(6 * SCALE), y + int(12 * SCALE)),
                   (x - int(3 * SCALE), y + ch - int(32 * SCALE)),
                   (x + cw - int(6 * SCALE), y + ch - int(32 * SCALE))]:
        pygame.draw.rect(surf, (15, 15, 15), (wx, wy, w_w, w_h), border_radius=int(3 * SCALE))
        pygame.draw.rect(surf, (45, 45, 45),
                         (wx + int(1 * SCALE), wy + int(2 * SCALE),
                          w_w - int(2 * SCALE), w_h - int(4 * SCALE)), border_radius=int(2 * SCALE))
        pygame.draw.rect(surf, (160, 160, 170),
                         (wx + int(2 * SCALE), wy + w_h // 2 - int(3 * SCALE),
                          w_w - int(4 * SCALE), int(6 * SCALE)), border_radius=int(2 * SCALE))

    pygame.draw.rect(surf, color, (x, y, cw, ch), border_radius=int(10 * SCALE))
    hs = pygame.Surface((cw, ch // 2), pygame.SRCALPHA)
    pygame.draw.rect(hs, (*color_light, 90), (0, 0, cw, ch // 2), border_radius=int(10 * SCALE))
    surf.blit(hs, (x, y))
    ss = pygame.Surface((cw, ch // 3), pygame.SRCALPHA)
    pygame.draw.rect(ss, (*color_dark, 150), (0, 0, cw, ch // 3), border_radius=int(10 * SCALE))
    surf.blit(ss, (x, y + ch - ch // 3))
    pygame.draw.rect(surf, color_dark, (x, y, cw, ch), int(2 * SCALE), border_radius=int(10 * SCALE))

    stripe_w = int(10 * SCALE)
    stripe_x = x + cw // 2 - stripe_w // 2
    pygame.draw.rect(surf, accent, (stripe_x, y + int(4 * SCALE), stripe_w, ch - int(8 * SCALE)),
                     border_radius=int(3 * SCALE))

    wr = pygame.Rect(x + int(8 * SCALE), y + int(14 * SCALE), cw - int(16 * SCALE), int(22 * SCALE))
    pygame.draw.rect(surf, WINDOW_COLOR, wr, border_radius=int(5 * SCALE))
    pygame.draw.rect(surf, color_dark, wr, int(2 * SCALE), border_radius=int(5 * SCALE))

    lr = int(5 * SCALE)
    pygame.draw.circle(surf, (255, 250, 200), (x + int(9 * SCALE), y + int(10 * SCALE)), lr)
    pygame.draw.circle(surf, (255, 250, 200), (x + cw - int(9 * SCALE), y + int(10 * SCALE)), lr)
    tr = int(4 * SCALE)
    pygame.draw.circle(surf, (255, 60, 60), (x + int(9 * SCALE), y + ch - int(8 * SCALE)), tr)
    pygame.draw.circle(surf, (255, 60, 60), (x + cw - int(9 * SCALE), y + ch - int(8 * SCALE)), tr)

    if is_player and face_img is not None:
        hw, hh = face_img.get_size()
        hx = x + cw // 2 - hw // 2
        hy = y + int(12 * SCALE)
        pygame.draw.rect(surf, (255, 255, 255), (hx - 3, hy - 3, hw + 6, hh + 6), border_radius=6)
        pygame.draw.rect(surf, (0, 0, 0), (hx - 3, hy - 3, hw + 6, hh + 6), 2, border_radius=6)
        surf.blit(face_img, (hx, hy))

    if is_player and (nitro_active or throttle_speed > 2.0):
        fl = int(22 * SCALE + throttle_speed * SCALE * 2)
        fc = (100, 200, 255) if nitro_active else (255, 160, 80)
        pygame.draw.polygon(surf, fc, [(x + int(10 * SCALE), y + ch - int(2 * SCALE)),
                                       (x + int(7 * SCALE), y + ch + fl),
                                       (x + int(14 * SCALE), y + ch - int(2 * SCALE))])
        pygame.draw.polygon(surf, fc, [(x + cw - int(10 * SCALE), y + ch - int(2 * SCALE)),
                                       (x + cw - int(7 * SCALE), y + ch + fl),
                                       (x + cw - int(14 * SCALE), y + ch - int(2 * SCALE))])

def draw_car_rotated(x, y, car, cw, ch, face_img, is_player, angle):
    if abs(angle) < 0.5:
        draw_car_shape(screen, x, y, car, cw, ch, face_img, is_player)
        return
    ext_w = cw + int(40 * SCALE)
    ext_h = ch + int(40 * SCALE)
    temp = pygame.Surface((ext_w, ext_h), pygame.SRCALPHA)
    draw_car_shape(temp, int(20 * SCALE), int(20 * SCALE), car, cw, ch, face_img, is_player)
    rotated = pygame.transform.rotate(temp, angle)
    r_rect = rotated.get_rect(center=(x + cw // 2, y + ch // 2))
    screen.blit(rotated, r_rect.topleft)

def draw_enemy_car(x, y, cw, ch, color):
    pygame.draw.rect(screen, SHADOW_COLOR, (x + int(3 * SCALE), y + int(3 * SCALE), cw, ch),
                     border_radius=int(8 * SCALE))
    pygame.draw.rect(screen, color, (x, y, cw, ch), border_radius=int(8 * SCALE))
    pygame.draw.rect(screen, WINDOW_COLOR, (x + int(8 * SCALE), y + int(12 * SCALE),
                                            cw - int(16 * SCALE), int(20 * SCALE)), border_radius=int(4 * SCALE))
    pygame.draw.rect(screen, WINDOW_COLOR, (x + int(8 * SCALE), y + ch - int(30 * SCALE),
                                            cw - int(16 * SCALE), int(16 * SCALE)), border_radius=int(4 * SCALE))

def draw_pedestrian(x, y, pw, ph):
    pygame.draw.ellipse(screen, (0, 0, 0), (x + int(2 * SCALE), y + ph - int(4 * SCALE),
                                            pw - int(4 * SCALE), int(8 * SCALE)))
    lw = int(7 * SCALE)
    lh = int(20 * SCALE)
    pygame.draw.rect(screen, PED_PANTS, (x + int(4 * SCALE), y + int(28 * SCALE), lw, lh), border_radius=int(2 * SCALE))
    pygame.draw.rect(screen, PED_PANTS, (x + pw - int(11 * SCALE), y + int(28 * SCALE), lw, lh),
                     border_radius=int(2 * SCALE))
    pygame.draw.rect(screen, PED_CLOTHES, (x + int(2 * SCALE), y + int(14 * SCALE),
                                           pw - int(4 * SCALE), int(16 * SCALE)), border_radius=int(4 * SCALE))
    pygame.draw.circle(screen, PED_COLOR, (x + pw // 2, y + int(8 * SCALE)), int(8 * SCALE))

def draw_coin(x, y, size):
    pygame.draw.circle(screen, SHADOW_COLOR, (x + size // 2 + int(2 * SCALE),
                                              y + size // 2 + int(2 * SCALE)), size // 2)
    pygame.draw.circle(screen, COIN_COLOR, (x + size // 2, y + size // 2), size // 2)
    pygame.draw.circle(screen, COIN_EDGE, (x + size // 2, y + size // 2), size // 2, int(2 * SCALE))

def draw_stars(x, y, count, max_stars=5, center=None):
    r = max(4, int(7 * SCALE))
    step = int(17 * SCALE)
    if center is not None:
        x = center - (r * 2 + (max_stars - 1) * step) // 2
    for i in range(max_stars):
        cx = x + r + i * step
        cy = y + r
        pts = []
        for k in range(10):
            ang = -math.pi / 2 + k * math.pi / 5
            rad = r if k % 2 == 0 else r * 0.45
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        pygame.draw.polygon(screen, STAR_ON if i < count else STAR_OFF, pts)

def draw_laser():
    if not laser_active:
        return
    now = get_now_ms()
    rr = (laser_end_ms - now) / LASER_DURATION
    if rr <= 0:
        return
    w = int(30 * SCALE * rr) + 4
    a = int(220 * rr)
    gs = pygame.Surface((w * 3, HEIGHT), pygame.SRCALPHA)
    pygame.draw.rect(gs, (100, 200, 255, a // 3), (0, 0, w * 3, HEIGHT))
    screen.blit(gs, (laser_x - w * 3 // 2, 0))
    cs = pygame.Surface((w, HEIGHT), pygame.SRCALPHA)
    pygame.draw.rect(cs, (200, 240, 255, a), (0, 0, w, HEIGHT))
    screen.blit(cs, (laser_x - w // 2, 0))
    pygame.draw.rect(screen, (255, 255, 255), (laser_x - 2, 0, 4, HEIGHT))

def draw_skill_hud():
    now = get_now_ms()
    sw = int(150 * SCALE)
    sh = int(40 * SCALE)
    gap = int(6 * SCALE)
    bx = int(15 * SCALE)
    by = HEIGHT - int(500 * SCALE)

    def draw_one(rect, active, cd_end, duration, cd, color_active, label_ready, label_key):
        if active:
            r = max(0, (cd_end - now) / 1000.0)
            bg = color_active
            lb = f"{label_ready} {r:.1f}s"
            pr = max(0, min(1, (cd_end - now) / duration))
        elif now < cd_end:
            r = (cd_end - now) / 1000.0
            bg = (80, 80, 100)
            lb = f"{label_ready} CD {r:.1f}s"
            pr = 1 - r / (cd / 1000.0)
        else:
            bg = tuple(int(c * 0.7) for c in color_active)
            lb = f"{label_ready} {label_key}"
            pr = 1.0
        pygame.draw.rect(screen, (0, 0, 0), rect.move(0, 3), border_radius=int(8 * SCALE))
        pygame.draw.rect(screen, bg, rect, border_radius=int(8 * SCALE))
        pygame.draw.rect(screen, color_active, rect, int(2 * SCALE), border_radius=int(8 * SCALE))
        br = pygame.Rect(rect.x + 4, rect.bottom - 6, int((rect.w - 8) * max(0, min(1, pr))), 3)
        pygame.draw.rect(screen, color_active, br, border_radius=2)
        screen.blit(skill_font.render(lb, True, WHITE),
                    (rect.centerx - skill_font.size(lb)[0] // 2,
                     rect.centery - skill_font.size(lb)[1] // 2))

    nr = pygame.Rect(bx, by, sw, sh)
    draw_one(nr, nitro_active, nitro_end_ms, NITRO_DURATION, NITRO_CD, NITRO_COLOR, "NITRO", "[SPACE]")
    sr = pygame.Rect(bx, by + sh + gap, sw, sh)
    draw_one(sr, shield_active, shield_end_ms, SHIELD_DURATION, SHIELD_CD, SHIELD_COLOR, "SHIELD", "[SHIFT]")
    lr = pygame.Rect(bx, by + 2 * (sh + gap), sw, sh)
    draw_one(lr, laser_active, laser_end_ms, LASER_DURATION, LASER_CD, LASER_COLOR, "LASER", "[E]")

    if drift_held and abs(drift_angle) > 1:
        df = pygame.Rect(bx, by + 3 * (sh + gap), sw, sh)
        pygame.draw.rect(screen, (0, 0, 0), df.move(0, 3), border_radius=int(8 * SCALE))
        pygame.draw.rect(screen, (200, 100, 50), df, border_radius=int(8 * SCALE))
        pygame.draw.rect(screen, (255, 200, 100), df, int(2 * SCALE), border_radius=int(8 * SCALE))
        screen.blit(skill_font.render("DRIFTING", True, WHITE),
                    (df.centerx - skill_font.size("DRIFTING")[0] // 2,
                     df.centery - skill_font.size("DRIFTING")[1] // 2))

    if throttle_speed > 0.01:
        tw = int(200 * SCALE)
        th = int(20 * SCALE)
        tx = WIDTH // 2 - tw // 2
        ty = int(15 * SCALE)
        ratio = throttle_speed / THROTTLE_MAX
        pygame.draw.rect(screen, (30, 30, 30), (tx, ty, tw, th), border_radius=int(10 * SCALE))
        color = (255, int(255 * (1 - ratio)), 50)
        pygame.draw.rect(screen, color, (tx, ty, int(tw * ratio), th), border_radius=int(10 * SCALE))
        pygame.draw.rect(screen, (255, 255, 255), (tx, ty, tw, th), int(2 * SCALE), border_radius=int(10 * SCALE))
        throttle_label = f"THROTTLE {throttle_speed:.1f}  |  DIFF x{1.0 + ratio * THROTTLE_ENEMY_BOOST:.2f}"
        screen.blit(small_font.render(throttle_label, True, (255, 255, 255)),
                    (WIDTH // 2 - small_font.size(throttle_label)[0] // 2, ty + th + 5))

# ============================================================
# RENDERING: SERIAL UI CONSOLE LOG WINDOW
# ============================================================
def draw_serial_log_ui():
    """Renders a translucent serial terminal monitor window on the bottom right UI."""
    panel_w = int(320 * SCALE)
    panel_h = int(115 * SCALE)
    margin = int(15 * SCALE)
    
    panel_x = WIDTH - panel_w - margin
    panel_y = HEIGHT - panel_h - margin
    
    log_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    pygame.draw.rect(log_surf, (15, 20, 28, 200), (0, 0, panel_w, panel_h), border_radius=int(8 * SCALE))
    pygame.draw.rect(log_surf, (40, 180, 100, 180), (0, 0, panel_w, panel_h), int(1.5 * SCALE), border_radius=int(8 * SCALE))
    
    title_text = log_font.render("CONSOLE > SERIAL LOG", True, (40, 220, 100))
    log_surf.blit(title_text, (int(8 * SCALE), int(5 * SCALE)))
    
    pygame.draw.line(log_surf, (30, 80, 50, 180), (0, int(20 * SCALE)), (panel_w, int(20 * SCALE)), 1)
    
    start_y = int(24 * SCALE)
    line_h = int(17 * SCALE)
    
    logs = list(serial_log_buffer)
    if not logs:
        text = log_font.render("Waiting for serial data...", True, (120, 130, 140))
        log_surf.blit(text, (int(8 * SCALE), start_y))
    else:
        for idx, item in enumerate(logs):
            color = (80, 240, 120) if idx == len(logs) - 1 else (150, 160, 170)
            text = log_font.render(item, True, color)
            log_surf.blit(text, (int(8 * SCALE), start_y + idx * line_h))
            
    screen.blit(log_surf, (panel_x, panel_y))

# ============================================================
# MENU HELPER FUNCTIONS
# ============================================================
def blit_center(surf, cx, cy, target=None):
    (target or screen).blit(surf, (cx - surf.get_width() // 2, cy - surf.get_height() // 2))

def blit_cx(surf, cx, top, target=None):
    (target or screen).blit(surf, (cx - surf.get_width() // 2, top))

def _mix(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def _soft(surf, factor=10):
    w, h = surf.get_size()
    small = pygame.transform.smoothscale(surf, (max(1, w // factor), max(1, h // factor)))
    return pygame.transform.smoothscale(small, (w, h))

def _glow_text(text, font_obj, color, glow_color, blur=10, spread=14, passes=2):
    core = font_obj.render(text, True, color)
    pad = int(spread * SCALE)
    halo = pygame.Surface((core.get_width() + pad * 2, core.get_height() + pad * 2), pygame.SRCALPHA)
    halo.blit(font_obj.render(text, True, glow_color), (pad, pad))
    halo = _soft(halo, blur)
    out = pygame.Surface(halo.get_size(), pygame.SRCALPHA)
    for _ in range(passes):
        out.blit(halo, (0, 0))
    out.blit(core, (pad, pad))
    return out, pad

def _glow_bar(target, cx, y, width=180):
    bar_w, bar_h, pad = int(width * SCALE), max(2, int(4 * SCALE)), int(24 * SCALE)
    bar = pygame.Surface((bar_w + pad * 2, bar_h + pad * 2), pygame.SRCALPHA)
    pygame.draw.rect(bar, (70, 210, 240), (pad, pad, bar_w, bar_h), border_radius=bar_h // 2)
    pos = (cx - bar.get_width() // 2, y - pad)
    target.blit(_soft(bar, 5), pos)
    target.blit(bar, pos)

_bg_layer = None

def _gradient_layer():
    global _bg_layer
    if _bg_layer is None:
        layer = pygame.Surface((WIDTH, HEIGHT)).convert()
        strip = pygame.Surface((1, HEIGHT))
        for y in range(HEIGHT):
            strip.set_at((0, y), _mix((24, 30, 48), (7, 9, 15), y / max(1, HEIGHT - 1)))
        layer.blit(pygame.transform.smoothscale(strip, (WIDTH, HEIGHT)), (0, 0))
        rnd = random.Random(20261013)
        for _ in range(18):
            band = pygame.Surface((rnd.randint(int(WIDTH * 0.10), int(WIDTH * 0.30)),
                                   rnd.choice((1, 1, 2))), pygame.SRCALPHA)
            band.fill((90, 190, 255, rnd.randint(6, 14)))
            layer.blit(band, (rnd.randint(0, WIDTH), rnd.randint(int(HEIGHT * 0.80), int(HEIGHT * 0.99))))
        _bg_layer = layer
    return _bg_layer

def paint_panel(layer, rect, radius=None):
    r = int(24 * SCALE) if radius is None else radius
    halo_rect = rect.inflate(int(90 * SCALE), int(90 * SCALE))
    halo = pygame.Surface(halo_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(halo, (70, 190, 240, 120), halo.get_rect().inflate(-int(90 * SCALE), -int(90 * SCALE)),
                     width=int(6 * SCALE), border_radius=r)
    layer.blit(_soft(halo, 10), halo_rect.topleft)
    shadow = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(shadow, (0, 0, 0, 125), shadow.get_rect(), border_radius=r)
    layer.blit(_soft(shadow, 16), rect.move(0, int(12 * SCALE)))
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (17, 23, 37, 248), panel.get_rect(), border_radius=r)
    layer.blit(panel, rect.topleft)
    pygame.draw.rect(layer, (46, 66, 100), rect, max(1, int(2 * SCALE)), border_radius=r)

# ============================================================
# MENU LAYOUT & CAR SELECTION
# ============================================================
SEL_PAD = int(44 * SCALE)
CARD_COLS = 3
CARD_W = int(330 * SCALE)
CARD_H = int(62 * SCALE)
CARD_GAP_X = int(22 * SCALE)
CARD_GAP_Y = int(10 * SCALE)
SEL_W = CARD_COLS * CARD_W + (CARD_COLS - 1) * CARD_GAP_X
SEL_H = int(620 * SCALE)
SEL_RECT = pygame.Rect(0, 0, SEL_W + SEL_PAD * 2, SEL_H)
SEL_RECT.center = (WIDTH // 2, HEIGHT // 2)
SEL_X = SEL_RECT.x + SEL_PAD
SEL_Y = SEL_RECT.y

CARD_Y = SEL_Y + int(140 * SCALE)
TRACK_LABEL_Y = SEL_Y + int(296 * SCALE)
TRACK_Y = SEL_Y + int(312 * SCALE)
ACTION_Y = SEL_Y + int(390 * SCALE)
START_Y = SEL_Y + int(470 * SCALE)
INFO_TEXT_Y = SEL_Y + int(545 * SCALE)
TIP_Y = SEL_Y + int(575 * SCALE)

def get_card_rect(idx):
    col, row = idx % CARD_COLS, idx // CARD_COLS
    return pygame.Rect(SEL_X + col * (CARD_W + CARD_GAP_X), CARD_Y + row * (CARD_H + CARD_GAP_Y), CARD_W, CARD_H)

TRACK_BTN_W = int(170 * SCALE)
TRACK_BTN_H = int(54 * SCALE)
TRACK_BTN_GAP = int(22 * SCALE)
TRACK_ROW_W = 3 * TRACK_BTN_W + 2 * TRACK_BTN_GAP
TRACK_X0 = SEL_X + (SEL_W - TRACK_ROW_W) // 2

def get_track_btn_rect(idx):
    return pygame.Rect(TRACK_X0 + idx * (TRACK_BTN_W + TRACK_BTN_GAP), TRACK_Y, TRACK_BTN_W, TRACK_BTN_H)

UPLOAD_BTN_RECT = pygame.Rect(SEL_X, ACTION_Y, SEL_W, int(60 * SCALE))
START_BTN_RECT = pygame.Rect(SEL_X, START_Y, SEL_W, int(60 * SCALE))

def button(rect, fill, border, radius=8, shadow=3, border_w=2):
    pygame.draw.rect(screen, (0, 0, 0), rect.move(0, shadow), border_radius=radius)
    pygame.draw.rect(screen, fill, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, border_w, border_radius=radius)

def hit_index(rects, pos):
    for i, r in enumerate(rects):
        if r.collidepoint(pos):
            return i
    return None

_select_layer = None

def _select_backdrop():
    global _select_layer
    if _select_layer is not None:
        return _select_layer
    layer = _gradient_layer().copy()
    paint_panel(layer, SEL_RECT)

    title, tpad = _glow_text("FACE RACER", title_font, WHITE, (44, 172, 220), blur=12, spread=10, passes=1)
    ty = SEL_Y + int(26 * SCALE)
    layer.blit(title, (SEL_RECT.centerx - title.get_width() // 2, ty - tpad))
    _glow_bar(layer, SEL_RECT.centerx, ty + title.get_height() - tpad * 2 + int(10 * SCALE))

    blit_cx(small_font.render("SELECT TRACK", True, (150, 172, 200)), SEL_RECT.centerx, TRACK_LABEL_Y, layer)
    blit_cx(tiny_font.render("MIDI 58 = Upload photo  |  F = fullscreen  |  ESC = quit", True, (120, 136, 160)),
            SEL_RECT.centerx, TIP_Y, layer)
    _select_layer = layer
    return layer

def draw_select_screen(mouse_pos):
    screen.blit(_select_backdrop(), (0, 0))

    blit_cx(small_font.render(f"Player: {username}   |   < > Car   |   1-3 Track   |   ENTER Start",
                              True, (150, 172, 200)), SEL_RECT.centerx, SEL_Y + int(108 * SCALE))

    for i, car in enumerate(CARS):
        cr = get_card_rect(i)
        selected = (i == selected_car_idx)
        button(cr, CARD_SEL if selected else (CARD_HOVER if cr.collidepoint(mouse_pos) else CARD_BG),
               CYAN if selected else (52, 68, 96),
               radius=int(12 * SCALE), shadow=int(3 * SCALE), border_w=int(2 * SCALE))
        sw = int(36 * SCALE)
        px, py = cr.x + int(12 * SCALE), cr.y + (cr.h - int(46 * SCALE)) // 2
        ph = int(46 * SCALE)
        pygame.draw.rect(screen, car["color"], (px, py, sw, ph), border_radius=int(6 * SCALE))
        pygame.draw.rect(screen, car["accent"],
                         (px + sw // 2 - int(2 * SCALE), py, int(4 * SCALE), ph), border_radius=int(2 * SCALE))
        if player_face is not None:
            fw, fh = player_face.get_size()
            k = min((sw - int(6 * SCALE)) / fw, (ph - int(6 * SCALE)) / fh)
            face = pygame.transform.smoothscale(player_face, (max(1, int(fw * k)), max(1, int(fh * k))))
            screen.blit(face, (px + sw // 2 - face.get_width() // 2, py + ph // 2 - face.get_height() // 2))
        tx = cr.x + int(58 * SCALE)
        screen.blit(font.render(car["name"], True, WHITE), (tx, cr.y + int(4 * SCALE)))
        screen.blit(small_font.render("SPD", True, (150, 172, 200)), (tx, cr.y + int(27 * SCALE)))
        draw_stars(tx + int(38 * SCALE), cr.y + int(27 * SCALE), car["stars_speed"])
        screen.blit(small_font.render("HND", True, (150, 172, 200)), (tx, cr.y + int(45 * SCALE)))
        draw_stars(tx + int(38 * SCALE), cr.y + int(45 * SCALE), car["stars_handling"])

    for i, track in enumerate(TRACKS):
        r = get_track_btn_rect(i)
        sel, hov = (i == selected_track), r.collidepoint(mouse_pos)
        button(r, TRACK_SEL if sel else (TRACK_HOVER if hov else TRACK_NORMAL),
               CYAN if sel else ((120, 148, 184) if hov else (52, 68, 96)),
               radius=int(12 * SCALE), shadow=int(3 * SCALE), border_w=int(2 * SCALE))
        blit_cx(font.render(track["name"], True, WHITE), r.centerx, r.y + int(7 * SCALE))
        draw_stars(0, r.y + int(30 * SCALE), track["stars"], center=r.centerx)

    face_ready = player_face is not None
    button(UPLOAD_BTN_RECT,
           (74, 130, 84) if face_ready else (
               UPLOAD_HOVER if UPLOAD_BTN_RECT.collidepoint(mouse_pos) else UPLOAD_NORMAL),
           CYAN if face_ready else WHITE, radius=int(14 * SCALE), shadow=int(4 * SCALE), border_w=int(2 * SCALE))
    blit_center(font.render("Face Ready (Change)" if face_ready else "Upload Photo (Face)", True, WHITE),
                *UPLOAD_BTN_RECT.center)

    button(START_BTN_RECT, (96, 210, 122) if START_BTN_RECT.collidepoint(mouse_pos) else (52, 160, 78),
           WHITE, radius=int(14 * SCALE), shadow=int(4 * SCALE), border_w=int(2 * SCALE))
    blit_center(btn_font.render("START GAME", True, WHITE), *START_BTN_RECT.center)

    car = CARS[selected_car_idx]
    blit_cx(small_font.render(f"{car['name']}  ·  {TRACKS[selected_track]['name']}",
                              True, (255, 220, 100)), SEL_RECT.centerx, INFO_TEXT_Y)

# ============================================================
# USER LOGIN SCREEN
# ============================================================
USER_CARD_W = int(800 * SCALE)
USER_CARD_H = int(320 * SCALE)
_user_layer = None

def user_card_rect():
    return pygame.Rect(WIDTH // 2 - USER_CARD_W // 2, HEIGHT // 2 - USER_CARD_H // 2, USER_CARD_W, USER_CARD_H)

def _user_backdrop():
    global _user_layer
    if _user_layer is not None:
        return _user_layer
    layer = _gradient_layer().copy()
    card = user_card_rect()
    paint_panel(layer, card)

    title, tpad = _glow_text("FACE CAR PRO", title_font, WHITE, (44, 172, 220), blur=12, spread=10, passes=1)
    ty = card.y + int(30 * SCALE)
    layer.blit(title, (card.centerx - title.get_width() // 2, ty - tpad))
    _glow_bar(layer, card.centerx, ty + title.get_height() - tpad * 2 + int(10 * SCALE))

    _user_layer = layer
    return layer

def draw_username_screen():
    screen.blit(_user_backdrop(), (0, 0))
    card = user_card_rect()
    pad = int(48 * SCALE)
    content_y = card.y + int(106 * SCALE)

    slot = pygame.Rect(card.x + pad, content_y + int(27 * SCALE), int(120 * SCALE), int(150 * SCALE))
    screen.blit(small_font.render("Your photo", True, (150, 172, 200)), (slot.x, content_y))

    pygame.draw.rect(screen, (24, 31, 48), slot, border_radius=int(14 * SCALE))
    if player_face:
        fw, fh = player_face.get_size()
        k = min((slot.w - int(10 * SCALE)) / fw, (slot.h - int(10 * SCALE)) / fh)
        face = pygame.transform.smoothscale(player_face, (max(1, int(fw * k)), max(1, int(fh * k))))
        screen.blit(face, (slot.centerx - face.get_width() // 2, slot.centery - face.get_height() // 2))
        pygame.draw.rect(screen, CYAN, slot, max(1, int(2 * SCALE)), border_radius=int(14 * SCALE))
    else:
        pygame.draw.rect(screen, (60, 78, 108), slot, max(1, int(2 * SCALE)), border_radius=int(14 * SCALE))
        arm, thick = int(34 * SCALE), max(2, int(4 * SCALE))
        pygame.draw.rect(screen, (76, 98, 132),
                         (slot.centerx - arm // 2, slot.centery - thick // 2, arm, thick), border_radius=thick // 2)
        pygame.draw.rect(screen, (76, 98, 132),
                         (slot.centerx - thick // 2, slot.centery - arm // 2, thick, arm), border_radius=thick // 2)
    status, status_col = ("Face detected & loaded", CYAN) if player_face else ("Press L to upload", (112, 132, 162))
    status_surf = tiny_font.render(status, True, status_col)
    blit_cx(status_surf, slot.centerx, slot.bottom + int(9 * SCALE))

    fx = slot.right + int(30 * SCALE)
    fw = card.right - pad - fx
    screen.blit(small_font.render("Enter your username:", True, (150, 172, 200)), (fx, content_y))

    box = pygame.Rect(fx, content_y + int(29 * SCALE), fw, int(78 * SCALE))
    pygame.draw.rect(screen, (26, 34, 52), box, border_radius=int(14 * SCALE))
    pygame.draw.rect(screen, CYAN if input_active else GRAY, box, max(1, int(3 * SCALE)), border_radius=int(14 * SCALE))

    typed = big_font.render(username, True, WHITE)
    text_x = box.x + int(22 * SCALE)
    screen.blit(typed, (text_x, box.centery - typed.get_height() // 2))
    if pygame.time.get_ticks() % 1000 < 500:
        caret_h = int(38 * SCALE)
        pygame.draw.rect(screen, CYAN,
                         (text_x + typed.get_width() + int(4 * SCALE), box.centery - caret_h // 2,
                          max(2, int(3 * SCALE)), caret_h), border_radius=2)

    screen.blit(small_font.render("Type name + ENTER to start   |   MIDI 58 = Upload photo", True, (120, 198, 140)),
                (fx, box.bottom + int(16 * SCALE)))

    blit_cx(tiny_font.render("F = Fullscreen   |   ESC = Quit", True, GRAY), WIDTH // 2, HEIGHT - int(46 * SCALE))

def draw_game_hud():
    screen.blit(font.render(f"SCORE  {score}", True, WHITE), (25, 25))
    screen.blit(small_font.render(f"Time  {game_time:.1f}s", True, (200, 200, 255)), (25, 60))
    screen.blit(small_font.render(f"Best Score {high_score}", True, (180, 180, 180)), (25, 90))
    if current_car:
        screen.blit(small_font.render(current_car["name"], True, current_car["color"]), (25, 120))
    if username:
        screen.blit(tiny_font.render(username, True, CYAN), (25, 150))

    lives_label = f"Lives  {'♥ ' * max(0, lives)}"
    screen.blit(small_font.render(lives_label, True, RED), (25, 175))
    if is_invulnerable():
        left = max(0, (invuln_until_ms - get_now_ms()) / 1000.0)
        screen.blit(tiny_font.render(f"INVULN {left:.1f}s", True, CYAN), (25, 198))

    speed_pct = int((obstacle_speed - MIN_SPEED) / max(1, MAX_SPEED - MIN_SPEED) * 100)
    if speed_locked:
        spd_label = f"Speed  {speed_pct}%  MAX LOCK"
        spd_col = ORANGE
    else:
        remain = max(0.0, SPEED_LOCK_AT - game_time)
        spd_label = f"Speed  {speed_pct}%  (max in {remain:.0f}s)"
        spd_col = YELLOW
    screen.blit(small_font.render(spd_label, True, spd_col), (25, 220))

    st = ("STEER ✓ " if serial_steer_active else "STEER ✗ ") + \
         ("THR ✓"   if serial_throttle_active else "THR ✗")
    col = GREEN if (serial_steer_active and serial_throttle_active) else \
          YELLOW if (serial_steer_active or serial_throttle_active) else RED
    screen.blit(tiny_font.render(st, True, col), (25, 250))

    if drift_amount > 0.4:
        screen.blit(small_font.render("DRIFT", True, ORANGE), (25, 275))

    if multiplier > 1:
        mult = font.render(f"x{multiplier} COMBO", True, YELLOW)
        screen.blit(mult, (WIDTH // 2 - mult.get_width() // 2, 25))

def draw_game_over():
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 210))
    screen.blit(overlay, (0, 0))

    blit_cx(big_font.render("CRASHED!", True, RED), WIDTH // 2, 40)

    screen.blit(font.render(f"{username}  •  {last_survival_time:.1f}s  •  Score {score}", True, WHITE),
                (WIDTH // 2 - 220, 110))

    track_name = TRACKS[selected_track]["name"]
    if player_rank is not None:
        if player_rank <= 10:
            blit_cx(font.render(f"You placed #{player_rank} on the {track_name} Leaderboard!", True, GOLD),
                    WIDTH // 2, 150)
        else:
            blit_cx(font.render(f"You placed #{player_rank} on {track_name}  (outside Top 10)", True, (255, 200, 100)),
                    WIDTH // 2, 150)
    else:
        blit_cx(small_font.render(f"Rank unavailable on {track_name}", True, (180, 180, 180)),
                WIDTH // 2, 155)

    blit_cx(font.render(f"LEADERBOARD — {track_name}  (Top 10 by Survival Time)", True, YELLOW),
            WIDTH // 2, 200)

    y_pos = 250
    top10 = leaderboard[:10] if leaderboard else []
    for i, entry in enumerate(top10):
        rank_col = GOLD if i < 3 else WHITE
        line = f"#{i + 1}  {entry['username']:<12}  {entry['time']:>6.1f}s   {entry['score']:>4} pts   {entry['car']}"
        if player_rank and i + 1 == player_rank:
            pygame.draw.rect(screen, (60, 60, 30), (WIDTH // 2 - 320, y_pos - 2, 640, 28), border_radius=6)
        text = small_font.render(line, True, rank_col)
        screen.blit(text, (WIDTH // 2 - 300, y_pos))
        y_pos += 32

    if player_rank is not None and player_rank > 10 and player_entry is not None:
        y_pos += 8
        blit_cx(tiny_font.render("· · ·", True, (140, 150, 170)), WIDTH // 2, y_pos)
        y_pos += 22
        pygame.draw.rect(screen, (50, 55, 30), (WIDTH // 2 - 320, y_pos - 2, 640, 28), border_radius=6)
        line = (f"#{player_rank}  {player_entry['username']:<12}  "
                f"{player_entry['time']:>6.1f}s   {player_entry['score']:>4} pts   {player_entry['car']}")
        text = small_font.render(line, True, (255, 220, 120))
        screen.blit(text, (WIDTH // 2 - 300, y_pos))
        y_pos += 32

    blit_cx(font.render("Press R / SPACE / ENTER → Play Again", True, GREEN), WIDTH // 2, HEIGHT - 70)

# ============================================================
# MAIN EVENT AND GAME LOOP
# ============================================================
running = True
road_offset = 0
mouse_pos = (0, 0)
play_bgm()

while running:
    dt = clock.tick(FPS) / 1000.0
    mouse_pos = pygame.mouse.get_pos()

    poll_midi()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if face_pick_open:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    close_face_picker()
                elif event.key in (pygame.K_1, pygame.K_l):
                    close_face_picker()
                    pick_face()
                elif event.key in (pygame.K_2, pygame.K_c):
                    close_face_picker()
                    shoot_face()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if FACE_PICK_UPLOAD_RECT.collidepoint(event.pos):
                    close_face_picker()
                    pick_face()
                elif FACE_PICK_CAMERA_RECT.collidepoint(event.pos):
                    close_face_picker()
                    shoot_face()
                elif not FACE_PICK_RECT.collidepoint(event.pos):
                    close_face_picker()
            continue

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_f:
                pygame.display.toggle_fullscreen()

            if state == STATE_USERNAME:
                if event.key == pygame.K_RETURN:
                    if len(username.strip()) >= 1:
                        username = username.strip()[:15]
                        state = STATE_SELECT
                elif event.key == pygame.K_BACKSPACE:
                    username = username[:-1]
                elif event.key == pygame.K_l:
                    open_face_picker()
                else:
                    if len(username) < 15 and event.unicode.isprintable():
                        username += event.unicode

            elif state == STATE_SELECT:
                if pygame.K_1 <= event.key <= pygame.K_6:
                    selected_car_idx = event.key - pygame.K_1
                if event.key == pygame.K_LEFT:
                    selected_car_idx = (selected_car_idx - 1) % 6
                if event.key == pygame.K_RIGHT:
                    selected_car_idx = (selected_car_idx + 1) % 6
                if event.key in TRACK_KEYS:
                    selected_track = TRACK_KEYS[event.key]
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    current_car = CARS[selected_car_idx]
                    start_game()
                if event.key == pygame.K_l:
                    open_face_picker()

            elif state == STATE_PLAY:
                if event.key == pygame.K_SPACE:
                    try_activate_nitro()
                elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                    try_activate_shield()
                elif event.key == pygame.K_e:
                    try_activate_laser()
                elif event.key == pygame.K_k:
                    death_trap_active = True

            elif state == STATE_GAMEOVER:
                if event.key in (pygame.K_r, pygame.K_SPACE, pygame.K_RETURN):
                    full_reset()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and state == STATE_SELECT:
            pos = event.pos
            idx = hit_index([get_card_rect(k) for k in range(len(CARS))], pos)
            if idx is not None:
                selected_car_idx = idx
            idx = hit_index([get_track_btn_rect(k) for k in range(len(TRACKS))], pos)
            if idx is not None:
                selected_track = idx
            if UPLOAD_BTN_RECT.collidepoint(pos):
                open_face_picker()
            if START_BTN_RECT.collidepoint(pos):
                current_car = CARS[selected_car_idx]
                start_game()

    # ============================================================
    # PLAY STATE GAME PHYSICS AND CONTROL UPDATES
    # ============================================================
    if state == STATE_PLAY:
        game_time += dt

        current_spawn_delay = max(15.0, base_spawn_delay - game_time * 1.9)

        if death_trap_active:
            current_spawn_delay = 45.0

        spawn_timer += 1
        if spawn_timer >= current_spawn_delay:
            spawn_timer = 0
            spawn_obstacle()

        ped_timer += 1
        if ped_timer >= 200 and not death_trap_active:
            ped_timer = 0
            spawn_pedestrian()
        coin_timer += 1
        if coin_timer >= 90 and not death_trap_active:
            coin_timer = 0
            spawn_coin()

        if not speed_locked and game_time >= SPEED_LOCK_AT:
            speed_locked = True
            obstacle_speed = MAX_SPEED
            throttle_speed = THROTTLE_MAX

        keys = pygame.key.get_pressed()
        if speed_locked:
            obstacle_speed = MAX_SPEED
            throttle_speed = THROTTLE_MAX
        else:
            # 1. Keyboard Speed Fallback Controls
            if keys[pygame.K_UP]:
                obstacle_speed = min(MAX_SPEED, obstacle_speed + SPEED_CHANGE_RATE * dt)
            if keys[pygame.K_DOWN]:
                obstacle_speed = max(MIN_SPEED, obstacle_speed - SPEED_CHANGE_RATE * dt)
            if throttle_held:
                throttle_speed = min(throttle_speed + THROTTLE_ACCEL, THROTTLE_MAX)
            else:
                throttle_speed = max(throttle_speed - THROTTLE_DECAY, 0.0)

            # 2. Serial Throttle & Road Friction Deceleration Mechanics
            if serial_throttle_active:
                if serial_throttle > 0.05:
                    # Depressed Pedal -> Accelerate towards target speed
                    target = MIN_SPEED + serial_throttle * (MAX_SPEED - MIN_SPEED)
                    if obstacle_speed < target:
                        obstacle_speed = min(target, obstacle_speed + SPEED_CHANGE_RATE * 2.5 * dt)
                    else:
                        obstacle_speed = max(target, obstacle_speed - SPEED_CHANGE_RATE * dt)
                    throttle_speed = max(throttle_speed, serial_throttle * THROTTLE_MAX)
                else:
                    # Released Pedal -> Road Friction Deceleration
                    obstacle_speed = max(MIN_SPEED, obstacle_speed - (SPEED_CHANGE_RATE * 0.8) * dt)
                    throttle_speed = max(0.0, throttle_speed - THROTTLE_DECAY)

        # 3. Steering Inputs (Keyboard Fallback + PKS Serial Wheel)
        target_steer = 0.0
        if keys[pygame.K_LEFT]:
            target_steer = -1.0
        if keys[pygame.K_RIGHT]:
            target_steer = 1.0
        if serial_steer_active:
            target_steer = serial_steer

        steer_input += (target_steer - steer_input) * 18.0 * dt
        moving = 0 if abs(target_steer) <= 0.2 else (1 if target_steer > 0 else -1)
        play_turn_sound(moving)

        target_ang = -moving * DRIFT_MAX_ANGLE if (drift_held and moving) else 0.0
        if abs(drift_angle - target_ang) < DRIFT_ANGLE_STEP:
            drift_angle = target_ang
        elif drift_angle < target_ang:
            drift_angle += DRIFT_ANGLE_STEP
        else:
            drift_angle -= DRIFT_ANGLE_STEP

        speed_factor = min(1.0, 6.5 * SCALE / max(0.1, obstacle_speed))
        effective_steer = steer_input * current_car["max_steer"] * (0.75 + 0.25 * speed_factor)
        desired_vx = effective_steer * 12.0 * SCALE * current_car["accel"]
        car_vx += (desired_vx - car_vx) * current_car["grip"] * GRIP_BOOST * 12.0 * dt

        drift_amount = abs(steer_input) * (obstacle_speed / (12 * SCALE)) * (1.1 - current_car["grip"] * 0.5)
        drift_amount = max(0.0, min(1.3, drift_amount))

        car_x += car_vx
        if drift_held and moving != 0:
            car_vx *= 1.0 + (DRIFT_SPEED_MULT - 1.0) * 0.3

        edge_pad = max(4, int(6 * SCALE))
        left_limit = ROAD_LEFT + edge_pad
        right_limit = ROAD_RIGHT - car_w - edge_pad
        if car_x < left_limit:
            car_x = left_limit
            car_vx *= -0.3
        if car_x > right_limit:
            car_x = right_limit
            car_vx *= -0.3

        if moving != 0 or nitro_active or throttle_speed > 0.5 or drift_held:
            start_engine_sound()
            spawn_smoke(moving, car_x, car_y, car_w, car_h)
        else:
            stop_engine_sound()
        update_smoke()

        if drift_amount > 0.45 and random.random() < 0.55:
            skid_particles.append({
                "x": car_x + car_w // 2 + random.randint(-15, 15),
                "y": car_y + car_h - 10,
                "life": random.randint(12, 25),
                "size": random.randint(3, 7)
            })

        enemy_speed_mult = 1.0 + (throttle_speed / THROTTLE_MAX) * THROTTLE_ENEMY_BOOST
        effective_obstacle_speed = obstacle_speed * enemy_speed_mult

        for obs in obstacles[:]:
            obs["y"] += effective_obstacle_speed
            if not obs.get("passed") and obs["y"] > car_y - 50:
                dist = abs((obs["x"] + obs["w"] / 2) - (car_x + car_w / 2))
                if car_w * 0.65 < dist < car_w * 1.8:
                    if near_miss_timer <= 0:
                        near_miss_timer = 16
                        score += 2 * multiplier
                        shake = 6
            if obs["y"] > HEIGHT:
                if not obs.get("passed"):
                    obs["passed"] = True
                    score += multiplier
                    combo += 1
                    if combo >= 5:
                        multiplier = min(5, 1 + combo // 5)
                    obstacle_speed = min(obstacle_speed + 0.06 * SCALE, MAX_SPEED)
                obstacles.remove(obs)

        for p in pedestrians[:]:
            p[1] += effective_obstacle_speed * 0.6
            p[0] += p[2] * 1.4 * SCALE
            if p[0] < ROAD_LEFT + 10:
                p[0] = ROAD_LEFT + 10
                p[2] = 1
            if p[0] > ROAD_RIGHT - int(26 * SCALE) - 10:
                p[0] = ROAD_RIGHT - int(26 * SCALE) - 10
                p[2] = -1
            if p[1] > HEIGHT:
                pedestrians.remove(p)
        for c in coins[:]:
            c[1] += effective_obstacle_speed * 0.9
            if c[1] > HEIGHT:
                coins.remove(c)

        pr = (car_x + 12, car_y + 18, car_w - 24, car_h - 30)
        invuln = is_invulnerable()

        for obs in obstacles[:]:
            obs_rect = (obs["x"] + 8, obs["y"] + 8, obs["w"] - 16, obs["h"] - 16)
            if rect_collide(pr, obs_rect):
                if shield_active:
                    obstacles.remove(obs)
                    score += 2
                    shake = 8
                    create_particles(obs["x"], obs["y"], 15)
                elif invuln:
                    pass
                else:
                    lose_life(reset_combo=True)
                    break

        if state == STATE_PLAY:
            for c in coins[:]:
                if rect_collide(pr, (c[0], c[1], int(24 * SCALE), int(24 * SCALE))):
                    coins.remove(c)
                    score += 5

            for p in pedestrians[:]:
                if rect_collide(pr, (p[0], p[1], int(26 * SCALE), int(48 * SCALE))):
                    if shield_active or invuln:
                        continue
                    lose_life()
                    break

        if obstacle_speed > 9 * SCALE and random.random() < 0.4:
            speed_lines.append({
                "x": random.randint(ROAD_LEFT, ROAD_RIGHT),
                "y": -40, "len": random.randint(50, 110),
                "speed": obstacle_speed * 2.0
            })

        update_skills()
        road_offset += effective_obstacle_speed

    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.3
        p["life"] -= 1
        if p["life"] <= 0:
            particles.remove(p)
    for s in skid_particles[:]:
        s["life"] -= 1
        if s["life"] <= 0:
            skid_particles.remove(s)
    for line in speed_lines[:]:
        line["y"] += line["speed"]
        if line["y"] > HEIGHT:
            speed_lines.remove(line)
    if near_miss_timer > 0:
        near_miss_timer -= 1
    if shake > 0:
        shake -= 1
    update_smoke()

    # ============================================================
    # SCREEN RENDER DISPATCH
    # ============================================================
    if state == STATE_USERNAME:
        draw_username_screen()

    elif state == STATE_SELECT:
        draw_select_screen(mouse_pos)

    elif state in (STATE_PLAY, STATE_GAMEOVER):
        shake_x = random.randint(-shake, shake) if shake else 0
        shake_y = random.randint(-shake, shake) if shake else 0

        draw_road(road_offset, shake_x, shake_y)

        for s in skid_particles:
            alpha = min(160, s["life"] * 8)
            surf = pygame.Surface((s["size"] * 2, s["size"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (30, 30, 30, alpha), (s["size"], s["size"]), s["size"])
            screen.blit(surf, (s["x"] + shake_x - s["size"], s["y"] + shake_y - s["size"]))

        for line in speed_lines:
            pygame.draw.line(screen, (190, 190, 220),
                             (line["x"] + shake_x, line["y"] + shake_y),
                             (line["x"] + shake_x, line["y"] + line["len"] + shake_y), 2)

        for obs in obstacles:
            if obs["kind"] == "car":
                draw_enemy_car(obs["x"] + shake_x, obs["y"] + shake_y, obs["w"], obs["h"], obs["color"])
            else:
                draw_barrier(screen, obs["x"] + shake_x, obs["y"] + shake_y, obs["w"], obs["h"])

        for c in coins:
            draw_coin(c[0] + shake_x, c[1] + shake_y, int(24 * SCALE))
        for p in pedestrians:
            draw_pedestrian(p[0] + shake_x, p[1] + shake_y, int(26 * SCALE), int(48 * SCALE))

        draw_smoke()
        draw_laser()

        show_player = state == STATE_PLAY or flash > 0
        if show_player and is_invulnerable() and (get_now_ms() // 100) % 2 == 0:
            show_player = False
        if show_player:
            draw_car_rotated(car_x + shake_x, car_y + shake_y, current_car,
                             car_w, car_h, player_face, True, drift_angle)

        for p in particles:
            alpha = min(255, p["life"] * 7)
            s = pygame.Surface((12, 12), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p["color"], alpha), (6, 6), 6)
            screen.blit(s, (p["x"] + shake_x, p["y"] + shake_y))

        if flash > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 15, 15, min(200, flash * 9)))
            screen.blit(overlay, (0, 0))
            flash -= 1

        draw_game_hud()
        draw_skill_hud()

        if state == STATE_GAMEOVER:
            draw_game_over()

    if face_pick_open:
        draw_face_picker(mouse_pos)

    # Render Bottom Right UI Serial Console Window
    draw_serial_log_ui()

    pygame.display.flip()

# RESOURCE CLEANUP
stop_all_sounds()
pygame.mixer.music.stop()
pygame.mixer.quit()
if MIDI_INPUT is not None:
    try:
        MIDI_INPUT.close()
    except:
        pass
try:
    pygame.midi.quit()
except:
    pass
if ser_conn is not None:
    try:
        ser_conn.close()
    except:
        pass
pygame.quit()
sys.exit()