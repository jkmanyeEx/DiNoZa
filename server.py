#!/usr/bin/env python3
#
import socket
import threading
import json
import struct
import time

import cv2
import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

from motor_control import MotorControl


# ============================================================
#                PERSON DETECTOR (MobileNet SSD)
# ============================================================
PROTOTXT = "./model/MobileNetSSD_deploy.prototxt"
MODEL = "./model/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
PERSON_CLASS_ID = 15


# ============================================================
#                SHARED STATE
# ============================================================
state = {
    "mode": "manual",      # "manual" / "auto"
    "command": "none",     # last manual command
    "auto_lr": "none",     # "left"/"right"/"center"/"none"
    "auto_ud": "none",     # "up"/"down"/"center"/"none"
    "persons": [],         # [[x1,y1,x2,y2], ...]
    "selected": -1,        # index of target
    "shooter": False,      # True if MOSFET DC ON
}
state_lock = threading.Lock()


def to_python(obj):
    """Recursively convert numpy → pure Python for JSON."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj


# ============================================================
#                MOTOR CONTROL
# ============================================================
# TODO: change pins to your actual wiring
motors = MotorControl(
    pinmap={
        "stepper1": [5, 6, 13, 19],      # L298N IN1..IN4 (pan)
        "stepper2": [12, 16, 20, 21],    # L298N IN1..IN4 (tilt)
        "dc": {"pin": 23},               # MOSFET gate pin
    }
)


# ============================================================
#                VIDEO STREAM THREAD (H.264)
# ============================================================
def video_thread(picam2: Picamera2):
    HOST = "0.0.0.0"
    PORT = 8000

    encoder = H264Encoder(bitrate=4_000_000)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    print("[VIDEO] Waiting for client...")
    conn, addr = srv.accept()
    print("[VIDEO] Client connected:", addr)

    sock_file = conn.makefile("wb")
    output = FileOutput(sock_file)

    try:
        picam2.start_encoder(encoder, output)
        print("[VIDEO] H.264 encoder started")

        while True:
            time.sleep(1)  # encoder runs in background

    except Exception as e:
        print("[VIDEO ERROR]", e)

    finally:
        try:
            picam2.stop_encoder()
        except Exception:
            pass
        sock_file.close()
        conn.close()
        print("[VIDEO] Closed")


# ============================================================
#                CONTROL THREAD (client → Pi)
# ============================================================
def control_thread():
    HOST = "0.0.0.0"
    PORT = 8001

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    print("[CONTROL] Waiting...")
    conn, addr = srv.accept()
    print("[CONTROL] Client connected:", addr)

    buffer = ""

    while True:
        data = conn.recv(32)
        if not data:
            print("[CONTROL] disconnected")
            break

        buffer += data.decode(errors="ignore")

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            cmd = line.strip()
            if not cmd:
                continue

            with state_lock:
                if cmd == "auto":
                    state["mode"] = "auto"
                    print("[CONTROL] mode -> auto")
                elif cmd == "manual":
                    state["mode"] = "manual"
                    print("[CONTROL] mode -> manual")
                elif cmd in ["a", "d", "w", "s", "f"]:
                    state["command"] = cmd
                    print("[CONTROL] cmd:", cmd)

                mode = state["mode"]
                command = state["command"]

            # act on motors (manual only)
            if mode == "manual" and command in ["a", "d", "w", "s", "f"]:
                motors.manual_control(command)


# ============================================================
#                METADATA THREAD (Pi → client)
# ============================================================
def metadata_thread():
    HOST = "0.0.0.0"
    PORT = 8002

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    print("[META] Waiting...")
    conn, addr = srv.accept()
    print("[META] Client connected:", addr)

    while True:
        with state_lock:
            # sync shooter state from motor controller
            state["shooter"] = bool(motors.dc.state)
            payload = to_python(state)

        try:
            meta = json.dumps(payload).encode()
        except Exception as e:
            print("[META] JSON error:", e)
            time.sleep(0.1)
            continue

        try:
            conn.sendall(struct.pack(">I", len(meta)) + meta)
        except Exception as e:
            print("[META] disconnected:", e)
            break

        time.sleep(0.03)  # ~33 Hz


# ============================================================
#                DETECTION THREAD (shared camera)
# ============================================================
def detection_thread(picam2: Picamera2):
    while True:
        frame = picam2.capture_array("main")  # RGB888

        if frame is None or frame.size == 0:
            continue

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # RGB -> BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w = frame.shape[:2]

        # Adaptive brightness normalization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()
        if mean_val < 60:
            frame = cv2.convertScaleAbs(frame, alpha=1.8, beta=40)
        elif mean_val < 90:
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20)

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )

        net.setInput(blob)
        detections = net.forward()

        persons = []

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            cls = int(detections[0, 0, i, 1])

            if cls == PERSON_CLASS_ID and conf > 0.15:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = [int(v) for v in box]

                # expand box slightly
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)

                persons.append([x1, y1, x2, y2])

        with state_lock:
            state["persons"] = persons

            if not persons:
                state["selected"] = -1
                state["auto_lr"] = "none"
                state["auto_ud"] = "none"
            else:
                cx, cy = w // 2, h // 2

                # pick nearest to horizontal center
                dlist = []
                for idx, (x1, y1, x2, y2) in enumerate(persons):
                    px = (x1 + x2) // 2
                    dlist.append((abs(px - cx), idx))

                _, sel = min(dlist)
                sel = int(sel)
                state["selected"] = sel

                x1, y1, x2, y2 = persons[sel]
                px = (x1 + x2) // 2
                py = (y1 + y2) // 2

                # LR control
                if px < cx - 64:
                    state["auto_lr"] = "left"
                elif px > cx + 64:
                    state["auto_lr"] = "right"
                else:
                    state["auto_lr"] = "center"

                # UD control
                if py < cy - 48:
                    state["auto_ud"] = "up"
                elif py > cy + 48:
                    state["auto_ud"] = "down"
                else:
                    state["auto_ud"] = "center"


# ============================================================
#                AUTO CONTROL LOOP
# ============================================================
def auto_loop():
    while True:
        time.sleep(0.05)  # 20 Hz loop

        with state_lock:
            if state["mode"] != "auto":
                continue
            lr = state["auto_lr"]
            ud = state["auto_ud"]

        motors.auto_control(lr, ud)


# ============================================================
#                MAIN
# ============================================================
if __name__ == "__main__":
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
    )
    picam2.configure(config)
    picam2.start()

    threading.Thread(target=video_thread, args=(picam2,), daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()
    threading.Thread(target=metadata_thread, daemon=True).start()
    threading.Thread(target=detection_thread, args=(picam2,), daemon=True).start()
    threading.Thread(target=auto_loop, daemon=True).start()

    print("[SERVER] Online (video + control + meta + detect + auto)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
        motors.cleanup()
        picam2.stop()
