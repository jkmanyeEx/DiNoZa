#!/usr/bin/env python3
import socket
import threading
import json
import struct
import time

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

import cv2
import numpy as np

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
    "mode": "manual",
    "command": "none",
    "auto_lr": "none",
    "auto_ud": "none",
    "persons": [],
    "selected": -1,
}
state_lock = threading.Lock()


def to_python(obj):
    """Recursively convert numpy types → pure Python for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_python(v) for v in obj.items()}
    return obj


# ============================================================
#                MOTOR MANAGER (toggle shooting)
# ============================================================
class MotorManager:
    motorController = MotorControl(
        pinmap={
            "stepper1": {"dir": 5, "step": 6},
            "stepper2": {"dir": 13, "step": 19},
            "dc": {"pin": 20},
        },
        step_count=200,
    )

    PAN_STEP = 10
    TILT_STEP = 10
    SPEED = 0.001

    shooting = False  # manual toggle

    def shoot_on(self):
        print("[MOTOR] SHOOT → ON")
        self.motorController.dc_on()

    def shoot_off(self):
        print("[MOTOR] SHOOT → OFF")
        self.motorController.dc_off()

    def manual_control(self, key: str):
        # toggle shooting
        if key == "f":
            self.shooting = not self.shooting
            if self.shooting:
                self.shoot_on()
            else:
                self.shoot_off()
            return

        if key == "a":
            print("[MOTOR] PAN LEFT")
            self.motorController.rotate_stepper1(-self.PAN_STEP, self.SPEED)
        elif key == "d":
            print("[MOTOR] PAN RIGHT")
            self.motorController.rotate_stepper1(self.PAN_STEP, self.SPEED)
        elif key == "w":
            print("[MOTOR] TILT UP")
            self.motorController.rotate_stepper2(self.TILT_STEP, self.SPEED)
        elif key == "s":
            print("[MOTOR] TILT DOWN")
            self.motorController.rotate_stepper2(-self.TILT_STEP, self.SPEED)

        # if toggle on, keep DC high
        if self.shooting:
            self.shoot_on()

    def auto_control(self, lr: str, ud: str):
        # horizontal
        if lr == "left":
            self.motorController.rotate_stepper1(-self.PAN_STEP, self.SPEED)
        elif lr == "right":
            self.motorController.rotate_stepper1(self.PAN_STEP, self.SPEED)

        # vertical
        if ud == "up":
            self.motorController.rotate_stepper2(self.TILT_STEP, self.SPEED)
        elif ud == "down":
            self.motorController.rotate_stepper2(-self.TILT_STEP, self.SPEED)

        # auto fire only when centered
        if lr == "center" and ud == "center":
            self.shoot_on()
        else:
            # if user hasn't forced shooting ON manually, allow auto to stop it
            if not self.shooting:
                self.shoot_off()


motors = MotorManager()


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
            time.sleep(1)

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
#                (line-based protocol: "a\n", "auto\n")
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
                    print("[CONTROL] mode → auto")
                elif cmd == "manual":
                    state["mode"] = "manual"
                    print("[CONTROL] mode → manual")
                elif cmd in ["a", "d", "w", "s", "f"]:
                    state["command"] = cmd
                    print("[CONTROL] cmd:", cmd)

                mode = state["mode"]
                command = state["command"]
                lr = state["auto_lr"]
                ud = state["auto_ud"]

            if mode == "manual":
                motors.manual_control(command)
            else:
                motors.auto_control(lr, ud)


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
            payload = to_python(state)

        try:
            meta = json.dumps(payload).encode()
        except Exception as e:
            print("[META] JSON error:", e)
            continue

        try:
            conn.sendall(struct.pack(">I", len(meta)) + meta)
        except Exception as e:
            print("[META] disconnected:", e)
            break

        time.sleep(0.03)


# ============================================================
#                DETECTION THREAD (shared camera)
# ============================================================
def detection_thread(picam2: Picamera2):
    while True:
        frame = picam2.capture_array("main")  # RGB888 from config
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # optional: slight brightness boost to help SSD
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5,
        )
        net.setInput(blob)
        detections = net.forward()

        persons = []

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            cls = int(detections[0, 0, i, 1])
            if cls == PERSON_CLASS_ID and conf > 0.45:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = [int(v) for v in box]
                persons.append([x1, y1, x2, y2])

        with state_lock:
            state["persons"] = persons

            if not persons:
                state["selected"] = -1
                state["auto_lr"] = "none"
                state["auto_ud"] = "none"
                continue

            cx, cy = w // 2, h // 2

            # select closest to center in X
            d_list = []
            for idx, (x1, y1, x2, y2) in enumerate(persons):
                px = (x1 + x2) // 2
                d_list.append((abs(px - cx), idx))

            _, sel_idx = min(d_list)
            sel_idx = int(sel_idx)
            state["selected"] = sel_idx

            x1, y1, x2, y2 = persons[sel_idx]
            px = int((x1 + x2) // 2)
            py = int((y1 + y2) // 2)

            # auto LR
            if px < cx - 40:
                state["auto_lr"] = "left"
            elif px > cx + 40:
                state["auto_lr"] = "right"
            else:
                state["auto_lr"] = "center"

            # auto UD
            if py < cy - 40:
                state["auto_ud"] = "up"
            elif py > cy + 40:
                state["auto_ud"] = "down"
            else:
                state["auto_ud"] = "center"


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

    print("[SERVER] Online")
    while True:
        time.sleep(1)
