#!/usr/bin/env python3
import socket
import threading
import struct
import json
import time

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

import cv2
import numpy as np

from motor_control import MotorControl


# ------------------------------------------------------------
# Load motor config
# ------------------------------------------------------------
with open("config.json", "r") as f:
    motor_cfg = json.load(f)

motors = MotorControl(motor_cfg)


# ------------------------------------------------------------
# Shared state
# ------------------------------------------------------------
state = {
    "mode": "manual",
    "command": "none",
    "shoot": False,
    "auto_lr": "none",
    "auto_ud": "none",
    "persons": [],
    "selected": -1
}
lock = threading.Lock()

PERSON_CLASS = 15

net = cv2.dnn.readNetFromCaffe(
    "./model/MobileNetSSD_deploy.prototxt",
    "./model/MobileNetSSD_deploy.caffemodel"
)


# ------------------------------------------------------------
# Safe JSON convertor
# ------------------------------------------------------------
def safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe(v) for k, v in obj.items()}
    return obj


# ------------------------------------------------------------
# Motor logic
# ------------------------------------------------------------
class MotorManager:
    def __init__(self):
        self.PAN_STEP = motor_cfg["step_size"]
        self.TILT_STEP = motor_cfg["step_size"]
        self.SPEED = motor_cfg["speed"]
        self.shooting = False

    def shoot_toggle(self):
        self.shooting = not self.shooting
        if self.shooting:
            motors.dc_on()
        else:
            motors.dc_off()

    def manual(self, key):
        if key == "a":
            motors.stepper1(-self.PAN_STEP)
        elif key == "d":
            motors.stepper1(self.PAN_STEP)
        elif key == "w":
            motors.stepper2(self.TILT_STEP)
        elif key == "s":
            motors.stepper2(-self.TILT_STEP)
        elif key == "f":
            self.shoot_toggle()

    def auto(self, lr, ud):
        if lr == "left":
            motors.stepper1(-self.PAN_STEP)
        elif lr == "right":
            motors.stepper1(self.PAN_STEP)

        if ud == "up":
            motors.stepper2(self.TILT_STEP)
        elif ud == "down":
            motors.stepper2(-self.TILT_STEP)

        if lr == "center" and ud == "center":
            motors.dc_on()
        else:
            motors.dc_off()


motman = MotorManager()


# ------------------------------------------------------------
# VIDEO THREAD
# ------------------------------------------------------------
def video_thread():
    while True:
        try:
            server = socket.socket()
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", 8000))
            server.listen(1)
            print("[VIDEO] Waiting for client...")

            conn, addr = server.accept()
            print("[VIDEO] Client connected:", addr)

            picam = Picamera2()
            config = picam.create_video_configuration(main={"size": (640, 480)})
            picam.configure(config)

            encoder = H264Encoder()
            output = FileOutput(conn)

            picam.start_recording(encoder, output)
            print("[VIDEO] H.264 streaming")

            while True:
                time.sleep(0.05)

        except Exception as e:
            print("[VIDEO ERROR]", e)

        finally:
            try:
                picam.stop_recording()
            except:
                pass
            try:
                conn.close()
            except:
                pass
            time.sleep(1)


# ------------------------------------------------------------
# CONTROL THREAD
# ------------------------------------------------------------
def control_thread():
    while True:
        try:
            srv = socket.socket()
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", 8001))
            srv.listen(1)

            print("[CONTROL] waiting for client...")
            conn, addr = srv.accept()
            print("[CONTROL] connected:", addr)

            while True:
                data = conn.recv(64)
                if not data:
                    break

                cmd = data.decode().strip()

                with lock:
                    if cmd == "manual":
                        state["mode"] = "manual"
                    elif cmd == "auto":
                        state["mode"] = "auto"
                    elif cmd == "f":
                        state["shoot"] = not state["shoot"]
                        motman.shoot_toggle()
                    elif cmd in ["a", "d", "w", "s"]:
                        state["command"] = cmd

                with lock:
                    if state["mode"] == "manual":
                        motman.manual(state["command"])
                    else:
                        motman.auto(state["auto_lr"], state["auto_ud"])

        except Exception as e:
            print("[CONTROL ERROR]", e)

        finally:
            try:
                conn.close()
            except:
                pass
            time.sleep(1)


# ------------------------------------------------------------
# METADATA THREAD
# ------------------------------------------------------------
def metadata_thread():
    while True:
        try:
            srv = socket.socket()
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", 8002))
            srv.listen(1)

            print("[META] waiting for client...")
            conn, addr = srv.accept()
            print("[META] connected:", addr)

            while True:
                with lock:
                    payload = json.dumps(safe(state)).encode()

                conn.sendall(struct.pack(">I", len(payload)) + payload)
                time.sleep(0.03)

        except Exception as e:
            print("[META ERROR]", e)

        finally:
            try:
                conn.close()
            except:
                pass
            time.sleep(1)


# ------------------------------------------------------------
# DETECTION THREAD
# ------------------------------------------------------------
def detection_thread():
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (640, 480)},
        lores={"size": (300, 300), "format": "YUV420"}
    )
    picam.configure(config)
    picam.start()

    while True:
        try:
            yuv = picam.capture_array("lores")
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            rgb = cv2.flip(rgb, 1)

            h, w = rgb.shape[:2]

            blob = cv2.dnn.blobFromImage(
                cv2.resize(rgb, (300, 300)),
                0.007843,
                (300, 300),
                127.5,
                swapRB=True
            )
            net.setInput(blob)
            det = net.forward()

            persons = []

            for i in range(det.shape[2]):
                conf = det[0, 0, i, 2]
                cls  = int(det[0, 0, i, 1])
                if cls == PERSON_CLASS and conf > 0.35:
                    box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    persons.append([x1, y1, x2, y2])

            with lock:
                state["persons"] = persons

                if len(persons) == 0:
                    state["selected"] = -1
                    state["auto_lr"] = "none"
                    state["auto_ud"] = "none"
                    continue

                cx = w // 2
                cy = h // 2
                dist = [(abs(((p[0]+p[2])//2) - cx), idx) for idx, p in enumerate(persons)]
                _, sel = min(dist)
                state["selected"] = sel

                x1, y1, x2, y2 = persons[sel]
                px = (x1 + x2) // 2
                py = (y1 + y2) // 2

                state["auto_lr"] = \
                    "left" if px < cx - 30 else "right" if px > cx + 30 else "center"

                state["auto_ud"] = \
                    "up" if py < cy - 30 else "down" if py > cy + 30 else "center"

        except Exception as e:
            print("[DETECT ERROR]", e)
            time.sleep(0.2)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=video_thread, daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()
    threading.Thread(target=metadata_thread, daemon=True).start()
    threading.Thread(target=detection_thread, daemon=True).start()

    print("[SERVER] ONLINE (H.264 + DNN + motors + metadata)")

    while True:
        time.sleep(1)
