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
#                SHARED STATE (Turret Mode)
# ============================================================
state = {
    "mode": "manual",     # manual / auto
    "command": "none",    # a,d,w,s
    "auto_lr": "none",    # left/right/center
    "auto_ud": "none",    # up/down/center
    "persons": [],        # list of [x1,y1,x2,y2]
    "selected": -1        # index of closest target
}

state_lock = threading.Lock()


# ============================================================
#                MOTOR MANAGER (stub)
# ============================================================
class MotorManager:

    def __init__(self):
        self.motorController = MotorControl(pinmap={
            "stepper1": {"dir": 5, "step": 6},     # PAN
            "stepper2": {"dir": 13, "step": 19},   # TILT
            "dc": {"pin": 20}                      # SHOOTER
        }, step_count=200)

        self.PAN_STEP = 10
        self.TILT_STEP = 10
        self.STEP_SPEED = 0.001

        self.shooting = False   # toggle state

    # ---------------------------------------------------------
    # DC CONTROL
    # ---------------------------------------------------------
    def shoot_on(self):
        print("[MOTOR] SHOOT → ON (HIGH)")
        self.motorController.dc_on()

    def shoot_off(self):
        print("[MOTOR] SHOOT → OFF (LOW)")
        self.motorController.dc_off()

    # ---------------------------------------------------------
    # MANUAL CONTROL  (TOGGLE SHOOT)
    # ---------------------------------------------------------
    def manual_control(self, key):

        # ----- TOGGLE SHOOT ON/OFF -----
        if key == "f":
            self.shooting = not self.shooting
            if self.shooting:
                self.shoot_on()
            else:
                self.shoot_off()
            return

        # ----- MOVEMENT -----
        if key == "a":
            print("[MOTOR] pan LEFT")
            self.motorController.rotate_stepper1(-self.PAN_STEP, self.STEP_SPEED)

        elif key == "d":
            print("[MOTOR] pan RIGHT")
            self.motorController.rotate_stepper1(self.PAN_STEP, self.STEP_SPEED)

        elif key == "w":
            print("[MOTOR] tilt UP")
            self.motorController.rotate_stepper2(self.TILT_STEP, self.STEP_SPEED)

        elif key == "s":
            print("[MOTOR] tilt DOWN")
            self.motorController.rotate_stepper2(-self.TILT_STEP, self.STEP_SPEED)

        # movement DOES NOT disable shooting in toggle mode
        if self.shooting:
            self.shoot_on()   # ensure it stays ON

    # ---------------------------------------------------------
    # AUTO CONTROL (unchanged)
    # ---------------------------------------------------------
    def auto_control(self, lr, ud):
        # ----- PAN -----
        if lr == "left":
            self.motorController.rotate_stepper1(-self.PAN_STEP, self.STEP_SPEED)
        elif lr == "right":
            self.motorController.rotate_stepper1(self.PAN_STEP, self.STEP_SPEED)

        # ----- TILT -----
        if ud == "up":
            self.motorController.rotate_stepper2(self.TILT_STEP, self.STEP_SPEED)
        elif ud == "down":
            self.motorController.rotate_stepper2(-self.TILT_STEP, self.STEP_SPEED)

        # ----- AUTO FIRE (only when centered) -----
        if lr == "center" and ud == "center":
            self.shoot_on()
        else:
            # only turn off if NOT manually toggled on
            if not self.shooting:
                self.shoot_off()

motors = MotorManager()

# ============================================================
#                VIDEO STREAM THREAD (H.264)
# ============================================================
def video_thread():
    HOST = "0.0.0.0"
    PORT = 8000

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    encoder = H264Encoder(bitrate=4000000)

    print("[VIDEO] Waiting for video client...")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    conn, addr = srv.accept()
    print(f"[VIDEO] Client connected:", addr)

    output = FileOutput(conn)
    picam2.start_recording(encoder, output)
    print("[VIDEO] Streaming now...")

    while True:
        time.sleep(1)


# ============================================================
#                CONTROL THREAD (Client → Pi)
# ============================================================
def control_thread():
    HOST = "0.0.0.0"
    PORT = 8001

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    print("[CONTROL] listening on 8001")
    conn, addr = srv.accept()
    print("[CONTROL] client connected:", addr)

    while True:
        msg = conn.recv(1024)
        if not msg:
            break

        try:
            data = msg.decode().strip()

            with state_lock:
                if data == "auto":
                    state["mode"] = "auto"
                elif data == "manual":
                    state["mode"] = "manual"
                elif data in ["a", "d", "w", "s"]:
                    state["command"] = data
                else:
                    print("[CONTROL] unknown:", data)

            # apply motor action
            with state_lock:
                if state["mode"] == "manual":
                    motors.manual_control(state["command"])
                else:
                    motors.auto_control(state["auto_lr"], state["auto_ud"])

        except:
            pass


# ============================================================
#                METADATA THREAD (Pi → Client)
# ============================================================
def metadata_thread():
    HOST = "0.0.0.0"
    PORT = 8002

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    print("[META] waiting on 8002")
    conn, addr = srv.accept()
    print("[META] client connected:", addr)

    while True:
        with state_lock:
            meta = json.dumps(state).encode()

        conn.sendall(struct.pack(">I", len(meta)) + meta)
        time.sleep(0.03)  # ~33 FPS metadata


# ============================================================
#                DETECTION THREAD
# ============================================================
def detection_thread():
    picam = Picamera2()
    config = picam.create_video_configuration(main={"size": (640, 480)})
    picam.configure(config)
    picam.start()

    while True:
        frame = picam.capture_array()
        (h, w) = frame.shape[:2]

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
            conf = detections[0, 0, i, 2]
            cls = int(detections[0, 0, i, 1])
            if cls == PERSON_CLASS_ID and conf > 0.45:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)
                persons.append([x1, y1, x2, y2])

        with state_lock:
            state["persons"] = persons

            if len(persons) == 0:
                state["selected"] = -1
                state["auto_lr"] = "none"
                state["auto_ud"] = "none"
                continue

            # pick person closest to horizontal center
            cx = w // 2
            cy = h // 2
            dist_list = []
            for idx, p in enumerate(persons):
                px_center = (p[0] + p[2]) // 2
                dist_list.append((abs(px_center - cx), idx))

            _, sel = min(dist_list)
            state["selected"] = sel

            x1, y1, x2, y2 = persons[sel]
            px = (x1 + x2) // 2
            py = (y1 + y2) // 2

            # -------- LEFT / RIGHT --------
            if px < cx - 40:
                state["auto_lr"] = "left"
            elif px > cx + 40:
                state["auto_lr"] = "right"
            else:
                state["auto_lr"] = "center"

            # -------- UP / DOWN (Turret) --------
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
    threading.Thread(target=video_thread, daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()
    threading.Thread(target=metadata_thread, daemon=True).start()
    threading.Thread(target=detection_thread, daemon=True).start()

    print("[SERVER] all subsystems online (H.264 + JSON + detection + control)")

    while True:
        time.sleep(1)
