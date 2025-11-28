#!/usr/bin/env python3
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
#   MobileNetSSD PERSON DETECTOR
# ============================================================
MODEL_DIR = "./model"
PROTOTXT = f"{MODEL_DIR}/MobileNetSSD_deploy.prototxt"
CAFFE_MODEL = f"{MODEL_DIR}/MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFE_MODEL)
PERSON_ID = 15


# ============================================================
#   SHARED STATE
# ============================================================
state = {
    "mode": "manual",
    "command": "none",
    "auto_lr": "none",
    "auto_ud": "none",
    "shooter": False,
    "persons": [],
    "selected": -1
}

state_lock = threading.Lock()


# ============================================================
#   TYPE SANITIZER (Fix int64 JSON errors)
# ============================================================
def to_python(obj):
    """Convert numpy and non-serializable types to Python built-ins."""
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(i) for i in obj]
    if isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
        return obj.item()
    return obj


# ============================================================
#   MOTOR MANAGER
# ============================================================
motors = MotorControl({
    "stepper1": [5, 6, 13, 19],      # L298N pins for PAN
    "stepper2": [12, 16, 20, 21],    # L298N pins for TILT
    "dc": {"pin": 23}                # MOSFET gate pin for shooter
})


# ============================================================
#   VIDEO STREAM THREAD (Persistent)
# ============================================================
def video_thread(picam2):
    HOST, PORT = "0.0.0.0", 8000

    encoder = H264Encoder(bitrate=4_000_000)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print("[VIDEO] Ready on port 8000")

    while True:
        print("[VIDEO] Waiting for client...")
        conn, addr = server.accept()
        print(f"[VIDEO] Client connected: {addr}")

        sock_file = conn.makefile("wb")
        output = FileOutput(sock_file)

        try:
            picam2.start_encoder(encoder, output)
            print("[VIDEO] Encoder started")

            while True:
                time.sleep(0.25)

        except Exception as e:
            print("[VIDEO ERROR]", e)

        finally:
            print("[VIDEO] Cleaning encoder...")
            try:
                picam2.stop_encoder()
            except:
                pass
            sock_file.close()
            conn.close()
            print("[VIDEO] Client disconnected")


# ============================================================
#   CONTROL THREAD (Persistent)
# ============================================================
def control_thread():

    HOST, PORT = "0.0.0.0", 8001

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print("[CONTROL] Ready on port 8001")

    while True:
        print("[CONTROL] Waiting for client...")
        conn, addr = server.accept()
        print(f"[CONTROL] Client connected: {addr}")

        buffer = ""

        try:
            while True:
                data = conn.recv(32)
                if not data:
                    break

                buffer += data.decode(errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    msg = line.strip()

                    if msg == "":
                        continue

                    with state_lock:
                        if msg == "auto":
                            state["mode"] = "auto"
                        elif msg == "manual":
                            state["mode"] = "manual"
                        elif msg in ["a", "d", "w", "s", "f"]:
                            state["command"] = msg

                        mode = state["mode"]
                        cmd = state["command"]

                    # apply immediately
                    if mode == "manual":
                        motors.manual_control(cmd)

        except Exception as e:
            print("[CONTROL ERROR]", e)

        finally:
            conn.close()
            print("[CONTROL] Client disconnected")


# ============================================================
#   METADATA THREAD (Persistent)
# ============================================================
def metadata_thread():

    HOST, PORT = "0.0.0.0", 8002

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print("[META] Ready on port 8002")

    while True:
        print("[META] Waiting for client...")
        conn, addr = server.accept()
        print(f"[META] Client connected: {addr}")

        try:
            while True:
                with state_lock:
                    state["shooter"] = bool(motors.dc.state)
                    meta = json.dumps(to_python(state)).encode()

                conn.sendall(struct.pack(">I", len(meta)) + meta)
                time.sleep(0.03)

        except Exception as e:
            print("[META ERROR]", e)

        finally:
            conn.close()
            print("[META] Client disconnected")


# ============================================================
#   DETECTION THREAD
# ============================================================
def detection_thread(picam2):

    while True:
        frame = picam2.capture_array()
        (h, w) = frame.shape[:2]

        # SSD blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )
        net.setInput(blob)
        detections = net.forward()

        persons = []

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            cls = int(detections[0, 0, i, 1])

            if cls == PERSON_ID and conf > 0.35:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)
                persons.append([int(x1), int(y1), int(x2), int(y2)])

        # update state
        with state_lock:
            state["persons"] = persons

            if not persons:
                state["selected"] = -1
                state["auto_lr"] = "none"
                state["auto_ud"] = "none"
                continue

            # choose closest to center horizontally
            cx = w // 2
            cy = h // 2

            nearest = min(
                enumerate(persons),
                key=lambda x: abs((x[1][0] + x[1][2]) // 2 - cx)
            )

            sel = nearest[0]
            state["selected"] = sel

            (x1, y1, x2, y2) = persons[sel]
            px = (x1 + x2) // 2
            py = (y1 + y2) // 2

            # left-right decision
            if px < cx - 40:
                state["auto_lr"] = "left"
            elif px > cx + 40:
                state["auto_lr"] = "right"
            else:
                state["auto_lr"] = "center"

            # up-down decision
            if py < cy - 40:
                state["auto_ud"] = "up"
            elif py > cy + 40:
                state["auto_ud"] = "down"
            else:
                state["auto_ud"] = "center"

            # AUTO MODE EXECUTION
            if state["mode"] == "auto":
                motors.auto_control(state["auto_lr"], state["auto_ud"])

        time.sleep(0.03)


# ============================================================
#   MAIN
# ============================================================
if __name__ == "__main__":
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(cfg)
    picam2.start()

    print("[SERVER] Camera online")

    threading.Thread(target=video_thread, args=(picam2,), daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()
    threading.Thread(target=metadata_thread, daemon=True).start()
    threading.Thread(target=detection_thread, args=(picam2,), daemon=True).start()

    print("[SERVER] All subsystems online")

    while True:
        time.sleep(1)
