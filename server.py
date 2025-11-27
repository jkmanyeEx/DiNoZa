import socket
import struct
import threading
import cv2
import numpy as np
from picamzero import Camera

VIDEO_HOST = "0.0.0.0"
VIDEO_PORT = 8000
CONTROL_HOST = "0.0.0.0"
CONTROL_PORT = 8001

PERSON = 15

# Load person detector
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

state = {
    "mode": "auto",
    "last_cmd": None
}

def handle_control(conn):
    print("[CONTROL] Client connected")
    try:
        buffer = b""
        while True:
            data = conn.recv(1024)
            if not data:
                break

            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.decode().strip()

                if line.startswith("MODE"):
                    state["mode"] = line.split()[1].lower()
                    print(f"[CONTROL] Mode set to {state['mode']}")

                elif line.startswith("CMD"):
                    cmd = line.split()[1].upper()
                    state["last_cmd"] = cmd
                    print(f"[CONTROL] Manual CMD = {cmd}")
                    # TODO: GPIO / motor control integration
    finally:
        conn.close()


def start_control_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((CONTROL_HOST, CONTROL_PORT))
    sock.listen(1)
    print(f"[CONTROL] Listening on port {CONTROL_PORT}")

    while True:
        conn, addr = sock.accept()
        print("[CONTROL] connection from:", addr)
        threading.Thread(target=handle_control, args=(conn,), daemon=True).start()


def start_video_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((VIDEO_HOST, VIDEO_PORT))
    sock.listen(1)
    print(f"[VIDEO] Listening on port {VIDEO_PORT}")

    conn, addr = sock.accept()
    print("[VIDEO] Client connected:", addr)

    # Initialize the Pi Camera
    cam = Camera(resolution=(640, 480), framerate=30)

    try:
        while True:
            # Capture frame from PiCamZero
            frame = cam.capture_array()

            h, w = frame.shape[:2]
            mid_x = w // 2

            # ---- PERSON DETECTION ----
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843,
                (300, 300),
                127.5
            )
            net.setInput(blob)
            det = net.forward()

            candidates = []

            for i in range(det.shape[2]):
                conf = det[0, 0, i, 2]
                cid = int(det[0, 0, i, 1])

                if cid == PERSON and conf > 0.5:
                    box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    cx = (x1 + x2) // 2
                    center_dist = abs(cx - mid_x)
                    candidates.append((center_dist, cx))

            person_detected = len(candidates) > 0

            # ---- AUTO LOGIC ----
            if person_detected:
                candidates.sort(key=lambda x: x[0])
                _, cx = candidates[0]

                if state["mode"] == "auto":
                    if cx < mid_x - 40:
                        print("[AUTO] Move LEFT")
                        # TODO send to motors
                    elif cx > mid_x + 40:
                        print("[AUTO] Move RIGHT")
                    else:
                        print("[AUTO] CENTERED")

            # ---- Send frame to client ----
            success, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                continue

            data = jpg.tobytes()
            conn.sendall(struct.pack(">I", len(data)) + data)

    finally:
        print("[VIDEO] Shutdown")
        conn.close()
        sock.close()
