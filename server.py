import socket
import struct
import threading
import cv2
import numpy as np
from picamera2 import Picamera2

VIDEO_HOST = "0.0.0.0"
VIDEO_PORT = 8000
CONTROL_HOST = "0.0.0.0"
CONTROL_PORT = 8001

PERSON = 15  # MobileNet SSD class ID for "person"

# Load detection model
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

state = {
    "mode": "auto",
    "last_cmd": None
}


# ------------------------
# CONTROL SERVER
# ------------------------
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
                    print("[CONTROL] Mode =", state["mode"])

                elif line.startswith("CMD"):
                    cmd = line.split()[1].upper()
                    state["last_cmd"] = cmd
                    print("[CONTROL] Manual CMD:", cmd)
                    # TODO: integrate GPIO / motor here

    finally:
        print("[CONTROL] Client disconnected")
        conn.close()


def start_control_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((CONTROL_HOST, CONTROL_PORT))
    sock.listen(1)
    print(f"[CONTROL] Listening on {CONTROL_PORT}")

    while True:
        conn, addr = sock.accept()
        print("[CONTROL] Connection from", addr)
        threading.Thread(target=handle_control, args=(conn,), daemon=True).start()


# ------------------------
# VIDEO SERVER (PICAMERA2)
# ------------------------
def start_video_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((VIDEO_HOST, VIDEO_PORT))
    sock.listen(1)
    print(f"[VIDEO] Listening on {VIDEO_PORT}")

    conn, addr = sock.accept()
    print("[VIDEO] Client connected:", addr)

    # Initialize Picamera2
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        buffer_count=4
    )
    picam.configure(config)
    picam.start()

    try:
        while True:
            # Capture array from PiCam
            frame = picam.capture_array()

            h, w = frame.shape[:2]
            mid_x = w // 2

            # ---- PERSON DETECTION ----
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5
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
                    elif cx > mid_x + 40:
                        print("[AUTO] Move RIGHT")
                    else:
                        print("[AUTO] CENTERED")

                    # TODO: send GPIO/motor signal here

            # ---- STREAM FRAME ----
            # Compress as JPEG
            success, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                continue

            data = jpg.tobytes()
            conn.sendall(struct.pack(">I", len(data)) + data)

    finally:
        picam.stop()
        conn.close()
        sock.close()
        print("[VIDEO] Shutdown")


# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    # Control server runs in background
    threading.Thread(target=start_control_server, daemon=True).start()

    # Video server runs foreground
    start_video_server()
