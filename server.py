import socket
import struct
import threading
import cv2
import numpy as np
from picamera2 import Picamera2

VIDEO_HOST = "0.0.0.0"
VIDEO_PORT = 8000
CONTROL_PORT = 8001

PERSON = 15

net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

state = {"mode": "auto", "last_cmd": None}


############################################################
# CONTROL SERVER
############################################################
def control_server():
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((VIDEO_HOST, CONTROL_PORT))
    sock.listen(1)
    print("[CONTROL] Listening")

    while True:
        conn, addr = sock.accept()
        print("[CONTROL] Client connected", addr)
        threading.Thread(target=control_handler, args=(conn,), daemon=True).start()


def control_handler(conn):
    buffer = b""
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                print("[CONTROL] Disconnected")
                break

            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.decode().strip()

                if line.startswith("MODE"):
                    state["mode"] = line.split()[1].lower()
                    print("[CONTROL] Mode =", state["mode"])

                elif line.startswith("CMD"):
                    state["last_cmd"] = line.split()[1].upper()
                    print("[CONTROL] Manual CMD =", state["last_cmd"])

    finally:
        conn.close()


############################################################
# VIDEO SERVER (Camera only initialized ONCE)
############################################################
def video_server():
    # ---- Initialize camera once ----
    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        buffer_count=3
    )
    cam.configure(config)
    cam.start()
    print("[VIDEO] Camera started")

    # ---- Socket ----
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((VIDEO_HOST, VIDEO_PORT))
    sock.listen(1)
    print("[VIDEO] Listening")

    while True:
        conn, addr = sock.accept()
        print("[VIDEO] Client connected:", addr)
        serve_video(conn, cam)
        print("[VIDEO] Client disconnected")
        conn.close()


############################################################
# VIDEO STREAM HANDLE
############################################################
def serve_video(conn, cam):
    while True:
        frame = cam.capture_array()
        h, w = frame.shape[:2]
        mid_x = w // 2

        # ---- Detection ----
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        det = net.forward()

        candidates = []

        for i in range(det.shape[2]):
            confidence = det[0, 0, i, 2]
            cls_id = int(det[0, 0, i, 1])

            if cls_id == PERSON and confidence > 0.5:
                box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                candidates.append((abs(cx - mid_x), cx, cy))

        # ---- Auto mode selection ----
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, cx, cy = candidates[0]
            cv2.circle(frame, (cx, cy), 12, (255, 0, 0), 2)

        # ---- Send frame ----
        ok, jpg = cv2.imencode(".jpg", frame, [80])
        if not ok: continue

        packet = struct.pack(">I", len(jpg))
        try:
            conn.sendall(packet + jpg.tobytes())
        except:
            break


############################################################
# MAIN
############################################################
if __name__ == "__main__":
    threading.Thread(target=control_server, daemon=True).start()
    video_server()
