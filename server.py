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

PERSON = 15  # MobileNetSSD class ID for "person"

# Load MobileNet SSD
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

state = {
    "mode": "auto",
    "last_cmd": None
}

###############################################
# CONTROL SERVER
###############################################

def start_control_server():
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    control_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    control_sock.bind((CONTROL_HOST, CONTROL_PORT))
    control_sock.listen(1)
    print(f"[CONTROL] Listening on {CONTROL_PORT}")

    while True:
        conn, addr = control_sock.accept()
        print("[CONTROL] Connection from:", addr)
        threading.Thread(target=handle_control, args=(conn,), daemon=True).start()


def handle_control(conn):
    print("[CONTROL] Client connected")
    buffer = b""

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                print("[CONTROL] Client disconnected")
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
                    print("[CONTROL] Manual CMD =", state["last_cmd"])

    except Exception as e:
        print("[CONTROL] Error:", e)

    finally:
        conn.close()


###############################################
# VIDEO SERVER (PICAMERA2)
###############################################

def start_video_server():
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    video_sock.bind((VIDEO_HOST, VIDEO_PORT))
    video_sock.listen(1)

    print(f"[VIDEO] Listening on {VIDEO_PORT}")

    while True:
        conn, addr = video_sock.accept()
        print("[VIDEO] Client connected:", addr)

        try:
            serve_video_stream(conn)
        except Exception as e:
            print("[VIDEO] Error:", e)
        finally:
            print("[VIDEO] Closing client connection")
            conn.close()


def serve_video_stream(conn):

    # Init Picamera2
    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        buffer_count=3
    )
    cam.configure(config)
    cam.start()

    print("[VIDEO] Camera started")

    while True:
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

        # ---- AUTO MODE LOGIC ----
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, cx = candidates[0]

            if state["mode"] == "auto":
                if cx < mid_x - 40:
                    print("[AUTO] LEFT")
                elif cx > mid_x + 40:
                    print("[AUTO] RIGHT")
                else:
                    print("[AUTO] CENTER")

        # ---- STREAM FRAME ----
        success, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            continue

        data = jpg.tobytes()
        packet = struct.pack(">I", len(data))

        try:
            conn.sendall(packet + data)
        except (BrokenPipeError, ConnectionResetError):
            print("[VIDEO] Client disconnected")
            break

    cam.stop()
    print("[VIDEO] Camera stopped")


###############################################
# MAIN
###############################################

if __name__ == "__main__":
    threading.Thread(target=start_control_server, daemon=True).start()
    start_video_server()
