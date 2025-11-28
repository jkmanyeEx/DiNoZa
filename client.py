import cv2
import json
import socket
import struct
import threading

PI_IP = "100.101.19.24"   # change this
VIDEO_PORT = 8000
CONTROL_PORT = 8001
META_PORT = 8002

state = {
    "persons": [],
    "selected": -1,
    "mode": "manual",
    "command": "none",
    "auto_dir": "none"
}


# -------------------------------
# Metadata Receiver (JSON)
# -------------------------------
def meta_thread():
    global state
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((PI_IP, META_PORT))

    while True:
        header = sock.recv(4)
        if not header:
            break
        length = struct.unpack(">I", header)[0]
        data = sock.recv(length)
        state = json.loads(data.decode())


# -------------------------------
# Control Sender (WASD or mode)
# -------------------------------
def control_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((PI_IP, CONTROL_PORT))

    print("[CONTROL] connected")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('m'):
            sock.sendall(b"manual")
        if key == ord('o'):
            sock.sendall(b"auto")

        if key in [ord('w'), ord('a'), ord('s'), ord('d'), ord('f')]:
            sock.sendall(bytes([key]))


# -------------------------------
# VIDEO Thread (H.264 → frames)
# -------------------------------
def video_thread():
    cap = cv2.VideoCapture(f"tcp://{PI_IP}:{VIDEO_PORT}")

    if not cap.isOpened():
        print("Failed to open video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended")
            break

        draw_hud(frame)
        cv2.imshow("DiNoZa HUD", frame)


def draw_hud(frame):
    h, w = frame.shape[:2]

    # persons
    for i, p in enumerate(state["persons"]):
        x1, y1, x2, y2 = p
        color = (0, 255, 0) if i != state["selected"] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # selected target dot
    if state["selected"] != -1:
        x1, y1, x2, y2 = state["persons"][state["selected"]]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    # HUD text
    cv2.putText(frame, f"MODE: {state['mode']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2)

    cv2.putText(frame, f"CMD: {state['command']}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 0), 2)

    cv2.putText(frame, f"AUTO LR: {state['auto_lr']}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2)

    cv2.putText(frame, f"AUTO UD: {state['auto_ud']}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 200, 100), 2)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    threading.Thread(target=meta_thread, daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()
    threading.Thread(target=video_thread, daemon=True).start()

    print("[CLIENT] all threads running")

    while True:
        pass
