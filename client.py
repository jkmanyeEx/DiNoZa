import cv2
import json
import socket
import struct
import threading

PI_IP = "100.97.150.114"   # <- change if needed
VIDEO_PORT = 8000
CONTROL_PORT = 8001
META_PORT = 8002

state = {
    "persons": [],
    "selected": -1,
    "mode": "manual",
    "command": "none",
    "auto_lr": "none",
    "auto_ud": "none",
}
state_lock = threading.Lock()


# -------------------- metadata receiver --------------------
def recv_all(sock, length):
    buf = b""
    while len(buf) < length:
        chunk = sock.recv(length - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def meta_thread():
    global state
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((PI_IP, META_PORT))
    print("[META] connected")

    while True:
        header = sock.recv(4)
        if not header:
            print("[META] disconnected")
            break
        length = struct.unpack(">I", header)[0]
        data = recv_all(sock, length)
        if data is None:
            print("[META] recv error")
            break

        try:
            obj = json.loads(data.decode())
        except Exception as e:
            print("[META] JSON error:", e)
            continue

        with state_lock:
            state = obj


# -------------------- HUD drawing --------------------
def draw_hud(frame):
    with state_lock:
        persons = state.get("persons", [])
        selected = state.get("selected", -1)
        mode = state.get("mode", "manual")
        cmd = state.get("command", "none")
        auto_lr = state.get("auto_lr", "none")
        auto_ud = state.get("auto_ud", "none")

    h, w = frame.shape[:2]

    # draw persons
    for i, p in enumerate(persons):
        x1, y1, x2, y2 = p
        color = (0, 255, 0) if i != selected else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # selected dot
    if 0 <= selected < len(persons):
        x1, y1, x2, y2 = persons[selected]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    # HUD text
    cv2.putText(frame, f"MODE: {mode}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2)
    cv2.putText(frame, f"CMD: {cmd}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 0), 2)
    cv2.putText(frame, f"AUTO LR: {auto_lr}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)
    cv2.putText(frame, f"AUTO UD: {auto_ud}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 200, 100), 2)


# -------------------- main --------------------
if __name__ == "__main__":
    # start metadata listener
    threading.Thread(target=meta_thread, daemon=True).start()

    # control socket
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    control_sock.connect((PI_IP, CONTROL_PORT))
    print("[CONTROL] connected")

    # video capture (H.264 over TCP)
    cap = cv2.VideoCapture(f"tcp://{PI_IP}:{VIDEO_PORT}")
    if not cap.isOpened():
        print("[VIDEO] failed to open stream")
        exit(1)
    print("[VIDEO] connected to stream")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[VIDEO] stream ended / error")
            break

        draw_hud(frame)
        cv2.imshow("DiNoZa HUD", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # mode switching
        if key == ord('m'):
            control_sock.sendall(b"manual")
        if key == ord('o'):
            control_sock.sendall(b"auto")

        # turret control + shooting
        if key in [ord('a'), ord('d'), ord('w'), ord('s'), ord('f')]:
            control_sock.sendall(bytes([key]))

    cap.release()
    control_sock.close()
    cv2.destroyAllWindows()
