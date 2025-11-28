import cv2
import json
import socket
import struct
import threading

PI_IP = "100.101.19.24"
VIDEO_PORT = 8000
CONTROL_PORT = 8001
META_PORT = 8002

state = {}
state_lock = threading.Lock()


def recv_all(sock, length):
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def metadata_thread():
    global state
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((PI_IP, META_PORT))
    print("[META] connected")

    while True:
        hdr = sock.recv(4)
        if not hdr:
            break
        length = struct.unpack(">I", hdr)[0]
        payload = recv_all(sock, length)
        if not payload:
            break

        try:
            obj = json.loads(payload.decode())
            with state_lock:
                state = obj
        except:
            continue


def draw_hud(frame):
    with state_lock:
        persons = state.get("persons", [])
        selected = state.get("selected", -1)
        mode = state.get("mode", "manual")
        cmd = state.get("command", "")
        lr = state.get("auto_lr", "")
        ud = state.get("auto_ud", "")

    for i, (x1, y1, x2, y2) in enumerate(persons):
        color = (0, 255, 0) if i != selected else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if selected != -1:
        x1, y1, x2, y2 = persons[selected]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    cv2.putText(frame, f"MODE: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"CMD: {cmd}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"AUTO LR: {lr}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"AUTO UD: {ud}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)


if __name__ == "__main__":
    threading.Thread(target=metadata_thread, daemon=True).start()

    control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    control.connect((PI_IP, CONTROL_PORT))
    print("[CONTROL] connected")

    cap = cv2.VideoCapture(f"tcp://{PI_IP}:{VIDEO_PORT}")
    if not cap.isOpened():
        print("[VIDEO] failed to connect")
        exit()

    print("[CLIENT] running")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_hud(frame)
        cv2.imshow("DiNoZa HUD", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('m'):
            control.sendall(b"manual")
        elif key == ord('o'):
            control.sendall(b"auto")
        elif key in [ord('a'), ord('d'), ord('w'), ord('s'), ord('f')]:
            control.sendall(chr(key).encode())

    cap.release()
    cv2.destroyAllWindows()
