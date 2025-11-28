import cv2
import json
import socket
import struct
import threading

PI_IP = "100.97.150.114"  # change to your Pi's IP
VIDEO_PORT = 8000
CONTROL_PORT = 8001
META_PORT = 8002

state = {
    "mode": "manual",
    "command": "none",
    "auto_lr": "none",
    "auto_ud": "none",
    "persons": [],
    "selected": -1,
    "shooter": False,
}
state_lock = threading.Lock()


def recv_all(sock, length):
    buf = b""
    while len(buf) < length:
        chunk = sock.recv(length - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def metadata_thread():
    global state

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((PI_IP, META_PORT))
    print("[META] connected")

    while True:
        hdr = sock.recv(4)
        if not hdr:
            print("[META] header lost")
            break

        length = struct.unpack(">I", hdr)[0]
        payload = recv_all(sock, length)
        if payload is None:
            print("[META] payload lost")
            break

        try:
            obj = json.loads(payload.decode())
        except Exception as e:
            print("[META] JSON error:", e)
            continue

        with state_lock:
            state = obj


def draw_hud(frame):
    with state_lock:
        persons = state.get("persons", [])
        selected = state.get("selected", -1)
        mode = state.get("mode", "manual")
        cmd = state.get("command", "none")
        auto_lr = state.get("auto_lr", "none")
        auto_ud = state.get("auto_ud", "none")
        shooter = bool(state.get("shooter", False))

    # draw bounding boxes
    for i, p in enumerate(persons):
        x1, y1, x2, y2 = p
        color = (0, 255, 0) if i != selected else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # draw target center dot
    if 0 <= selected < len(persons):
        x1, y1, x2, y2 = persons[selected]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    # HUD text
    cv2.putText(frame, f"MODE: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"CMD: {cmd}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"AUTO LR: {auto_lr}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"AUTO UD: {auto_ud}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)

    shoot_text = "ON" if shooter else "OFF"
    shoot_color = (0, 0, 255) if shooter else (200, 200, 200)
    cv2.putText(frame, f"SHOOT: {shoot_text}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, shoot_color, 2)


if __name__ == "__main__":
    # metadata
    threading.Thread(target=metadata_thread, daemon=True).start()

    # control socket
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    control_sock.connect((PI_IP, CONTROL_PORT))
    print("[CONTROL] connected")

    # video stream
    cap = cv2.VideoCapture(f"tcp://{PI_IP}:{VIDEO_PORT}")
    if not cap.isOpened():
        print("[VIDEO] failed to connect")
        exit(1)
    print("[VIDEO] connected")

    print("[CLIENT] running (q=quit, m=manual, o=auto, WASD move, f=fire toggle)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[VIDEO] stream ended")
            break

        draw_hud(frame)
        cv2.imshow("DiNoZa HUD", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # mode switch
        if key == ord("m"):
            control_sock.sendall(b"manual\n")
        elif key == ord("o"):
            control_sock.sendall(b"auto\n")

        # movement + fire
        elif key in [ord("a"), ord("d"), ord("w"), ord("s"), ord("f")]:
            control_sock.sendall(f"{chr(key)}\n".encode())

    cap.release()
    control_sock.close()
    cv2.destroyAllWindows()
