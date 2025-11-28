#!/usr/bin/env python3
import socket
import threading
import struct
import json
import time

import cv2
import numpy as np

# ================= GLOBAL STATE =================

state = {
    "mode": "manual",      # "manual" / "auto"
    "command": "none",     # last WASD press
    "shoot": False,        # shooter toggle
    "auto_lr": "none",     # "left" / "right" / "center" / "none"
    "auto_ud": "none",     # "up" / "down" / "center" / "none"
    "persons": [],         # list of [x1,y1,x2,y2]
    "selected": -1         # index of target
}
state_lock = threading.Lock()

latest_frame = None
frame_lock = threading.Lock()

PERSON_CLASS = 15

net = cv2.dnn.readNetFromCaffe(
    "./model/MobileNetSSD_deploy.prototxt",
    "./model/MobileNetSSD_deploy.caffemodel"
)


# ================= CAMERA THREAD =================

def camera_thread():
    """Grab frames from Mac webcam into latest_frame."""
    global latest_frame

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[CAM] ERROR: cannot open webcam")
        return

    print("[CAM] Webcam online")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Some Mac builds give BGRA, so normalize to BGR
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Flip to mirror user
        frame = cv2.flip(frame, 1)

        with frame_lock:
            latest_frame = frame.copy()


# ================= VIDEO THREAD (MJPEG) =================

def video_thread():
    """Send MJPEG frames: [4-byte len][jpeg bytes] on port 8000."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", 8000))
    srv.listen(1)
    print("[VIDEO] Listening on :8000...")

    while True:
        conn, addr = srv.accept()
        print("[VIDEO] Client connected:", addr)

        try:
            while True:
                with frame_lock:
                    if latest_frame is None:
                        continue
                    frame = latest_frame.copy()

                ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if not ok:
                    continue

                data = jpg.tobytes()
                conn.sendall(struct.pack(">I", len(data)) + data)
                time.sleep(1/30)

        except Exception as e:
            print("[VIDEO ERROR]", e)

        finally:
            try:
                conn.close()
            except:
                pass
            print("[VIDEO] Client disconnected, waiting for new client...")


# ================= METADATA THREAD =================

def metadata_thread():
    """Send JSON-encoded state on port 8002."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", 8002))
    srv.listen(1)
    print("[META] Listening on :8002...")

    while True:
        conn, addr = srv.accept()
        print("[META] Client connected:", addr)

        try:
            while True:
                with state_lock:
                    payload = json.dumps(state).encode("utf-8")

                header = struct.pack(">I", len(payload))
                conn.sendall(header + payload)

                time.sleep(0.03)

        except Exception as e:
            print("[META ERROR]", e)

        finally:
            try:
                conn.close()
            except:
                pass
            print("[META] Client disconnected, waiting for new client...")


# ================= CONTROL THREAD =================

def control_thread():
    """Receive commands (WASD, f, manual, auto) on port 8001."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", 8001))
    srv.listen(1)
    print("[CONTROL] Listening on :8001...")

    while True:
        conn, addr = srv.accept()
        print("[CONTROL] Client connected:", addr)

        try:
            f = conn.makefile("r")
            for line in f:
                cmd = line.strip()
                if not cmd:
                    continue

                print("[CONTROL] CMD:", cmd)

                with state_lock:
                    if cmd in ["w", "a", "s", "d"]:
                        state["command"] = cmd
                    elif cmd == "manual":
                        state["mode"] = "manual"
                    elif cmd == "auto":
                        state["mode"] = "auto"
                    elif cmd == "f":
                        state["shoot"] = not state["shoot"]

        except Exception as e:
            print("[CONTROL ERROR]", e)

        finally:
            try:
                conn.close()
            except:
                pass
            print("[CONTROL] Client disconnected, waiting for new client...")


# ================= DETECTION THREAD =================

def detection_thread():
    """Run MobileNetSSD on latest_frame and update state."""
    global latest_frame

    tick = 0

    while True:
        try:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                img = latest_frame.copy()

            h, w = img.shape[:2]

            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)),
                0.007843,
                (300, 300),
                127.5,
                swapRB=False  # BGR
            )
            net.setInput(blob)
            det = net.forward()

            persons = []

            for i in range(det.shape[2]):
                conf = float(det[0, 0, i, 2])
                cls_id = int(det[0, 0, i, 1])

                if cls_id == PERSON_CLASS and conf > 0.35:
                    box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    if (x2 - x1) < 20 or (y2 - y1) < 40:
                        continue
                    persons.append([int(x1), int(y1), int(x2), int(y2)])

            with state_lock:
                state["persons"] = persons

                if not persons:
                    state["selected"] = -1
                    state["auto_lr"] = "none"
                    state["auto_ud"] = "none"
                else:
                    cx = w // 2
                    cy = h // 2

                    sel = min(
                        range(len(persons)),
                        key=lambda idx: abs((persons[idx][0] + persons[idx][2]) // 2 - cx)
                    )
                    state["selected"] = sel

                    x1, y1, x2, y2 = persons[sel]
                    px = (x1 + x2) // 2
                    py = (y1 + y2) // 2

                    if px < cx - 30:
                        state["auto_lr"] = "left"
                    elif px > cx + 30:
                        state["auto_lr"] = "right"
                    else:
                        state["auto_lr"] = "center"

                    if py < cy - 30:
                        state["auto_ud"] = "up"
                    elif py > cy + 30:
                        state["auto_ud"] = "down"
                    else:
                        state["auto_ud"] = "center"

            tick += 1
            if tick % 30 == 0:
                # debug every ~1s
                print(f"[DETECT] persons={len(persons)}, selected={state['selected']}")

        except Exception as e:
            print("[DETECT ERROR]", e)
            time.sleep(0.1)


# ================= MAIN =================

if __name__ == "__main__":
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=video_thread, daemon=True).start()
    threading.Thread(target=metadata_thread, daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()
    threading.Thread(target=detection_thread, daemon=True).start()

    print("[SERVER] Running on Mac (MJPEG + SSD + metadata)")

    while True:
        time.sleep(1)
