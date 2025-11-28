#!/usr/bin/env python3
import socket
import threading
import struct
import json
import cv2
import numpy as np
import time
import queue

VIDEO_HOST = "127.0.0.1"
CONTROL_HOST = "127.0.0.1"
META_HOST = "127.0.0.1"

VIDEO_PORT = 8000
CONTROL_PORT = 8001
META_PORT = 8002

cmd_queue = queue.Queue()
latest_frame = None
frame_lock = threading.Lock()

state = {
    "mode": "manual",
    "command": "none",
    "shoot": False,
    "auto_lr": "none",
    "auto_ud": "none",
    "persons": [],
    "selected": -1
}
state_lock = threading.Lock()


def recv_all(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


# ================= METADATA =================

def metadata_thread():
    global state
    while True:
        sock = None
        try:
            sock = socket.socket()
            sock.connect((META_HOST, META_PORT))
            print("[META] Connected")

            while True:
                hdr = recv_all(sock, 4)
                if hdr is None:
                    break
                size = struct.unpack(">I", hdr)[0]
                payload = recv_all(sock, size)
                if payload is None:
                    break

                with state_lock:
                    state = json.loads(payload.decode("utf-8"))

        except Exception as e:
            print("[META ERROR]", e)
            time.sleep(1)
        finally:
            if sock:
                try:
                    sock.close()
                except:
                    pass


# ================= CONTROL =================

def control_thread():
    while True:
        sock = None
        try:
            sock = socket.socket()
            sock.connect((CONTROL_HOST, CONTROL_PORT))
            print("[CONTROL] Connected")
            f = sock.makefile("w")

            while True:
                cmd = cmd_queue.get()
                print("[CONTROL SEND]", cmd)
                f.write(cmd + "\n")
                f.flush()

        except Exception as e:
            print("[CONTROL ERROR]", e)
            time.sleep(1)
        finally:
            if sock:
                try:
                    sock.close()
                except:
                    pass


# ================= VIDEO =================

def video_thread():
    global latest_frame
    while True:
        sock = None
        try:
            sock = socket.socket()
            sock.connect((VIDEO_HOST, VIDEO_PORT))
            print("[VIDEO] Connected")

            while True:
                hdr = recv_all(sock, 4)
                if hdr is None:
                    break
                size = struct.unpack(">I", hdr)[0]
                jpg = recv_all(sock, size)
                if jpg is None:
                    break

                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                with frame_lock:
                    latest_frame = frame

        except Exception as e:
            print("[VIDEO ERROR]", e)
            time.sleep(1)
        finally:
            if sock:
                try:
                    sock.close()
                except:
                    pass


# ================= MAIN (HUD + KEYS) =================

if __name__ == "__main__":
    print("[CLIENT] Starting...")

    threading.Thread(target=metadata_thread, daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()
    threading.Thread(target=video_thread, daemon=True).start()

    print("[CLIENT] Running (WASD = move, f = shoot, m = manual, o = auto, q = quit)")

    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is not None:
            with state_lock:
                st = state.copy()

            # HUD text
            cv2.putText(frame, f"Mode: {st['mode']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Cmd: {st['command']}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            cv2.putText(frame, f"Shooter: {'ON' if st['shoot'] else 'OFF'}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if st['shoot'] else (0, 0, 255), 2)
            cv2.putText(frame, f"Auto-LR: {st['auto_lr']}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Auto-UD: {st['auto_ud']}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # boxes
            for i, p in enumerate(st["persons"]):
                x1, y1, x2, y2 = p
                color = (0, 255, 0) if i == st["selected"] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # yellow dot on selected
            if st["selected"] != -1 and 0 <= st["selected"] < len(st["persons"]):
                x1, y1, x2, y2 = st["persons"][st["selected"]]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

            cv2.imshow("DiNoZa HUD", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("w"), ord("a"), ord("s"), ord("d")]:
            cmd_queue.put(chr(key))
        elif key == ord("f"):
            cmd_queue.put("f")
        elif key == ord("m"):
            cmd_queue.put("manual")
        elif key == ord("o"):
            cmd_queue.put("auto")
        elif key == ord("q"):
            break

        time.sleep(0.01)

    cv2.destroyAllWindows()
