#!/usr/bin/env python3
import socket
import threading
import struct
import json
import cv2
import numpy as np
import time


VIDEO_HOST = "0.0.0.0"
CONTROL_HOST = "0.0.0.0"
META_HOST = "0.0.0.0"

VIDEO_PORT = 8000
CONTROL_PORT = 8001
META_PORT = 8002


# =============================================================
# GLOBAL STATE FROM SERVER
# =============================================================
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


# =============================================================
# METADATA RECEIVER
# =============================================================
def metadata_thread():
    global state

    while True:
        try:
            sock = socket.socket()
            sock.connect((META_HOST, META_PORT))
            print("[META] Connected to server")

            while True:
                header = sock.recv(4)
                if not header:
                    break

                size = struct.unpack(">I", header)[0]
                payload = sock.recv(size)

                with state_lock:
                    state = json.loads(payload.decode())

        except Exception as e:
            print("[META ERROR]", e)
            time.sleep(1)
        finally:
            try:
                sock.close()
            except:
                pass


# =============================================================
# CONTROL SENDER
# =============================================================
def control_thread():
    while True:
        try:
            sock = socket.socket()
            sock.connect((CONTROL_HOST, CONTROL_PORT))
            print("[CONTROL] Connected to server")

            while True:
                key = cv2.waitKey(1) & 0xFF

                # WASD manual movement
                if key in [ord("w"), ord("a"), ord("s"), ord("d")]:
                    sock.send(chr(key).encode())

                # toggle shooter
                if key == ord("f"):
                    sock.send(b"f")

                # mode switch
                if key == ord("m"):    # manual
                    sock.send(b"manual")
                if key == ord("o"):    # auto
                    sock.send(b"auto")

                # quit
                if key == ord("q"):
                    exit(0)

                time.sleep(0.01)

        except Exception as e:
            print("[CONTROL ERROR]", e)
            time.sleep(1)
        finally:
            try:
                sock.close()
            except:
                pass


# =============================================================
# VIDEO RECEIVER
# =============================================================
def video_thread():
    """Receives H.264 stream → decode → HUD draw → show."""
    import av   # PyAV required (pip install av)

    while True:
        try:
            sock = socket.socket()
            sock.connect((VIDEO_HOST, VIDEO_PORT))
            print("[VIDEO] Connected to server")

            container = av.open(sock.makefile("rb"))

            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="bgr24")

                # flip to match detection orientation
                img = cv2.flip(img, 1)

                # ================= HUD =================
                with state_lock:
                    st = state.copy()

                # Show mode
                cv2.putText(img, f"Mode: {st['mode']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)

                # Show command
                cv2.putText(img, f"Cmd: {st['command']}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (200, 200, 200), 2)

                # Shooter status
                color = (0, 255, 0) if st["shoot"] else (0, 0, 255)
                cv2.putText(img, f"Shooter: {'ON' if st['shoot'] else 'OFF'}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

                # Auto direction
                cv2.putText(img, f"Auto-LR: {st['auto_lr']}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)

                cv2.putText(img, f"Auto-UD: {st['auto_ud']}",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)

                # ================= Draw Persons =================
                for i, p in enumerate(st["persons"]):
                    x1, y1, x2, y2 = p
                    color = (0, 255, 0) if i == st["selected"] else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                cv2.imshow("DiNoZa HUD", img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    exit(0)

        except Exception as e:
            print("[VIDEO ERROR]", e)
            time.sleep(1)
        finally:
            try:
                sock.close()
            except:
                pass


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("[CLIENT] Starting all threads")

    threading.Thread(target=metadata_thread, daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()
    threading.Thread(target=video_thread, daemon=True).start()

    while True:
        time.sleep(1)
