import socket
import struct
import threading
import cv2
import numpy as np

SERVER_IP = "100.102.74.23"  # 👈 change to your Pi's IP
VIDEO_PORT = 8000
CONTROL_PORT = 8001

client_state = {
    "mode": "auto"
}


def video_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, VIDEO_PORT))
    print("[VIDEO] Connected to server")

    data_buffer = b""

    try:
        while True:
            # we first need 4 bytes for the length
            while len(data_buffer) < 4:
                chunk = sock.recv(4096)
                if not chunk:
                    print("[VIDEO] Disconnected from server")
                    return
                data_buffer += chunk

            frame_len = struct.unpack(">I", data_buffer[:4])[0]
            data_buffer = data_buffer[4:]

            # now read the full frame
            while len(data_buffer) < frame_len:
                chunk = sock.recv(4096)
                if not chunk:
                    print("[VIDEO] Disconnected from server")
                    return
                data_buffer += chunk

            frame_data = data_buffer[:frame_len]
            data_buffer = data_buffer[frame_len:]

            # decode JPEG
            np_data = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            cv2.imshow("Pi Stream", frame)
            key = cv2.waitKey(1) & 0xFF

            # Handle keys locally (we just update some flag; actual sending is in control thread)
            if key == ord('q'):
                print("[CLIENT] Quit requested")
                return

            # store last key somewhere if needed
    finally:
        sock.close()
        cv2.destroyAllWindows()


def control_sender():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, CONTROL_PORT))
    print("[CONTROL] Connected to server")

    try:
        while True:
            # Use OpenCV window for keyboard polling
            key = cv2.waitKey(50) & 0xFF
            if key == 255:  # no key
                continue

            # Quit
            if key == ord('q'):
                print("[CONTROL] Quit requested")
                break

            # Toggle mode
            if key == ord('m'):
                if client_state["mode"] == "auto":
                    client_state["mode"] = "manual"
                else:
                    client_state["mode"] = "auto"
                msg = f"MODE {client_state['mode'].upper()}\n"
                print("[CONTROL] Sending:", msg.strip())
                sock.sendall(msg.encode())

            # WASD (manual movement)
            if client_state["mode"] == "manual":
                if key in (ord('w'), ord('a'), ord('s'), ord('d')):
                    cmd = chr(key).upper()
                    msg = f"CMD {cmd}\n"
                    print("[CONTROL] Sending:", msg.strip())
                    sock.sendall(msg.encode())

    finally:
        sock.close()


if __name__ == "__main__":
    # Run both in parallel:
    # - video_receiver: receiving and showing frames
    # - control_sender: sending mode/WASD via keyboard
    vr_thread = threading.Thread(target=video_receiver, daemon=True)
    vr_thread.start()

    control_sender()
    print("[CLIENT] Exiting")
