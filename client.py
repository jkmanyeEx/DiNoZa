import socket
import struct
import threading
import cv2
import numpy as np
import time

SERVER_IP = "100.101.19.24"
VIDEO_PORT = 8000
CONTROL_PORT = 8001

running = True
latest_frame = None
frame_lock = threading.Lock()

############################################################
# VIDEO RECEIVER
############################################################
def video_receiver():
    global latest_frame, running
    print("[VIDEO] Connecting...")

    sock = socket.socket()
    sock.connect((SERVER_IP, VIDEO_PORT))
    print("[VIDEO] Connected")

    buffer = b""

    while running:
        # read 4-byte header
        while len(buffer) < 4:
            data = sock.recv(4096)
            if not data:
                running = False
                return
            buffer += data

        size = struct.unpack(">I", buffer[:4])[0]
        buffer = buffer[4:]

        # read frame
        while len(buffer) < size:
            data = sock.recv(4096)
            if not data:
                running = False
                return
            buffer += data

        frame_data = buffer[:size]
        buffer = buffer[size:]

        img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        with frame_lock:
            latest_frame = img


############################################################
# CONTROL SENDER (NO waitKey here!)
############################################################
def control_sender(send_queue):
    sock = socket.socket()
    sock.connect((SERVER_IP, CONTROL_PORT))
    print("[CONTROL] Connected")

    while running:
        if send_queue:
            msg = send_queue.pop(0)
            sock.sendall(msg.encode())
        time.sleep(0.01)

    sock.close()


############################################################
# MAIN THREAD (handles keys safely)
############################################################
if __name__ == "__main__":
    send_queue = []

    threading.Thread(target=video_receiver, daemon=True).start()
    threading.Thread(target=control_sender, args=(send_queue,), daemon=True).start()

    print("[CLIENT] UI Loop starting")

    while running:
        with frame_lock:
            frame = latest_frame

        if frame is not None:
            cv2.imshow("DiNoZa Client", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            running = False
            break
        elif key == ord('m'):
            send_queue.append("MODE AUTO\n")
        elif key == ord('n'):
            send_queue.append("MODE MANUAL\n")
        elif key in (ord('w'), ord('a'), ord('s'), ord('d')):
            send_queue.append(f"CMD {chr(key).upper()}\n")

    cv2.destroyAllWindows()
