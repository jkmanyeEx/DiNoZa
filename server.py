import socket
import struct
import threading
import cv2
import numpy as np
from picamera2 import Picamera2

VIDEO_HOST = "0.0.0.0"
VIDEO_PORT = 8000
CONTROL_PORT = 8001

PERSON = 15  # class id for "person" in MobileNetSSD

# Load MobileNet SSD model (make sure these files exist in same dir)
net = cv2.dnn.readNetFromCaffe(
    "./model/MobileNetSSD_deploy.prototxt",
    "./model/MobileNetSSD_deploy.caffemodel"
)

state = {
    "mode": "auto",   # "auto" or "manual"
    "last_cmd": None  # latest manual command
}


############################################################
# CONTROL SERVER
############################################################
def control_server():
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((VIDEO_HOST, CONTROL_PORT))
    sock.listen(1)
    print("[CONTROL] Listening")

    while True:
        conn, addr = sock.accept()
        print("[CONTROL] Client connected", addr)
        threading.Thread(target=control_handler, args=(conn,), daemon=True).start()


def control_handler(conn):
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
                    mode = line.split()[1].lower()
                    state["mode"] = mode
                    print("[CONTROL] Mode =", mode)

                elif line.startswith("CMD"):
                    cmd = line.split()[1].upper()
                    state["last_cmd"] = cmd
                    print("[CONTROL] Manual CMD =", cmd)

    finally:
        conn.close()


############################################################
# VIDEO SERVER (camera initialized once)
############################################################
def video_server():
    # Initialize camera ONCE
    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        buffer_count=3
    )
    cam.configure(config)
    cam.start()
    print("[VIDEO] Camera started")

    # Setup TCP server
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((VIDEO_HOST, VIDEO_PORT))
    sock.listen(1)
    print("[VIDEO] Listening on", VIDEO_PORT)

    while True:
        conn, addr = sock.accept()
        print("[VIDEO] Client connected:", addr)
        serve_video(conn, cam)
        conn.close()
        print("[VIDEO] Client disconnected")


############################################################
# HANDLE VIDEO STREAM FOR ONE CLIENT
############################################################
def serve_video(conn, cam):
    while True:
        frame = cam.capture_array()
        h, w = frame.shape[:2]
        mid_x = w // 2

        # ------------- PERSON DETECTION -------------
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        confs = []

        # Collect raw boxes & confidences
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            cls = int(detections[0, 0, i, 1])

            if cls == PERSON and conf > 0.60:  # threshold tuned
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h
                confs.append(float(conf))

        candidates = []

        # ------------- NMS TO MERGE OVERLAPS -------------
        if len(boxes) > 0:
            idxs = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.60, nms_threshold=0.40)

            if len(idxs) > 0:
                # idxs might be list of lists or np array
                for i in np.array(idxs).flatten():
                    x, y, bw, bh = boxes[i]
                    x1, y1 = x, y
                    x2, y2 = x + bw, y + bh

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    # Center point
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                    # For AUTO logic: (distance to center, cx, cy)
                    candidates.append((abs(cx - mid_x), cx, cy))

        # Debug: how many persons after NMS
        # print("[DETECT] Persons after NMS:", len(candidates))

        # ------------- AUTO MODE: PICK CLOSEST TO CENTER -------------
        if state["mode"] == "auto" and candidates:
            # sort by |cx - mid_x| (closest to screen center)
            candidates.sort(key=lambda x: x[0])
            _, cx, cy = candidates[0]

            # highlight chosen target
            cv2.circle(frame, (cx, cy), 12, (255, 0, 0), 2)

            if cx < mid_x - 40:
                print("[AUTO] LEFT")
            elif cx > mid_x + 40:
                print("[AUTO] RIGHT")
            else:
                print("[AUTO] CENTER")

        # ------------- ENCODE & SEND FRAME -------------
        ok, jpg = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        try:
            conn.sendall(struct.pack(">I", len(jpg)) + jpg.tobytes())
        except (BrokenPipeError, ConnectionResetError, OSError):
            break


############################################################
# MAIN ENTRY
############################################################
if __name__ == "__main__":
    threading.Thread(target=control_server, daemon=True).start()
    video_server()
