import cv2
import numpy as np

PERSON = 15  # MobileNet SSD class ID for person

net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # mirror view
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    mid_x = w // 2

    # prepare blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    det = net.forward()

    candidates = []

    # find people
    for i in range(det.shape[2]):
        conf = det[0, 0, i, 2]
        cid = int(det[0, 0, i, 1])
        if cid == PERSON and conf > 0.5:
            box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            dist = abs(cx - mid_x)
            candidates.append((dist, (x1, y1, x2, y2), (cx, cy), conf))

    person_detected = len(candidates) > 0

    # 🔥🔥🔥 Your logic here (runs every frame when there is someone)
    if person_detected:
        # Example action:
        pass

    # draw only the closest-to-center person
    if person_detected:
        candidates.sort(key=lambda x: x[0])
        _, (x1, y1, x2, y2), (cx, cy), conf = candidates[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"person {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)
        cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 0), 1)

    cv2.imshow("Person Detection (Closest to Center)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
