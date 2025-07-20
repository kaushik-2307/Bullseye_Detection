import torch
import cv2
from pymavlink import mavutil

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='C:/Users/nayak/Documents/Bullseye_Detection/yolov5/runs/train/bullseye_detector8/weights/best.pt',
                       force_reload=True)
model.conf = 0.4  # confidence threshold

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]

    h, w = frame.shape[:2]
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rel_x, rel_y = (cx - w / 2) / w, (cy - h / 2) / h

        # Draw visuals
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        label = f"{conf:.2f} ({rel_x:.2f}, {rel_y:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        print(f"Normalized center coords: x={rel_x:.3f}, y={rel_y:.3f}")

    cv2.imshow("Bullseye Detector", frame)
    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
