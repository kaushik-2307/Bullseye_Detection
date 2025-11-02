import torch
import cv2
import time

# Load YOLOv5 model (official API)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt', force_reload=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("⚠️ Unable to open webcam. Check connection.")
    exit()

print("✅ Webcam connected successfully!")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No frame captured. Check camera feed.")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img, size=640)

    results.render()
    annotated = cv2.cvtColor(results.imgs[0], cv2.COLOR_RGB2BGR)

    confidences = results.xyxy[0][:, 4].cpu().numpy() if results.xyxy[0].shape[0] > 0 else []
    avg_conf = float(confidences.mean()) if len(confidences) else 0

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(annotated, f"Avg Conf: {avg_conf:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(annotated, f"Objects: {len(results.xyxy[0])}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    cv2.imshow("YOLOv5 Webcam Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()