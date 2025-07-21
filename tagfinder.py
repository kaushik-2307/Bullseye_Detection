# 🚀 YOLOv5 Class Tag Debugger
import torch
import cv2

# === CONFIGURATION ===
YOLO_MODEL = 'C:/Users/nayak/Documents/Bullseye_Detection/yolov5/runs/train/bullseye_detector8/weights/best.pt'
CAMERA_SOURCE = 0  # 0 for webcam, or path to video file

# === LOAD YOLOv5 MODEL ===
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', YOLO_MODEL)
model.conf = 0.4

# === OPEN CAMERA STREAM ===
print("[INFO] Opening camera stream...")
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    raise IOError("Cannot open camera/stream")

print("[INFO] Starting detection (press 'q' to quit)...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame received, retrying...")
            continue

        # YOLOv5 detection
        results = model(frame)
        detections = results.pandas().xyxy[0]

        # Print unique class names
        detected_classes = detections['name'].unique()
        if len(detected_classes) > 0:
            print("[DEBUG] Detected classes:", detected_classes)
        else:
            print("[DEBUG] No detections in this frame")

        # Draw all detections
        for _, row in detections.iterrows():
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])),
                          (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(row['xmin']), int(row['ymin']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('YOLOv5 Class Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream closed.")
