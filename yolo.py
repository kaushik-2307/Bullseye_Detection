import torch
import cv2
import time
import numpy as np

# List of your class names in order (must match your model!)
NAMES = [
    'Bicycle', 'Fire-extinguisher', 'ball-basketball', 'ball-football', 'ball-volleyball',
    'barrel', 'cone-queuemanager cone', 'cone-sportscone', 'cone-trafficcone', 'drum',
    'gunnybag', 'jerrycan', 'lifeguard tube', 'motorbike', 'scooter', 'watercans'
]

# confidence threshold for displayed detections
CONF_THRESH = 0.55

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Ensure model is on a safe device. Many Windows installs have a CPU-only torchvision
# where torchvision.ops.nms isn't available for CUDA. Default to CPU to avoid
# NotImplementedError from torchvision::nms. If you have a compatible CUDA
# torchvision build, change device manually to 'cuda:0'.
device = torch.device('cpu')
model.to(device).eval()
model.float()

# Camera initialization: prefer Picamera2 on Raspberry Pi 5 (libcamera),
# otherwise fall back to OpenCV VideoCapture.
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

if PICAMERA2_AVAILABLE:
    picam2 = Picamera2()
    # small preview size for real-time processing; adjust if you need higher res
    preview_config = picam2.create_preview_configuration({"main": {"size": (1280, 720), "format": "RGB888"}})
    picam2.configure(preview_config)
    picam2.start()
    print("Picamera2 started")
else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Unable to open webcam. Check connection.")
        exit()
    print("Webcam connected successfully!")

prev_time = 0

while True:
    if PICAMERA2_AVAILABLE:
        # Picamera2 returns RGB arrays; convert to BGR for OpenCV and our model input
        frame_rgb = picam2.capture_array()
        if frame_rgb is None:
            print("No frame captured from Picamera2.")
            break
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cap.read()
        if not ret:
            print("No frame captured. Check camera feed.")
            break

    # pass the original BGR frame to the model (the hub model handles conversion)
    # Try running inference; if torchvision.nms is not available for CUDA this
    # may raise NotImplementedError from a CUDA backend. In that case we move
    # the model to CPU and retry once.
    try:
        with torch.no_grad():
            results = model(frame, size=640)
    except NotImplementedError as e:
        # Detect torchvision::nms CUDA backend issue and fallback to CPU
        msg = str(e)
        if 'torchvision::nms' in msg or 'Could not run' in msg:
            print("torchvision NMS not available for CUDA backend, falling back to CPU.")
            model.to('cpu')
            with torch.no_grad():
                results = model(frame, size=640)
        else:
            raise

    # Draw directly on the original BGR frame to preserve real-life colors
    annotated = frame.copy()
    # Count per class and draw boxes/labels on annotated (BGR)
    det = results.xyxy[0]
    class_counts = {}
    det_np_all = det.cpu().numpy() if det.shape[0] > 0 else np.zeros((0,6))

    # filter by confidence threshold
    det_np = det_np_all[det_np_all[:, 4] >= CONF_THRESH]

    if det_np.shape[0] > 0:
        class_indices = det_np[:, 5].astype(int)
        for idx in class_indices:
            class_counts[idx] = class_counts.get(idx, 0) + 1

        # Draw boxes and labels (BGR colors)
        for *box, conf, cls in det_np:
            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            label = (f"{NAMES[cls]} {conf:.2f}" if cls < len(NAMES) else f"{cls} {conf:.2f}")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        print("No objects detected this frame.")

    confidences = det_np[:, 4] if det_np.shape[0] > 0 else []
    avg_conf = float(confidences.mean()) if len(confidences) else 0

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(annotated, f"Avg Conf: {avg_conf:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(annotated, f"Objects: {int(det.shape[0])}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    cv2.imshow("YOLOv5 Webcam Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if PICAMERA2_AVAILABLE:
    try:
        picam2.stop()
    except Exception:
        pass
else:
    try:
        cap.release()
    except Exception:
        pass

cv2.destroyAllWindows()