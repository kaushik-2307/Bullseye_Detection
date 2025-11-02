import cv2
import time
import numpy as np
import onnxruntime as rt

# List of your class names in order (must match your model!)
NAMES = [
    'Bicycle', 'Fire-extinguisher', 'ball-basketball', 'ball-football', 'ball-volleyball',
    'barrel', 'cone-queuemanager cone', 'cone-sportscone', 'cone-trafficcone', 'drum',
    'gunnybag', 'jerrycan', 'lifeguard tube', 'motorbike', 'scooter', 'watercans'
]

# Confidence threshold for displayed detections
CONF_THRESH = 0.55

# Load ONNX model
session = rt.InferenceSession('best.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
print(f"ONNX model loaded. Input: {input_name}, Outputs: {output_names}")

# Camera initialization: prefer Picamera2 on Raspberry Pi 5 (libcamera),
# otherwise fall back to OpenCV VideoCapture.
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

if PICAMERA2_AVAILABLE:
    picam2 = Picamera2()
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

def preprocess_frame(frame):
    """Preprocess frame for ONNX model (640x640, normalized)"""
    h, w = frame.shape[:2]
    # Resize to 640x640
    img = cv2.resize(frame, (640, 640))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    # Convert to NCHW format (1, 3, 640, 640)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, (h, w)

def postprocess_results(outputs, original_shape, conf_thresh=0.55):
    """Parse ONNX model outputs (typically shape: 1, 25200, 85 for YOLOv5)"""
    detections = []
    
    # outputs[0] shape: (1, 25200, 85) - [x, y, w, h, conf, class_probs...]
    output = outputs[0][0]  # Get first batch
    
    h, w = original_shape
    scale_x = w / 640
    scale_y = h / 640
    
    for pred in output:
        conf = pred[4]  # Confidence score
        if conf >= conf_thresh:
            # Get class with highest probability
            class_probs = pred[5:]
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            
            # Decode bounding box (ONNX outputs center_x, center_y, width, height)
            cx, cy, bw, bh = pred[:4]
            x1 = (cx - bw/2) * scale_x
            y1 = (cy - bh/2) * scale_y
            x2 = (cx + bw/2) * scale_x
            y2 = (cy + bh/2) * scale_y
            
            detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'conf': float(conf),
                'class': int(class_id),
                'class_conf': float(class_conf)
            })
    
    return detections

while True:
    if PICAMERA2_AVAILABLE:
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

    # Preprocess and run inference
    img_input, orig_shape = preprocess_frame(frame)
    outputs = session.run(output_names, {input_name: img_input})
    
    # Postprocess detections
    detections = postprocess_results(outputs, orig_shape, CONF_THRESH)
    
    # Draw results on original frame
    annotated = frame.copy()
    class_counts = {}
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        conf = det['conf']
        class_id = det['class']
        
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        label = f"{NAMES[class_id]} {conf:.2f}" if class_id < len(NAMES) else f"{class_id} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Calculate FPS and display stats
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    
    confidences = [det['conf'] for det in detections]
    avg_conf = float(np.mean(confidences)) if confidences else 0
    
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f"Avg Conf: {avg_conf:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(annotated, f"Objects: {len(detections)}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    cv2.imshow("YOLOv5 ONNX Detection", annotated)
    
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
