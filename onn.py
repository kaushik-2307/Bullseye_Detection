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

# Detection thresholds
CONF_THRESH = 0.25
IOU_THRESH = 0.45

# Load ONNX model
print("Loading YOLOv8 ONNX model...")
session = rt.InferenceSession('best.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_shape = session.get_outputs()[0].shape
print(f"Model loaded successfully")
print(f"Input shape: {input_shape}")
print(f"Output shape: {output_shape}")

# Camera initialization
try:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    picam2.start()
    PICAMERA2_AVAILABLE = True
    print("Picamera2 started")
except Exception as e:
    print(f"Picamera2 not available: {e}")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Unable to open webcam. Check connection.")
        exit()
    PICAMERA2_AVAILABLE = False
    print("Webcam connected successfully!")

def preprocess_frame(frame):
    """Preprocess frame for YOLOv8 ONNX (handles RGBA input from Picamera2)"""
    h, w = frame.shape[:2]
    
    # Remove alpha channel if present (Picamera2 outputs RGBA with 4 channels)
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]  # Keep only RGB channels
    
    # Resize to 640x640
    img = cv2.resize(frame, (640, 640))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert HWC to CHW format: (640, 640, 3) -> (3, 640, 640)
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension: (3, 640, 640) -> (1, 3, 640, 640)
    img = np.expand_dims(img, axis=0)
    
    return img, (h, w)

def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression to filter overlapping detections"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def postprocess_yolov8(outputs, original_shape, conf_thresh=0.25, iou_thresh=0.45):
    """
    Parse YOLOv8 ONNX outputs and apply NMS
    YOLOv8 output format: [1, 84, 8400] 
    - 84 = 4 bbox coords + 80 class scores (no objectness score)
    - 8400 = number of predictions
    """
    output = outputs[0]  # Shape: (1, 84, 8400) or (1, num_classes+4, num_predictions)
    
    # Transpose to (num_predictions, num_classes+4)
    if len(output.shape) == 3:
        output = output[0].T  # (8400, 84)
    else:
        output = output.T
    
    h, w = original_shape
    scale_x = w / 640
    scale_y = h / 640
    
    boxes = []
    scores = []
    class_ids = []
    
    for pred in output:
        # YOLOv8: first 4 values are bbox coordinates (x_center, y_center, width, height)
        # remaining values are class scores (already processed, no sigmoid needed for YOLOv8)
        cx, cy, bw, bh = pred[:4]
        class_scores = pred[4:]
        
        # Find class with highest score
        class_id = np.argmax(class_scores)
        confidence = float(class_scores[class_id])
        
        if confidence >= conf_thresh:
            # Convert center coordinates to corner coordinates
            x1 = (cx - bw/2) * scale_x
            y1 = (cy - bh/2) * scale_y
            x2 = (cx + bw/2) * scale_x
            y2 = (cy + bh/2) * scale_y
            
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
            class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression
    if len(boxes) > 0:
        keep_indices = nms(boxes, scores, iou_thresh)
        detections = []
        for idx in keep_indices:
            detections.append({
                'box': [int(b) for b in boxes[idx]],
                'conf': float(scores[idx]),
                'class': int(class_ids[idx])
            })
        return detections
    return []

prev_time = 0

print("\nStarting YOLOv8 detection loop. Press 'q' to quit.\n")

while True:
    # Capture frame
    if PICAMERA2_AVAILABLE:
        frame = picam2.capture_array()  # RGBA format from Picamera2
        if frame is None:
            print("No frame captured from Picamera2.")
            break
    else:
        ret, frame_bgr = cap.read()
        if not ret:
            print("No frame captured. Check camera feed.")
            break
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocess frame for ONNX
    img_input, orig_shape = preprocess_frame(frame)
    
    # Run inference
    outputs = session.run(None, {input_name: img_input})
    
    # Postprocess detections with NMS
    detections = postprocess_yolov8(outputs, orig_shape, CONF_THRESH, IOU_THRESH)
    
    # Convert RGB to BGR for OpenCV display
    display_frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
    
    # Draw detections on frame
    for det in detections:
        x1, y1, x2, y2 = det['box']
        conf = det['conf']
        class_id = det['class']
        
        # Get class name
        if class_id < len(NAMES):
            label = f"{NAMES[class_id]} {conf:.2f}"
        else:
            label = f"Class {class_id} {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put label with background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 4), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(display_frame, label, (x1, y1 - 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    
    # Calculate average confidence
    confidences = [det['conf'] for det in detections]
    avg_conf = float(np.mean(confidences)) if confidences else 0
    
    # Display stats on frame
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Avg Conf: {avg_conf:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Objects: {len(detections)}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # Show frame
    cv2.imshow("YOLOv8 ONNX Detection", display_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
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
print("\nYOLOv8 detection stopped.")
