#!/usr/bin/env python3

# yolov8_tflite_pi5_picam_mission.py

# Raspberry Pi 5 — Picamera2 capture + TFLite + ByteTrack + Mission Control
# Optimized for drone survey at 8m altitude, 1-2 m/s speed
# Pauses mission and triggers servo when 'disaster-flood' detected
# Fire-and-forget MAVLink commands with detailed timestamp logging

import os
import time
import traceback
from datetime import datetime
from collections import Counter
import numpy as np
import cv2

# Picamera2 (Pi)
try:
    from picamera2 import Picamera2, Preview
    from libcamera import controls
except Exception as e:
    raise SystemExit("Picamera2 import failed: install picamera2 and libcamera. Error: " + str(e))

# tflite runtime prefered on Pi, fallback to tensorflow
Interpreter = None
try:
    # prefer tflite_runtime (lightweight)
    from tflite_runtime.interpreter import Interpreter as _RTInterpreter
    Interpreter = _RTInterpreter
    print("Using tflite_runtime.Interpreter")
except Exception:
    try:
        import tensorflow as tf
        from tensorflow.lite.python.interpreter import Interpreter as _TFInterpreterInternal
        Interpreter = _TFInterpreterInternal
        print("Using tensorflow.lite.Interpreter")
    except Exception as e:
        traceback.print_exc()
        raise SystemExit("Neither tflite_runtime nor tensorflow available. Install one in your venv.")

# ByteTrack
from cjm_byte_track.core import BYTETracker

# Pymavlink for mission control
try:
    from pymavlink import mavutil
    print("Pymavlink loaded successfully")
except Exception as e:
    raise SystemExit("Pymavlink import failed: pip install pymavlink. Error: " + str(e))

# -------------------------
# CONFIG
# -------------------------
NAMES = [
    'Bicycle', 'traffic cone', 'ball-basketball', 'ball-basket ball', 'ball-basket ball',
    'card board', 'cardboard', 'traffic cone', 'taffic cone', 'trafficcone',
    'disaster-flood', 'disaster flood', 'gunnybag', 'card board', 'card board',
    'mannequin', 'motorbike', 'sandbag', 'scooter', 'watercans'
]

CONF_THRESH = 0.25
IOU_THRESH = 0.45

# Path to your TFLite model on the Pi filesystem
MODEL_PATH = "/home/pi/models/best_10_float16.tflite"  # <<-- edit this path

# MAVLink connection string - adjust for your setup
# Examples: '/dev/ttyACM0' for USB, 'udp:127.0.0.1:14550' for SITL, '/dev/serial0' for Pi UART
MAVLINK_CONNECTION = '/dev/ttyACM0'  # <<-- edit this connection string
BAUD_RATE = 57600  # or 115200 depending on your setup

# Display / model sizes
DISPLAY_SIZE = 640
MODEL_INPUT_SIZE = 640

# zoom controls
ZOOM_FACTOR = 1.0
ZOOM_STEP = 0.1
ZOOM_MIN = 1.0
ZOOM_MAX = 4.0
LINE_FRACTION = 0.7

# UI colors / style
INFO_BG_ALPHA = 0.65
INFO_BG_COLOR = (20, 20, 20)
INFO_TEXT_COLOR = (220, 220, 220)
BOX_COLOR = (0, 200, 0)
FLOOD_BOX_COLOR = (0, 0, 255)  # Red for disaster-flood
LINE_COLOR = (0, 0, 220)
LINE_THICKNESS = 2
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICK = 1
WINDOW_TITLE = "YOLOv8 TFLite — Pi5 (Picamera2)"
LINE_Y = int(DISPLAY_SIZE * LINE_FRACTION)

# Disaster detection settings
DISASTER_CLASS_NAMES = ['disaster-flood', 'disaster flood']  # variants of the class name
SERVO_CHANNEL = 9
SERVO_ACTIVE_PWM = 1100
SERVO_IDLE_PWM = 1500
SERVO_DELAY_SEC = 12.0

# Command delays (in seconds) - no ACK waiting, just fire and forget
COMMAND_DELAY = 0.1  # Small delay between commands to avoid flooding

# Logging settings
LOG_ALL_DETECTIONS = True  # Set to False to only log disaster-flood detections

flood_detected = False
flood_handling = False

# -------------------------
# Helper function for timestamps
# -------------------------
def iso_timestamp():
    """Get current timestamp in ISO format with milliseconds"""
    return datetime.now().isoformat(timespec="milliseconds")

def log_with_timestamp(message, prefix="INFO"):
    """Print message with timestamp prefix"""
    print(f"[{iso_timestamp()}] [{prefix}] {message}")

# -------------------------
# MAVLink Connection Setup
# -------------------------
log_with_timestamp(f"Connecting to autopilot on {MAVLINK_CONNECTION}...", "MAVLINK")
try:
    master = mavutil.mavlink_connection(MAVLINK_CONNECTION, baud=BAUD_RATE)
    master.wait_heartbeat(timeout=10)
    log_with_timestamp(
        f"Heartbeat received from system {master.target_system} component {master.target_component}", 
        "MAVLINK"
    )
except Exception as e:
    log_with_timestamp(f"Failed to connect to autopilot: {e}", "WARNING")
    log_with_timestamp("Continuing without MAVLink - mission control disabled", "WARNING")
    master = None

# -------------------------
# MAVLink Helper Functions (Fire-and-Forget with Logging)
# -------------------------
def pause_mission():
    """Pause the current mission - no ACK waiting"""
    if master is None:
        log_with_timestamp("MAVLink not connected - cannot pause mission", "WARNING")
        return False
    try:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE,
            0,  # confirmation
            0,  # param1: 0=pause, 1=continue
            0, 0, 0, 0, 0, 0
        )
        log_with_timestamp("Mission PAUSE command sent", "MAVLINK")
        time.sleep(COMMAND_DELAY)
        return True
    except Exception as e:
        log_with_timestamp(f"Error pausing mission: {e}", "ERROR")
        return False

def resume_mission():
    """Resume the current mission - no ACK waiting"""
    if master is None:
        log_with_timestamp("MAVLink not connected - cannot resume mission", "WARNING")
        return False
    try:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE,
            0,  # confirmation
            1,  # param1: 1=continue
            0, 0, 0, 0, 0, 0
        )
        log_with_timestamp("Mission RESUME command sent", "MAVLINK")
        time.sleep(COMMAND_DELAY)
        return True
    except Exception as e:
        log_with_timestamp(f"Error resuming mission: {e}", "ERROR")
        return False

def set_servo(channel, pwm):
    """Set servo PWM value using MAV_CMD_DO_SET_SERVO - no ACK waiting"""
    if master is None:
        log_with_timestamp("MAVLink not connected - cannot set servo", "WARNING")
        return False
    try:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,  # confirmation
            channel,  # param1: servo channel
            pwm,      # param2: PWM value
            0, 0, 0, 0, 0
        )
        log_with_timestamp(f"Servo {channel} set to {pwm} PWM", "MAVLINK")
        time.sleep(COMMAND_DELAY)
        return True
    except Exception as e:
        log_with_timestamp(f"Error setting servo: {e}", "ERROR")
        return False

def handle_flood_detection():
    """Execute the flood detection response sequence - fire and forget all commands"""
    global flood_handling
    flood_handling = True

    print("\n" + "="*70)
    log_with_timestamp("DISASTER-FLOOD DETECTED! Executing response sequence...", "ALERT")
    print("="*70)

    # 1. Pause mission (fire and forget)
    pause_mission()
    time.sleep(0.2)  # Brief delay for command to propagate

    # 2. Activate servo (e.g., deploy marker/payload)
    set_servo(SERVO_CHANNEL, SERVO_ACTIVE_PWM)

    # 3. Wait for specified delay
    log_with_timestamp(f"Waiting {SERVO_DELAY_SEC} seconds for payload deployment...", "INFO")
    time.sleep(SERVO_DELAY_SEC)

    # 4. Return servo to idle
    set_servo(SERVO_CHANNEL, SERVO_IDLE_PWM)
    time.sleep(0.2)

    # 5. Resume mission
    resume_mission()

    log_with_timestamp("Response sequence completed. Mission resumed.", "INFO")
    print("="*70 + "\n")

    flood_handling = False

# -------------------------
# Load TFLite model
# -------------------------
if not os.path.isfile(MODEL_PATH):
    raise SystemExit(f"Model not found at {MODEL_PATH}")

try:
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=2)
    interpreter.allocate_tensors()
except Exception:
    traceback.print_exc()
    raise SystemExit("Failed to load TFLite model. Check compatibility and runtime.")

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()
input_shape = input_details.get('shape', None)
input_dtype = input_details.get('dtype', np.float32)
input_quant = input_details.get('quantization', (0.0, 0))

log_with_timestamp(f"Model loaded: {MODEL_PATH}", "MODEL")
log_with_timestamp(f"Input shape: {input_shape}, dtype: {input_dtype}", "MODEL")
log_with_timestamp(f"Classes: {len(NAMES)}", "MODEL")

# layout guess
if input_shape is not None and len(input_shape) == 4 and input_shape[1] == 3:
    MODEL_LAYOUT = 'NCHW'
else:
    MODEL_LAYOUT = 'NHWC'

# -------------------------
# Picamera2 setup with optimized settings for 8m altitude survey
# -------------------------
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (DISPLAY_SIZE, DISPLAY_SIZE)},
    buffer_count=2
)
picam2.configure(camera_config)
picam2.start()

# Optimized camera settings for 1 m/s drone speed at 8m altitude
# Adjust ExposureTime to 1000 for 2 m/s speed if needed
try:
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 0.0,           # Infinity focus for 8m altitude
        "ExposureTime": 2000,          # 2ms shutter for 1 m/s (2mm motion blur)
        "AnalogueGain": 2.0,           # Lower gain for cleaner image
        "AeEnable": False,             # Lock exposure for consistency
        "AwbEnable": True,             # Keep auto white balance
        "Contrast": 1.0,
        "Sharpness": 1.0
    })
    log_with_timestamp("Camera optimized for 1 m/s survey at 8m altitude", "CAMERA")
    log_with_timestamp("Settings: ExposureTime=2000μs, AnalogueGain=2.0, Manual Focus", "CAMERA")
except Exception as e:
    log_with_timestamp(f"Could not set all camera controls: {e}", "WARNING")

time.sleep(0.8)  # let camera warm up

# -------------------------
# Helpers
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -88, 88)))

def dequantize_if_needed(arr, quant_params):
    scale, zp = quant_params if quant_params is not None else (0.0, 0)
    if arr.dtype in (np.uint8, np.int8) and scale and scale != 0:
        return (arr.astype(np.float32) - zp) * scale
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    return arr.astype(np.float32)

def center_square_crop(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    return frame[y1:y1+side, x1:x1+side]

def apply_digital_zoom(square_frame, zoom_factor, out_size=DISPLAY_SIZE):
    h = square_frame.shape[0]
    if zoom_factor <= 1.0:
        cropped = square_frame
    else:
        r = int(h / (2.0 * zoom_factor))
        cx = h // 2
        cy = h // 2
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(h, cx + r)
        y2 = min(h, cy + r)
        cropped = square_frame[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return resized

def preprocess_frame_for_model(frame_rgb, size=MODEL_INPUT_SIZE):
    # frame_rgb: HxWx3 uint8 RGB
    img = cv2.resize(frame_rgb, (size, size)).astype(np.float32) / 255.0

    if MODEL_LAYOUT == 'NHWC':
        inp = img[None, :, :, :]
    else:
        inp = np.transpose(img, (2, 0, 1))[None, ...]

    if input_dtype in (np.uint8, np.int8):
        scale, zp = input_quant if input_quant is not None else (0.0, 0)
        if scale and scale != 0:
            q = np.round(inp / scale + zp).astype(input_dtype)
        else:
            q = np.clip(np.round(inp * 255.0), np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
        return q, (DISPLAY_SIZE, DISPLAY_SIZE)
    else:
        return inp.astype(np.float32), (DISPLAY_SIZE, DISPLAY_SIZE)

def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1)*(y2 - y1)
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
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def postprocess_yolov8(tflite_outputs, original_shape):
    # choose biggest output tensor
    if len(tflite_outputs) == 1:
        output = tflite_outputs[0]
    else:
        output = max(tflite_outputs, key=lambda x: x.size)

    output = dequantize_if_needed(output, (0.0, 0))

    if len(output.shape) == 3:
        if output.shape[1] < output.shape[2]:
            output = output[0].T
        else:
            output = output[0]
    else:
        output = output.reshape(-1, output.shape[-1])

    h_orig, w_orig = original_shape
    sx = w_orig / float(MODEL_INPUT_SIZE)
    sy = h_orig / float(MODEL_INPUT_SIZE)

    boxes, scores, class_ids = [], [], []
    num_features = output.shape[1]
    has_obj = (num_features == (4 + 1 + len(NAMES)))

    for p in output:
        if not np.all(np.isfinite(p)):
            continue

        if has_obj:
            obj = sigmoid(p[4])
            if obj < CONF_THRESH:
                continue
            class_probs = sigmoid(p[5:])
            cid = int(np.argmax(class_probs))
            conf = float(obj * class_probs[cid])
        else:
            class_part = p[4:]
            cid = int(np.argmax(class_part))
            conf = float(class_part[cid])

        if conf < CONF_THRESH or cid >= len(NAMES):
            continue

        cx, cy, bw, bh = p[0], p[1], p[2], p[3]
        max_val = max(cx, cy, bw, bh)

        if max_val <= 1.5:
            cx_px = cx * MODEL_INPUT_SIZE
            cy_px = cy * MODEL_INPUT_SIZE
            bw_px = bw * MODEL_INPUT_SIZE
            bh_px = bh * MODEL_INPUT_SIZE
        elif max_val <= MODEL_INPUT_SIZE + 1.0:
            cx_px = cx; cy_px = cy; bw_px = bw; bh_px = bh
        else:
            cx_px = cx; cy_px = cy; bw_px = bw; bh_px = bh

        if max_val > MODEL_INPUT_SIZE + 1.0:
            x1 = cx_px - bw_px/2
            y1 = cy_px - bh_px/2
            x2 = cx_px + bw_px/2
            y2 = cy_px + bh_px/2
        else:
            x1 = (cx_px - bw_px/2)*sx
            y1 = (cy_px - bh_px/2)*sy
            x2 = (cx_px + bw_px/2)*sx
            y2 = (cy_px + bh_px/2)*sy

        x1 = max(0, min(x1, w_orig-1))
        y1 = max(0, min(y1, h_orig-1))
        x2 = max(0, min(x2, w_orig-1))
        y2 = max(0, min(y2, h_orig-1))

        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(cid)

    keep = nms(boxes, scores, IOU_THRESH)
    dets = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        dets.append({
            "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "conf": float(scores[i]),
            "class": int(class_ids[i])
        })

    return dets

# -------------------------
# Tracker & state
# -------------------------
tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30)
track_to_class = {}
counted_tracks = set()
track_last_bottom = {}
class_counts = Counter()
first_detection_time = {}
crossing_time = {}

# UI helpers
def draw_info_panel(img, zoom, fps):
    panel_w, panel_h = 200, 80
    x0, y0 = 8, 8
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), INFO_BG_COLOR, -1)
    cv2.addWeighted(overlay, INFO_BG_ALPHA, img, 1 - INFO_BG_ALPHA, 0, img)

    text1 = f"Zoom: {zoom:.1f}x"
    text2 = f"FPS: {fps:.1f}"
    text3 = "FLOOD HANDLING" if flood_handling else "SURVEYING"

    cv2.putText(img, text1, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INFO_TEXT_COLOR, 1)
    cv2.putText(img, text2, (x0 + 8, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, INFO_TEXT_COLOR, 1)

    status_color = (255, 0, 0) if flood_handling else (0, 255, 0)
    cv2.putText(img, text3, (x0 + 8, y0 + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

def draw_counts_overlay(img, class_counts_dict):
    total = sum(class_counts_dict.values())
    lines = [f"Total: {total}"]

    nonzero = [(cid, cnt) for cid, cnt in class_counts_dict.items() if cnt > 0]
    nonzero_sorted = sorted(nonzero, key=lambda x: x[1], reverse=True)[:6]

    for cid, cnt in nonzero_sorted:
        name = NAMES[cid] if 0 <= cid < len(NAMES) else str(cid)
        lines.append(f"{name}: {cnt}")

    padding = 8
    line_h = 18
    panel_w = 220
    panel_h = padding * 2 + line_h * len(lines)

    x1 = img.shape[1] - panel_w - 10
    y1 = 10

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), INFO_BG_COLOR, -1)
    cv2.addWeighted(overlay, INFO_BG_ALPHA, img, 1 - INFO_BG_ALPHA, 0, img)

    for i, line in enumerate(lines):
        y = y1 + padding + line_h * i + 12
        cv2.putText(img, line, (x1 + padding, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INFO_TEXT_COLOR, 1, cv2.LINE_AA)

def print_summary():
    try:
        print("\n" + "="*70)
        log_with_timestamp("FINAL RESULTS SUMMARY", "SUMMARY")
        print("="*70)
        print("CLASS COUNTS:")
        total = 0
        for cid in sorted(class_counts.keys()):
            print(f"  {NAMES[cid]}: {class_counts[cid]}")
            total += class_counts[cid]
        print(f"Total unique crossings: {total}\n")

        print("TRACK TIMESTAMPS:")
        for tid in sorted(first_detection_time.keys()):
            cls_id = track_to_class.get(tid, -1)
            cls_name = NAMES[cls_id] if cls_id != -1 else "Unknown"
            t_first = first_detection_time[tid]
            t_cross = crossing_time.get(tid, "NO CROSS")
            print(f"Track {tid} | Class: {cls_name}")
            print(f"  First detected: {t_first}")
            print(f"  Crossed line  : {t_cross}")
            print("-" * 70)
        log_with_timestamp("Processing complete", "SUMMARY")
        print("="*70)
    except Exception:
        traceback.print_exc()

# -------------------------
# Main loop
# -------------------------
prev = time.time()
frame_idx = 0

cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_TITLE, DISPLAY_SIZE, DISPLAY_SIZE)

print("="*70)
log_with_timestamp("YOLOv8 TFLite — Raspberry Pi 5 + Picamera2 + Mission Control", "START")
log_with_timestamp("Disaster-flood detection active (Fire-and-Forget mode)", "START")
log_with_timestamp("Controls: + - z reset | q to quit", "START")
print("="*70 + "\n")

try:
    while True:
        # capture (RGB888)
        frame_rgb = picam2.capture_array()
        if frame_rgb is None:
            time.sleep(0.02)
            continue

        # center crop and zoom (operate on RGB image)
        square = center_square_crop(frame_rgb)
        zoomed_display_rgb = apply_digital_zoom(square, ZOOM_FACTOR, out_size=DISPLAY_SIZE)

        # prepare input for model (expects RGB)
        img_input, orig_shape = preprocess_frame_for_model(zoomed_display_rgb, MODEL_INPUT_SIZE)

        # set tensor and invoke
        input_index = input_details['index']
        if input_dtype in (np.uint8, np.int8):
            interpreter.set_tensor(input_index, img_input)
        else:
            interpreter.set_tensor(input_index, img_input.astype(np.float32))

        interpreter.invoke()

        # collect outputs
        outputs = []
        for od in output_details:
            out = interpreter.get_tensor(od['index'])
            outputs.append(dequantize_if_needed(out, od.get('quantization', (0.0, 0))))

        # postprocess (detections in DISPLAY coords)
        dets = postprocess_yolov8(outputs, orig_shape)

        # Log all detections if enabled
        if LOG_ALL_DETECTIONS and len(dets) > 0:
            detection_summary = []
            for d in dets:
                class_name = NAMES[d['class']] if d['class'] < len(NAMES) else f"Class{d['class']}"
                detection_summary.append(f"{class_name}({d['conf']:.2f})")
            log_with_timestamp(f"Frame {frame_idx}: Detected {len(dets)} objects: {', '.join(detection_summary)}", "DETECT")

        # Check for disaster-flood detection
        flood_detected_this_frame = False
        for d in dets:
            class_name = NAMES[d['class']] if d['class'] < len(NAMES) else ""
            if class_name in DISASTER_CLASS_NAMES:
                flood_detected_this_frame = True
                log_with_timestamp(
                    f"!!! DISASTER-FLOOD DETECTED !!! Confidence: {d['conf']:.2f}, BBox: {d['box']}", 
                    "ALERT"
                )
                break

        # Trigger flood response if detected and not already handling
        if flood_detected_this_frame and not flood_handling:
            handle_flood_detection()

        if dets:
            detections = np.array([[d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['conf']] for d in dets], dtype=np.float32)
        else:
            detections = np.empty((0,5), dtype=np.float32)

        # update tracker
        online_targets = tracker.update(
            output_results=detections,
            img_info=(DISPLAY_SIZE, DISPLAY_SIZE),
            img_size=(DISPLAY_SIZE, DISPLAY_SIZE)
        )

        # convert display image to BGR for OpenCV drawing & imshow
        zoomed_display_bgr = cv2.cvtColor(zoomed_display_rgb, cv2.COLOR_RGB2BGR)

        for track in online_targets:
            tlwh = track.tlwh
            tid = track.track_id

            if tid not in first_detection_time:
                first_detection_time[tid] = iso_timestamp()
                log_with_timestamp(f"New track initialized: ID {tid}", "TRACK")

            x1, y1 = int(tlwh[0]), int(tlwh[1])
            x2, y2 = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
            bottom_y = y2

            prev_bottom = track_last_bottom.get(tid, None)
            track_last_bottom[tid] = bottom_y

            # assign class via IoU
            if tid not in track_to_class:
                best_iou = 0.0
                best_cls = None
                for d in dets:
                    dx1, dy1, dx2, dy2 = d['box']
                    inter_x1 = max(x1, dx1)
                    inter_y1 = max(y1, dy1)
                    inter_x2 = min(x2, dx2)
                    inter_y2 = min(y2, dy2)
                    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                    area_t = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                    area_d = max(0.0, (dx2 - dx1)) * max(0.0, (dy2 - dy1))
                    union = area_t + area_d - inter + 1e-9
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_cls = d['class']
                if best_iou >= 0.25:
                    track_to_class[tid] = best_cls
                    class_name = NAMES[best_cls] if best_cls < len(NAMES) else f"Class{best_cls}"
                    log_with_timestamp(f"Track {tid} assigned to class: {class_name}", "TRACK")

            # crossing detection
            if prev_bottom is not None and prev_bottom < LINE_Y and bottom_y >= LINE_Y:
                if tid not in counted_tracks:
                    counted_tracks.add(tid)
                    cls_id = track_to_class.get(tid, -1)
                    if cls_id != -1:
                        class_counts[cls_id] += 1
                        class_name = NAMES[cls_id] if cls_id < len(NAMES) else f"Class{cls_id}"
                        crossing_time[tid] = iso_timestamp()
                        log_with_timestamp(
                            f"Track {tid} ({class_name}) crossed counting line! Total count: {class_counts[cls_id]}", 
                            "COUNT"
                        )

            # Determine box color based on class
            cls_id = track_to_class.get(tid, -1)
            cls_name = NAMES[cls_id] if cls_id != -1 else f"Cls{cls_id}"
            box_color = FLOOD_BOX_COLOR if (cls_id != -1 and NAMES[cls_id] in DISASTER_CLASS_NAMES) else BOX_COLOR

            # draw rect and label
            cv2.rectangle(zoomed_display_bgr, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

            if tid in track_to_class:
                label = f"{cls_name} ID:{tid}"
            else:
                label = f"ID:{tid}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICK)
            lx, ly = x1, max(0, y1 - th - 6)
            cv2.rectangle(zoomed_display_bgr, (lx, ly), (lx + tw + 8, ly + th + 6), box_color, -1)
            cv2.putText(zoomed_display_bgr, label, (lx + 4, ly + th + 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, (10,10,10), LABEL_FONT_THICK, cv2.LINE_AA)

        # draw counting line & overlays
        cv2.line(zoomed_display_bgr, (0, LINE_Y), (DISPLAY_SIZE, LINE_Y), LINE_COLOR, LINE_THICKNESS)

        now = time.time()
        fps = 1.0 / (now - prev) if (now - prev) > 0 else 0.0
        prev = now

        draw_info_panel(zoomed_display_bgr, ZOOM_FACTOR, fps)
        draw_counts_overlay(zoomed_display_bgr, class_counts)

        cv2.imshow(WINDOW_TITLE, zoomed_display_bgr)

        # keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log_with_timestamp("Quit signal received", "INFO")
            break
        elif key in (ord('+'), ord('=')):
            ZOOM_FACTOR = min(ZOOM_MAX, round(ZOOM_FACTOR + ZOOM_STEP, 2))
        elif key in (ord('-'), ord('_')):
            ZOOM_FACTOR = max(ZOOM_MIN, round(ZOOM_FACTOR - ZOOM_STEP, 2))
        elif key == ord('z'):
            ZOOM_FACTOR = 1.0

        frame_idx += 1

except KeyboardInterrupt:
    log_with_timestamp("Keyboard interrupt received", "INFO")

finally:
    try:
        picam2.stop()
        log_with_timestamp("Camera stopped", "INFO")
    except Exception:
        pass
    cv2.destroyAllWindows()
    print_summary()

    if master:
        master.close()
        log_with_timestamp("MAVLink connection closed", "MAVLINK")
