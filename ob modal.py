import cv2
import numpy as np
import time
import tflite  # or import tflite_runtime.interpreter as tflite

# Path to your YOLOv8 TFLite model
TFLITE_MODEL_PATH = "best_float16"

# Load TFLite model
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Use the laptop's built-in webcam
cap = cv2.VideoCapture(0)

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

    # Preprocess frame
    input_size = input_details[0]['shape'][1:3]  # (height, width)
    img = cv2.resize(frame, (input_size[1], input_size[0]))
    img = img / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    # Set input and run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Get output (You MUST adjust this to fit your model's output signature!)
    outputs = [interpreter.get_tensor(x['index']) for x in output_details]

    # ------ Example Postprocessing (You MUST adjust for your model) ------
    # This is a placeholder. You need to adapt this to your TFLite output!
    # For many YOLOv8 TFLite exports, output[0] is (1, N, 6): [x1, y1, x2, y2, confidence, class]
    detections = outputs[0][0]  # (N, 6)
    valid = detections[:, 4] > 0.55
    detections = detections[valid]
    annotated = frame.copy()
    avg_conf = 0
    num_objs = len(detections)
    if num_objs > 0:
        avg_conf = detections[:, 4].mean()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{int(cls)}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(annotated, f"Avg Conf: {avg_conf:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(annotated, f"Objects: {num_objs}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    cv2.imshow("Webcam Live Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
