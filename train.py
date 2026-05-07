import urllib.request
import os
import numpy as np
import cv2
import time
import imutils
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ── Download DNN face detector files if not present ───────────────────────────
FACE_DIR      = "face_detector"
PROTOTXT_PATH = os.path.join(FACE_DIR, "deploy.prototxt")
WEIGHTS_PATH  = os.path.join(FACE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master"
    "/samples/dnn/face_detector/deploy.prototxt"
)
WEIGHTS_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw"
    "/dnn_samples_face_detector_20170830"
    "/res10_300x300_ssd_iter_140000.caffemodel"
)

os.makedirs(FACE_DIR, exist_ok=True)

if not os.path.exists(PROTOTXT_PATH):
    print("[INFO] Downloading face detector prototxt...")
    urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)

if not os.path.exists(WEIGHTS_PATH):
    print("[INFO] Downloading face detector weights (~10 MB)...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)

# ─────────────────────────────────────────────────────────────────────────────
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]

    # FIX: SSD ResNet-10 face detector specifically expects 300x300 input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs  = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clamp to frame boundaries
            startX, startY = max(0, startX), max(0, startY)
            endX,   endY   = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            # Preprocess for MobileNetV2 (this model uses 224x224)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32, verbose=0)

    return (locs, preds)

# ── Load models ───────────────────────────────────────────────────────────────
print("[INFO] Loading face detector...")
faceNet = cv2.dnn.readNet(PROTOTXT_PATH, WEIGHTS_PATH)

print("[INFO] Loading mask detector model...")
maskNet = load_model("mask_detector.h5")

SMOOTHING   = 5
pred_buffer = []

# ── Start video stream ────────────────────────────────────────────────────────
print("[INFO] Starting video stream... Press 'q' to quit.")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    if frame is None:
        continue
        
    frame = imutils.resize(frame, width=640)
    h, w  = frame.shape[:2]

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    if len(locs) == 0:
        pred_buffer.clear()
        cv2.putText(
            frame, "Position your face in frame", (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2,
        )

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask_conf, no_mask_conf)    = pred

        pred_buffer.append((mask_conf, no_mask_conf))
        if len(pred_buffer) > SMOOTHING:
            pred_buffer.pop(0)
            
        avg_mask    = float(np.mean([p[0] for p in pred_buffer]))
        avg_no_mask = float(np.mean([p[1] for p in pred_buffer]))

        wearing_mask = avg_mask > avg_no_mask
        confidence   = max(avg_mask, avg_no_mask) * 100

        color = (0, 200, 0) if wearing_mask else (0, 0, 220)
        label = f"{'MASK' if wearing_mask else 'NO MASK'} {confidence:.0f}%"

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        pad      = 6
        label_y1 = max(startY - text_h - pad * 2 - baseline, 0)
        label_y2 = startY if startY - text_h - pad * 2 - baseline > 0 else text_h + pad * 2
        
        cv2.rectangle(
            frame,
            (startX, label_y1),
            (startX + text_w + pad * 2, label_y2),
            color, -1,
        )
        cv2.putText(
            frame, label,
            (startX + pad, label_y2 - pad - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(
        frame, "Face Mask Detector  |  Press Q to quit",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
    )

    cv2.imshow("Face Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()