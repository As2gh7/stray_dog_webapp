
import os
import cv2
import uuid
import time
import json
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
import torch.nn as nn

# ========== CONFIG ==========
SIMILARITY_THRESHOLD = 0.88
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_FEATURES = 5
os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ========== Load YOLOv8 ==========
model = YOLO("models/best.pt")

# ========== Load ResNet50 ==========
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========== Load Persistent UUIDs ==========
MASTER_UUID_FILE = "logs/dog_log_master.json"
if os.path.exists(MASTER_UUID_FILE):
    with open(MASTER_UUID_FILE, "r") as f:
        master_data = json.load(f)
        known_features_dict = {
            dog_id: deque([np.array(feat) for feat in feats], maxlen=MAX_FEATURES)
            for dog_id, feats in master_data["features_by_dog"].items()
        }
        uuid_set = set(known_features_dict.keys())
else:
    known_features_dict = defaultdict(lambda: deque(maxlen=MAX_FEATURES))
    uuid_set = set()

log_by_id = defaultdict(list)

# ========== Feature Extraction ==========
def extract_features(image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet(image).squeeze(0).numpy()
    return features / np.linalg.norm(features)

# ========== UUID Assignment ==========
def get_or_assign_id(feat):
    for dog_id, feats in known_features_dict.items():
        for known_feat in feats:
            if cosine_similarity([feat], [known_feat])[0][0] > SIMILARITY_THRESHOLD:
                feats.append(feat)
                return dog_id
    new_id = str(uuid.uuid4())
    known_features_dict[new_id].append(feat)
    uuid_set.add(new_id)
    return new_id

# ========== Log Detection ==========
def log_detection(dog_id, bbox, conf):
    log_by_id[dog_id].append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bbox": bbox,
        "confidence": float(conf)
    })

# ========== Save Output Logs ==========
def save_output_log():
    session_filename = f"output_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_log = {
        "total_unique_dogs": len(uuid_set),
        "detections_by_dog": log_by_id
    }
    with open(f"output/{session_filename}", "w") as f:
        json.dump(output_log, f, indent=4)
    return session_filename

# ========== Update Persistent UUID DB ==========
def update_master_log():
    serializable_dict = {
        dog_id: [feat.tolist() for feat in feats]
        for dog_id, feats in known_features_dict.items()
    }
    with open(MASTER_UUID_FILE, "w") as f:
        json.dump({"features_by_dog": serializable_dict}, f, indent=4)

# ========== Process Frame ==========
def process_frame(frame):
    results = model(frame)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if model.names[cls].lower() == "stray_dog" and conf > 0.75:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:int(y1 + 0.6 * (y2 - y1)), x1:x2]
            if cropped.size == 0:
                continue
            feat = extract_features(cropped)
            dog_id = get_or_assign_id(feat)
            log_detection(dog_id, [x1, y1, x2, y2], conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, dog_id[:8], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
    return frame

# ========== Public Functions ==========
def process_uploaded_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "output/output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        processed = process_frame(frame)
        out.write(processed)

    cap.release()
    out.release()

    log_file = save_output_log()
    update_master_log()
    return output_path, f"output/{log_file}"

def process_webcam_video():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "output/output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        processed = process_frame(frame)
        out.write(processed)
        cv2.imshow("Live Detection (Press Q to Stop)", processed)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    log_file = save_output_log()
    update_master_log()
    return output_path, f"output/{log_file}"

