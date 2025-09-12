from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import torch
from torchvision import models, transforms
import io
import torch.nn as nn
import json, os

# =====================
# Initialize FastAPI
# =====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === EDIT THESE PATHS ===
CKPT_PATH   = "models/rtk_resnet/resnet18_rtk_best.pth"
LABELS_PATH = "models/rtk_resnet/class_names.json"
TS_PATH     = "models/rtk_resnet/resnet18_rtk_scripted.pt"

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# =====================
# Preprocessing
# =====================
preproc = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# =====================
# Load labels
# =====================
with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)
print("Classes:", CLASS_NAMES)

# =====================
# Load model
# =====================
model = None
if os.path.isfile(TS_PATH):
    try:
        print("Loading TorchScript model:", TS_PATH)
        model = torch.jit.load(TS_PATH, map_location=device).eval().to(device)
    except Exception as e:
        print("TorchScript load failed, falling back to .pth:", e)

if model is None:
    print("Loading PyTorch checkpoint:", CKPT_PATH)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state = ckpt.get("model_state", ckpt)
    m.load_state_dict(state)
    model = m.eval().to(device)

# =====================
# Prediction function
# =====================
@torch.no_grad()
def predict_pil(img: Image.Image):
    x = preproc(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    i = int(torch.argmax(logits, dim=1))
    label = CLASS_NAMES[i]
    road_type, condition = label.split("_", 1) if "_" in label else (label, "unknown")
    return {
        "label": label,
        "type": road_type,
        "condition": condition,
        "confidence": float(probs[i]),
        "probs": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }

# =====================
# Load YOLOv8
# =====================
yolo = YOLO("models/yolov8n.pt")  # trained damage detection model

# =====================
# Risk index function
# =====================
def calculate_risk(detections):
    risk = 0
    for d in detections:
        label = d['class']
        if label.lower() == "pothole":
            risk += 5
        elif label.lower() == "crack":
            risk += 3
        else:
            risk += 1
    return risk

# =====================
# API: Road Surface Classification
# =====================
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    res = predict_pil(image)
    return res

# =====================
# API: Road Damage Detection + Risk Index
# =====================
@app.post("/risk")
async def risk(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    results = yolo.predict(image, conf=0.4)[0]  # YOLOv8 prediction
    detections = []
    for r in results.boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        detections.append({
            "class": yolo.names[cls_id],
            "confidence": conf
        })
    
    risk_score = calculate_risk(detections)
    return {"detections": detections, "risk_index": risk_score}
