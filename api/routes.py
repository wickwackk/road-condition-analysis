# api/routes.py
import io, os, json
from datetime import datetime, timezone
from typing import List
from flask import Blueprint, request, jsonify
from PIL import Image

# ML
import torch
import torch.nn as nn
from torchvision import models, transforms

# local utils
from utils.geopose_utils import (
    ypr_deg_to_quat, quat_to_ypr_deg,
    geopose_basic_ypr, geopose_basic_quat
)

api_bp = Blueprint("api", __name__)

# ---------- Model setup ----------
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "road_condition_resnet50.pth"))
CLASSES_TXT = os.getenv("CLASSES_TXT", "classes.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def load_classes() -> List[str]:
    if os.path.isfile(CLASSES_TXT):
        with open(CLASSES_TXT, "r", encoding="utf-8") as f:
            xs = [ln.strip() for ln in f if ln.strip()]
            if xs:
                return xs
    return ["asphalt", "paved", "unpaved"]

classes = load_classes()

model = None
try:
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    state = state.get("state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    m.load_state_dict(state, strict=False)
    m.to(DEVICE).eval()
    model = m
except Exception as e:
    print("[WARN] Model not loaded:", e)

# ---------- Routes ----------
@api_bp.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "classes": classes,
        "model_loaded": bool(model)
    })

@api_bp.post("/api/geopose/convert")
def geopose_convert():
    data = request.get_json(force=True, silent=True) or {}
    out = (data.get("output") or "both").lower()
    try:
        lat = float(data["lat"]); lon = float(data["lon"]); h = float(data.get("h", 0.0))
    except Exception:
        return jsonify({"error": "Provide lat, lon (and optional h)"}), 400

    res = {}
    if all(k in data for k in ("yaw", "pitch", "roll")):
        yaw, pitch, roll = float(data["yaw"]), float(data["pitch"]), float(data["roll"])
        if out in ("ypr", "both"):
            res["basicYPR"] = geopose_basic_ypr(lat, lon, h, yaw, pitch, roll)
        if out in ("quat", "both"):
            qx, qy, qz, qw = ypr_deg_to_quat(yaw, pitch, roll)
            res["basicQuaternion"] = geopose_basic_quat(lat, lon, h, qx, qy, qz, qw)
        return jsonify(res)

    if all(k in data for k in ("qx", "qy", "qz", "qw")):
        qx, qy, qz, qw = float(data["qx"]), float(data["qy"]), float(data["qz"]), float(data["qw"])
        if out in ("quat", "both"):
            res["basicQuaternion"] = geopose_basic_quat(lat, lon, h, qx, qy, qz, qw)
        if out in ("ypr", "both"):
            yaw, pitch, roll = quat_to_ypr_deg(qx, qy, qz, qw)
            res["basicYPR"] = geopose_basic_ypr(lat, lon, h, yaw, pitch, roll)
        return jsonify(res)

    return jsonify({"error": "Provide yaw,pitch,roll OR qx,qy,qz,qw"}), 400

@api_bp.post("/api/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Put .pth at models/ and restart."}), 503
    if "image" not in request.files:
        return jsonify({"error": "Send multipart/form-data with field 'image'"}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
    top_idx = int(torch.tensor(probs).argmax().item())

    # optional geo + orientation
    lat = request.form.get("lat", type=float)
    lon = request.form.get("lon", type=float)
    h   = request.form.get("h", type=float, default=0.0)
    yaw = request.form.get("yaw", type=float)
    pitch = request.form.get("pitch", type=float)
    roll  = request.form.get("roll", type=float)

    geoPose = None
    if lat is not None and lon is not None and yaw is not None and pitch is not None and roll is not None:
        qx, qy, qz, qw = ypr_deg_to_quat(yaw, pitch, roll)
        geoPose = {
            "basicYPR": geopose_basic_ypr(lat, lon, h, yaw, pitch, roll),
            "basicQuaternion": geopose_basic_quat(lat, lon, h, qx, qy, qz, qw),
        }

    now_iso = datetime.now(timezone.utc).isoformat()
    return jsonify({
        "timestamp": now_iso,
        "label": classes[top_idx],
        "prob": float(probs[top_idx]),
        "probs": {classes[i]: float(p) for i, p in enumerate(probs)},
        "image_id": file.filename,
        "geoPose": geoPose
    })
