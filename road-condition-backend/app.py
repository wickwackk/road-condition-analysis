from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import torch
from torchvision import models, transforms
import io
import torch.nn as nn

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

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Load ResNet50
# =====================
resnet = models.resnet50(pretrained=False)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 4)  # adjust number of classes
resnet.load_state_dict(torch.load(r"D:\previous\semester 6\grad proj\project1\road-condition-analysis\models\road_resnet50.pth", map_location=device))
resnet.eval().to(device)
class_names = ["asphalt", "paved", "unpaved"]  # adjust to your dataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

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
    img_t = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = resnet(img_t)
        _, preds = torch.max(outputs, 1)
    
    return {"surface": class_names[preds.item()]}

# =====================
# API: Road Damage Detection + Risk Index
# =====================
@app.post("/risk")
async def risk(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    results = yolo.predict(image, conf=0.4)  # YOLOv8 prediction
    detections = []
    for r in results[0].boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        detections.append({
            "class": yolo.names[cls_id],
            "confidence": conf
        })
    
    risk_score = calculate_risk(detections)
    return {"detections": detections, "risk_index": risk_score}
