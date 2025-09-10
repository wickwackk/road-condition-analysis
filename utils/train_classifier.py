# utils/train_classifier.py
import argparse, os, time, math, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn: deterministic off by default; turning it on can slow down training
    torch.backends.cudnn.benchmark = True

def build_dataloaders(train_dir: str, val_dir: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, List[str], List[int]]:
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(root=val_dir,   transform=val_tfms)

    classes = train_ds.classes
    # class counts from train set
    counts = [0] * len(classes)
    for _, y in train_ds.samples:
        counts[y] += 1

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, classes, counts

def build_model(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    if pretrained:
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        m = models.resnet50(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    if freeze_backbone:
        for name, p in m.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    return m

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * y.size(0)
        preds = logits.argmax(1)
        correct += int((preds == y).sum().item())
        total += y.size(0)
    return (loss_sum / max(1, total)), (correct / max(1, total))

def train_one_epoch(model: nn.Module, loader: DataLoader, opt, scaler, device: torch.device, ce, epoch: int):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
            loss = ce(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        running_loss += float(loss.item()) * y.size(0)
        running_correct += int((logits.argmax(1) == y).sum().item())
        total += y.size(0)

    train_loss = running_loss / max(1, total)
    train_acc  = running_correct / max(1, total)
    print(f"epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f}")
    return train_loss, train_acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/train")
    ap.add_argument("--val_dir",   default="data/val")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained weights")
    ap.add_argument("--freeze_backbone", action="store_true", help="freeze all except final FC layer")
    ap.add_argument("--num_workers", type=int, default=0, help="Windows users: keep 0")
    ap.add_argument("--out_model", default="models/road_condition_resnet50.pth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes, class_counts = build_dataloaders(
        args.train_dir, args.val_dir, args.batch_size, args.num_workers
    )
    print(f"classes: {classes}  counts: {class_counts}")

    # class-weighted CE for imbalance
    tot = sum(class_counts) if sum(class_counts) > 0 else 1
    weights = torch.tensor([tot / max(1, c) for c in class_counts], dtype=torch.float32, device=device)
    weights = weights / weights.mean()  # normalize
    ce = nn.CrossEntropyLoss(weight=weights)

    model = build_model(num_classes=len(classes), pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)
    model.to(device)

    # only trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_acc = 0.0
    best_path = Path(args.out_model)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    # save classes.txt in project root for your API
    with open("classes.txt", "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, opt, scaler, device, ce, epoch)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} |   val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "img_size": IMG_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "args": vars(args),
            }, str(best_path))
            print(f"[checkpoint] saved best to {best_path} (val_acc={best_val_acc:.4f})")

    print(f"done. best val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
