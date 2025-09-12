# utils/dataset_prepare.py
# Scans images, infers labels (from subfolders or labels.csv), extracts EXIF GPS/time,
# stratified train/val split, writes:
#   - data/manifest.csv
#   - data/train/<label>/* and data/val/<label>/* (copied)
#   - exports/trainingdml.json  (minimal TrainingDML-AI 1.0)
import argparse, csv, os, shutil, sys, uuid, random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from PIL import Image, ExifTags

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

@dataclass
class Item:
    id: str
    rel_path: str
    label: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    h: Optional[float] = None
    timestamp: Optional[str] = None

def is_image(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMAGE_EXTS

def list_images(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if is_image(p):
                files.append(os.path.normpath(p))
    return files

def load_labels_from_csv(root: str, csv_path: str) -> Dict[str, str]:
    """
    CSV columns: path,label  (path relative to images root)
    """
    mapping: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if "path" not in r.fieldnames or "label" not in r.fieldnames:
            raise ValueError("labels.csv must have columns: path,label")
        for row in r:
            rel = os.path.normpath(row["path"])
            mapping[rel] = row["label"].strip()
    return mapping

def infer_label_from_parent(rel_path: str) -> str:
    # label = immediate parent folder name
    parts = os.path.normpath(rel_path).split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"

def exif_to_latlon_h_ts(img_path: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    try:
        img = Image.open(img_path)
        exif = img._getexif()
        if not exif:
            return None, None, None, None
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        gps = tags.get("GPSInfo")
        ts = tags.get("DateTimeOriginal") or tags.get("DateTime")
        lat = lon = h = None
        if gps:
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
            def to_deg(value):
                # value like ((deg_num,deg_den), (min_num, min_den), (sec_num,sec_den))
                d = value[0][0] / value[0][1]
                m = value[1][0] / value[1][1]
                s = value[2][0] / value[2][1]
                return d + (m/60.0) + (s/3600.0)
            if "GPSLatitude" in gps and "GPSLatitudeRef" in gps:
                lat = to_deg(gps["GPSLatitude"])
                if gps["GPSLatitudeRef"] not in ("N", b"N"):
                    lat = -lat
            if "GPSLongitude" in gps and "GPSLongitudeRef" in gps:
                lon = to_deg(gps["GPSLongitude"])
                if gps["GPSLongitudeRef"] not in ("E", b"E"):
                    lon = -lon
            if "GPSAltitude" in gps:
                alt = gps["GPSAltitude"]
                h = alt[0] / alt[1] if isinstance(alt, tuple) else float(alt)
        # Convert EXIF date/time "YYYY:MM:DD HH:MM:SS" to ISO-ish "YYYY-MM-DDTHH:MM:SS"
        if ts and isinstance(ts, str) and ":" in ts:
            try:
                y, m, d = ts.split(" ")[0].split(":")
                hh, mm, ss = ts.split(" ")[1].split(":")
                ts_iso = f"{y}-{m}-{d}T{hh}:{mm}:{ss}Z"
            except Exception:
                ts_iso = ts
        else:
            ts_iso = None
        return lat, lon, h, ts_iso
    except Exception:
        return None, None, None, None

def stratified_split(items: List[Item], train_ratio: float, seed: int = 42) -> Tuple[List[Item], List[Item]]:
    random.seed(seed)
    by_label: Dict[str, List[Item]] = {}
    for it in items:
        by_label.setdefault(it.label, []).append(it)
    train, val = [], []
    for label, arr in by_label.items():
        random.shuffle(arr)
        k = max(1, int(len(arr) * train_ratio)) if len(arr) > 1 else len(arr)
        train.extend(arr[:k])
        val.extend(arr[k:])
    return train, val

def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def copy_items(items: List[Item], images_root: str, out_split_root: str):
    """
    Copies images into out_split_root/<label>/<id>.<ext>
    Returns list of new relative paths.
    """
    new_paths: Dict[str, str] = {}
    for it in items:
        src = os.path.join(images_root, it.rel_path)
        ext = os.path.splitext(src)[1].lower()
        dst_dir = os.path.join(out_split_root, it.label)
        ensure_dirs(dst_dir)
        dst = os.path.join(dst_dir, f"{it.id}{ext}")
        shutil.copy2(src, dst)
        new_paths[it.id] = os.path.relpath(dst, start=os.path.dirname(out_split_root))
    return new_paths

def write_manifest_csv(path: str, items: List[Item]):
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","rel_path","label","lat","lon","h","timestamp"])
        for it in items:
            w.writerow([it.id, it.rel_path, it.label, it.lat, it.lon, it.h, it.timestamp])

def build_trainingdml(dataset_name: str, items: List[Item]) -> dict:
    return {
        "type": "AI_TrainingDataset",
        "name": dataset_name,
        "version": "1.0",
        "geospatial": {"crs": "EPSG:4326"},
        "task": {"type": "scene-classification",
                 "classes": [{"name": c} for c in sorted({it.label for it in items})]},
        "data": [
            {
                "type": "AI_TrainingData",
                "id": it.id,
                "asset": {"uri": it.rel_path.replace("\\", "/"), "mediaType": "image/jpeg"},
                "position": {"lat": it.lat, "lon": it.lon, "h": it.h} if it.lat is not None and it.lon is not None else None,
                "timestamp": it.timestamp,
                "labels": [{"type": "classification", "class": it.label}],
            }
            for it in items
        ],
    }

def save_json(path: str, obj: dict):
    import json
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Prepare dataset (manifest CSV, train/val split, TrainingDML-AI JSON).")
    ap.add_argument("--images-root", required=True, help="Root folder containing images (and optionally class subfolders).")
    ap.add_argument("--labels-csv", help="Optional CSV with columns: path,label (paths relative to images-root).")
    ap.add_argument("--out-root", default="data", help="Output root (default: data)")
    ap.add_argument("--dataset-name", default="Road Condition Training")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images_root = os.path.normpath(args.images_root)
    all_paths = list_images(images_root)
    if not all_paths:
        print(f"[ERR] No images under {images_root}", file=sys.stderr)
        sys.exit(1)

    # Make rel paths
    rel_paths = [os.path.relpath(p, start=images_root) for p in all_paths]

    # Labels
    mapping = load_labels_from_csv(images_root, args.labels_csv) if args.labels_csv else {}
    items: List[Item] = []
    for rel in rel_paths:
        label = mapping.get(rel, infer_label_from_parent(rel))
        pid = uuid.uuid4().hex[:12]
        abs_path = os.path.join(images_root, rel)
        lat, lon, h, ts = exif_to_latlon_h_ts(abs_path)
        items.append(Item(id=pid, rel_path=rel.replace("\\", "/"), label=label, lat=lat, lon=lon, h=h, timestamp=ts))

    # Write manifest of all items
    manifest_path = os.path.join(args.out_root, "manifest.csv")
    write_manifest_csv(manifest_path, items)
    print(f"[OK] Wrote manifest: {manifest_path} ({len(items)} rows)")

    # Split + copy into train/val
    train, val = stratified_split(items, args.train_ratio, seed=args.seed)
    split_root = os.path.join(args.out_root)  # we'll create train/ and val/ inside out_root
    train_map = copy_items(train, images_root, os.path.join(split_root, "train"))
    val_map   = copy_items(val, images_root, os.path.join(split_root, "val"))

    # Update rel_path to new locations for TDML export (use paths relative to out_root)
    for it in train:
        it.rel_path = os.path.join("train", it.label, f"{it.id}{os.path.splitext(it.rel_path)[1].lower()}").replace("\\", "/")
    for it in val:
        it.rel_path = os.path.join("val", it.label, f"{it.id}{os.path.splitext(it.rel_path)[1].lower()}").replace("\\", "/")

    tdml = build_trainingdml(args.dataset_name, items)
    tdml_path = os.path.join("exports", "trainingdml.json")
    save_json(tdml_path, tdml)
    print(f"[OK] Wrote TrainingDML-AI JSON: {tdml_path}")

if __name__ == "__main__":
    main()
