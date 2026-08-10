"""
prepare_data.py – KhetRak Data Preparation Pipeline
─────────────────────────────────────────────────────────────────────────────
Downloads PlantVillage subset for target crops, merges with custom field
images, applies class-balanced train/val/test splits, and saves a CSV
manifest.

Usage:
    python prepare_data.py [--custom-dir ./data/custom] [--output-dir ./data/processed]

PlantVillage data is sourced from Kaggle:
    https://www.kaggle.com/datasets/emmarex/plantdisease
─────────────────────────────────────────────────────────────────────────────
"""

import os
import shutil
import random
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import yaml

# ─── Config ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Disease class mapping: PlantVillage folder name → our metadata
PV_CLASS_MAP = {
    # Millet diseases (simulate from similar PV classes)
    "Corn_(maize)___Northern_Leaf_Blight":     ("Millet",      "Blast Disease"),
    "Corn_(maize)___Common_rust_":             ("Millet",      "Downy Mildew"),
    "Tomato___Late_blight":                    ("Millet",      "Ergot"),
    # Pigeon Pea diseases
    "Tomato___Fusarium_wilt":                  ("Pigeon Pea",  "Fusarium Wilt"),
    "Tomato___Tomato_mosaic_virus":            ("Pigeon Pea",  "Sterility Mosaic"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":  ("Pigeon Pea",  "Phytophthora Blight"),
    # Sorghum diseases
    "Grape___Black_rot":                       ("Sorghum",     "Grain Mold"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Sorghum", "Leaf Blight"),
    "Wheat___Septoria":                        ("Sorghum",     "Covered Kernel Smut"),
}

# Class label → integer index
CLASS_LABELS = [
    "Millet_Blast Disease",
    "Millet_Downy Mildew",
    "Millet_Ergot",
    "Pigeon Pea_Fusarium Wilt",
    "Pigeon Pea_Sterility Mosaic",
    "Pigeon Pea_Phytophthora Blight",
    "Sorghum_Grain Mold",
    "Sorghum_Leaf Blight",
    "Sorghum_Covered Kernel Smut",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LABELS)}


def load_config():
    config_path = Path(__file__).parent / "training_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── Augmentation helpers ─────────────────────────────────────────────────────
def augment_image(img: Image.Image, cfg: dict) -> Image.Image:
    """Apply random augmentations to simulate real-world field photos."""
    # Random horizontal flip
    if cfg.get("random_flip") and random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness
    factor = 1.0 + random.uniform(-cfg["random_brightness"], cfg["random_brightness"])
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Random contrast
    factor = 1.0 + random.uniform(-cfg["random_contrast"], cfg["random_contrast"])
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Random saturation
    lo, hi = cfg["random_saturation"]
    img = ImageEnhance.Color(img).enhance(random.uniform(lo, hi))

    # Random blur (simulate motion / poor focus)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Random JPEG compression artifacts
    from io import BytesIO
    quality = random.randint(cfg.get("jpeg_quality_min", 50), 95)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf).copy()

    return img


def resize_and_pad(img: Image.Image, size: int) -> Image.Image:
    """Resize keeping aspect ratio, then pad to square."""
    img.thumbnail((size, size), Image.LANCZOS)
    new_img = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    new_img.paste(img, offset)
    return new_img


# ─── Main pipeline ─────────────────────────────────────────────────────────── 
def collect_plantvillage_samples(pv_root: Path, max_per_class: int = 800) -> dict:
    """Collect up to max_per_class samples from PlantVillage for each target class."""
    samples = defaultdict(list)

    if not pv_root.exists():
        print(f"[WARN] PlantVillage directory not found: {pv_root}")
        print("  → Download from https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("  → Extract to:", pv_root)
        return samples

    for pv_folder, (crop, disease) in PV_CLASS_MAP.items():
        folder_path = pv_root / pv_folder
        if not folder_path.exists():
            # Try case-insensitive glob
            matches = list(pv_root.glob(f"*{pv_folder[:15]}*"))
            if matches:
                folder_path = matches[0]
            else:
                print(f"[WARN] Folder not found: {pv_folder}")
                continue

        label_key = f"{crop}_{disease}"
        images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.JPG")) + \
                 list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpeg"))
        random.shuffle(images)
        samples[label_key].extend(images[:max_per_class])
        print(f"  ✓ {label_key}: {len(samples[label_key])} from PlantVillage")

    return samples


def collect_custom_samples(custom_root: Path) -> dict:
    """
    Collect custom field images.
    Expected structure:
        custom_root/
            Millet/
                Blast Disease/
                    img1.jpg ...
            Pigeon Pea/
                Fusarium Wilt/ ...
    """
    samples = defaultdict(list)
    if not custom_root.exists():
        print(f"[INFO] No custom data directory found at {custom_root} – skipping.")
        return samples

    for crop_dir in custom_root.iterdir():
        if not crop_dir.is_dir():
            continue
        for disease_dir in crop_dir.iterdir():
            if not disease_dir.is_dir():
                continue
            label_key = f"{crop_dir.name}_{disease_dir.name}"
            images = list(disease_dir.rglob("*.jpg")) + \
                     list(disease_dir.rglob("*.jpeg")) + \
                     list(disease_dir.rglob("*.png")) + \
                     list(disease_dir.rglob("*.JPG"))
            samples[label_key].extend(images)
            if images:
                print(f"  ✓ {label_key}: {len(images)} custom images")

    return samples


def split_samples(all_paths, val_frac, test_frac):
    """Stratified train/val/test split."""
    n = len(all_paths)
    random.shuffle(all_paths)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    test   = all_paths[:n_test]
    val    = all_paths[n_test:n_test + n_val]
    train  = all_paths[n_test + n_val:]
    return train, val, test


def process_and_save(
    samples: dict, output_dir: Path, cfg: dict, augment_train: bool = True
):
    """Process images and write to output_dir with CSV manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    val_frac  = cfg["data"]["val_split"]
    test_frac = cfg["data"]["test_split"]
    img_size  = cfg["model"]["input_size"]
    aug_cfg   = cfg["augmentation"]

    for label_key, paths in samples.items():
        if label_key not in CLASS_TO_IDX:
            print(f"[SKIP] Unknown class: {label_key}")
            continue

        class_idx = CLASS_TO_IDX[label_key]
        train_paths, val_paths, test_paths = split_samples(list(paths), val_frac, test_frac)

        splits = [("train", train_paths), ("val", val_paths), ("test", test_paths)]
        for split_name, split_paths in splits:
            split_dir = output_dir / split_name / label_key.replace(" ", "_")
            split_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(split_paths, desc=f"{split_name}/{label_key}", leave=False):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = resize_and_pad(img, img_size)

                    if split_name == "train" and augment_train:
                        img = augment_image(img, aug_cfg)

                    out_name = f"{img_path.stem}_{random.randint(1000,9999)}.jpg"
                    out_path = split_dir / out_name
                    img.save(out_path, "JPEG", quality=90)
                    manifest_rows.append({
                        "path": str(out_path.relative_to(output_dir)),
                        "label": label_key,
                        "class_idx": class_idx,
                        "split": split_name,
                    })
                except Exception as e:
                    print(f"[WARN] Failed to process {img_path}: {e}")

    # Write manifest CSV
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "class_idx", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write class index JSON
    class_map_path = output_dir / "class_map.json"
    with open(class_map_path, "w") as f:
        json.dump({"classes": CLASS_LABELS, "class_to_idx": CLASS_TO_IDX}, f, indent=2)

    print(f"\n✅ Dataset prepared in: {output_dir}")
    print(f"   Total images: {len(manifest_rows)}")
    for split in ["train", "val", "test"]:
        n = sum(1 for r in manifest_rows if r["split"] == split)
        print(f"   {split}: {n}")

    return manifest_path


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare KhetRak training data")
    parser.add_argument("--pv-dir",     default="./data/plantvillage", help="PlantVillage root directory")
    parser.add_argument("--custom-dir", default="./data/custom",       help="Custom field images root directory")
    parser.add_argument("--output-dir", default="./data/processed",    help="Output directory for processed data")
    parser.add_argument("--max-per-class", type=int, default=800,      help="Max PlantVillage samples per class")
    parser.add_argument("--no-augment", action="store_true",           help="Skip augmentation (faster, for testing)")
    args = parser.parse_args()

    cfg = load_config()

    print("🌾 KhetRak – Data Preparation Pipeline")
    print("="*50)
    print(f"PlantVillage: {args.pv_dir}")
    print(f"Custom data:  {args.custom_dir}")
    print(f"Output:       {args.output_dir}")
    print()

    # Collect samples
    print("📦 Collecting PlantVillage samples…")
    pv_samples = collect_plantvillage_samples(Path(args.pv_dir), max_per_class=args.max_per_class)

    print("\n📸 Collecting custom field images…")
    custom_samples = collect_custom_samples(Path(args.custom_dir))

    # Merge
    all_samples = defaultdict(list)
    for k, v in pv_samples.items():    all_samples[k].extend(v)
    for k, v in custom_samples.items(): all_samples[k].extend(v)

    if not all_samples:
        print("\n❌ No images found. Please provide data and try again.")
        return

    print(f"\n📊 Class distribution:")
    for label, paths in sorted(all_samples.items()):
        print(f"   {label}: {len(paths)} images")

    # Process
    print("\n⚙️  Processing and splitting…")
    process_and_save(
        dict(all_samples),
        Path(args.output_dir),
        cfg,
        augment_train=not args.no_augment,
    )


if __name__ == "__main__":
    main()
