"""
robustness_benchmark.py – KhetRak Level 4 Robustness Tester
─────────────────────────────────────────────────────────────────────────────
Applies synthetic real-world corruptions to the test set and measures how
much the model's accuracy drops — the core Level 4 claim.

Corruptions tested (matching real farmer-photo conditions):
  - Gaussian blur (out-of-focus shots)
  - Motion blur (camera shake)
  - JPEG compression artefacts (low-end phone cameras)
  - Low brightness (shade / indoor / dusk)
  - High brightness / overexposure (direct sunlight)
  - Shadow overlay (partial shadow from plant canopy)
  - Gaussian noise (noisy sensor)
  - Saturation drop (washed-out colours)
  - Multiple-leaf clutter simulation (tile image 2×2)
  - Soil / dirt occlusion (random brown patches)

Usage:
    python robustness_benchmark.py \
        --model-path ./checkpoints/best_model.keras \
        --data-dir   ./data/processed \
        --output-dir ./robustness_results
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ─── Corruption functions ─────────────────────────────────────────────────────

def gaussian_blur(img, severity=3):
    radius = [0, 1, 2, 3, 4, 5][severity]
    return img.filter(ImageFilter.GaussianBlur(radius))

def motion_blur(img, severity=3):
    from PIL import ImageFilter
    size = [0, 3, 5, 8, 12, 16][severity]
    kernel = [0] * (size * size)
    for i in range(size):
        kernel[i * size + i] = 1
    kernel = [k / size for k in kernel]
    return img.filter(ImageFilter.Kernel((size, size), kernel, scale=1))

def jpeg_compression(img, severity=3):
    from io import BytesIO
    quality = [100, 60, 40, 25, 15, 8][severity]
    buf = BytesIO(); img.save(buf, 'JPEG', quality=quality); buf.seek(0)
    return Image.open(buf).copy()

def low_brightness(img, severity=3):
    factor = [1.0, 0.75, 0.55, 0.38, 0.22, 0.12][severity]
    return ImageEnhance.Brightness(img).enhance(factor)

def high_brightness(img, severity=3):
    factor = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0][severity]
    return ImageEnhance.Brightness(img).enhance(min(factor, 3.0))

def shadow_overlay(img, severity=3):
    """Simulate shadows from canopy by darkening a random band."""
    arr = np.array(img).copy().astype(np.float32)
    h, w = arr.shape[:2]
    alpha = [0, 0.2, 0.35, 0.5, 0.65, 0.8][severity]
    # Random diagonal band
    x1 = random.randint(0, w // 2); x2 = random.randint(w // 2, w)
    arr[:, x1:x2] *= (1 - alpha)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def gaussian_noise(img, severity=3):
    arr = np.array(img).astype(np.float32)
    std = [0, 8, 15, 25, 38, 55][severity]
    noise = np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def saturation_drop(img, severity=3):
    factor = [1.0, 0.75, 0.55, 0.35, 0.15, 0.0][severity]
    return ImageEnhance.Color(img).enhance(factor)

def clutter_simulation(img, severity=3):
    """Tile the image to simulate multiple leaves in frame."""
    if severity == 0: return img
    w, h = img.size
    scale = [1.0, 0.8, 0.65, 0.5, 0.4, 0.3][severity]
    small_w, small_h = int(w * scale), int(h * scale)
    small = img.resize((small_w, small_h))
    canvas = Image.new('RGB', (w, h), (30, 80, 30))  # green background
    for y in range(0, h, small_h):
        for x in range(0, w, small_w):
            canvas.paste(small, (x, y))
    return canvas

def soil_occlusion(img, severity=3):
    """Add random brown patches to simulate soil/dirt on lens."""
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    n_patches = [0, 1, 2, 4, 6, 9][severity]
    for _ in range(n_patches):
        px = random.randint(0, w - 30)
        py = random.randint(0, h - 30)
        pw = random.randint(15, 40)
        ph = random.randint(15, 40)
        arr[py:py+ph, px:px+pw] = [random.randint(80,130), random.randint(50,90), random.randint(20,50)]
    return Image.fromarray(arr)

CORRUPTIONS = {
    'Gaussian Blur':      gaussian_blur,
    'Motion Blur':        motion_blur,
    'JPEG Compression':   jpeg_compression,
    'Low Brightness':     low_brightness,
    'High Brightness':    high_brightness,
    'Shadow Overlay':     shadow_overlay,
    'Gaussian Noise':     gaussian_noise,
    'Saturation Drop':    saturation_drop,
    'Leaf Clutter':       clutter_simulation,
    'Soil Occlusion':     soil_occlusion,
}

# ─── Benchmark runner ─────────────────────────────────────────────────────────
def run_benchmark(model, test_images, class_map, img_size, output_dir):
    results = {}
    clean_acc = evaluate_clean(model, test_images, class_map, img_size)
    print(f"\n  Clean accuracy: {clean_acc:.1f}%")
    results['Clean (no corruption)'] = clean_acc

    for name, fn in CORRUPTIONS.items():
        acc = evaluate_corrupted(model, test_images, class_map, img_size, fn, severity=3)
        drop = clean_acc - acc
        print(f"  {name:<25}: {acc:5.1f}%  (drop: {drop:+.1f}%)")
        results[name] = acc

    # Plot
    plot_results(results, output_dir)
    save_report(results, clean_acc, output_dir)
    return results


def predict(model, img, img_size):
    arr = np.array(img.resize((img_size, img_size))).astype(np.float32) / 255.0
    return np.argmax(model.predict(arr[np.newaxis], verbose=0)[0])


def evaluate_clean(model, test_images, class_map, img_size):
    correct = 0
    for path, true_idx in test_images:
        try:
            img = Image.open(path).convert('RGB')
            pred = predict(model, img, img_size)
            if pred == true_idx: correct += 1
        except: pass
    return 100 * correct / max(len(test_images), 1)


def evaluate_corrupted(model, test_images, class_map, img_size, fn, severity):
    correct = 0
    for path, true_idx in test_images:
        try:
            img = Image.open(path).convert('RGB')
            img = fn(img, severity)
            pred = predict(model, img, img_size)
            if pred == true_idx: correct += 1
        except: pass
    return 100 * correct / max(len(test_images), 1)


def plot_results(results, output_dir):
    labels = list(results.keys())
    values = list(results.values())
    colors = ['#22c55e' if l == 'Clean (no corruption)' else
              '#22c55e' if v >= 80 else '#f59e0b' if v >= 60 else '#ef4444'
              for l, v in zip(labels, values)]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor='none')
    ax.set_xlim(0, 105)
    ax.set_xlabel('Accuracy (%)', color='white')
    ax.set_title('KhetRak – Robustness to Real-World Corruptions', color='white', pad=14)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#2d2d2d')
    ax.set_facecolor('#111827'); fig.set_facecolor('#060c14')

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', color='white', fontsize=9)

    ax.axvline(80, color='rgba(34,197,94,0.4)', linestyle='--', linewidth=1)
    ax.text(80.5, -0.6, '80% target', color='#6b7280', fontsize=8)

    plt.tight_layout()
    out = Path(output_dir) / 'robustness_benchmark.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f'\n  Chart saved → {out}')
    plt.close()


def save_report(results, clean_acc, output_dir):
    report = {
        'clean_accuracy': clean_acc,
        'per_corruption': results,
        'mean_corruption_accuracy': np.mean([v for k, v in results.items() if k != 'Clean (no corruption)']),
        'worst_corruption': min(((k, v) for k, v in results.items() if k != 'Clean (no corruption)'), key=lambda x: x[1]),
    }
    out = Path(output_dir) / 'robustness_report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  Report saved → {out}')
    print(f'\n  Mean accuracy under corruption: {report["mean_corruption_accuracy"]:.1f}%')
    print(f'  Worst corruption: {report["worst_corruption"][0]} ({report["worst_corruption"][1]:.1f}%)')


def load_test_images(data_dir):
    """Load test image paths and class indices from processed data dir."""
    import csv
    manifest = Path(data_dir) / 'manifest.csv'
    images = []
    with open(manifest) as f:
        for row in csv.DictReader(f):
            if row['split'] == 'test':
                images.append((Path(data_dir) / row['path'], int(row['class_idx'])))
    random.shuffle(images)
    return images[:200]  # benchmark on up to 200 test samples


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KhetRak robustness benchmark')
    parser.add_argument('--model-path',  default='./checkpoints/best_model.keras')
    parser.add_argument('--data-dir',    default='./data/processed')
    parser.add_argument('--output-dir',  default='./robustness_results')
    parser.add_argument('--img-size',    type=int, default=224)
    args = parser.parse_args()

    print('🛡️  KhetRak – Level 4 Robustness Benchmark')
    print('='*50)
    print(f'Model: {args.model_path}')
    print(f'Data:  {args.data_dir}')

    model = tf.keras.models.load_model(args.model_path)
    class_map_path = Path(args.data_dir) / 'class_map.json'
    with open(class_map_path) as f:
        class_map = json.load(f)

    test_images = load_test_images(args.data_dir)
    print(f'\nRunning {len(CORRUPTIONS)} corruption types on {len(test_images)} test images…\n')
    run_benchmark(model, test_images, class_map, args.img_size, args.output_dir)
