"""
convert_model.py – KhetRak TFLite + TF.js Conversion Script
─────────────────────────────────────────────────────────────────────────────
Converts the trained Keras model to:
  1. TensorFlow Lite (integer-quantised, ≤5 MB) → ../model.tflite
  2. TensorFlow.js graph model → ../public/tfjs_model/

Usage:
    python convert_model.py [--model-path ./checkpoints/best_model.keras]
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import glob
import shutil
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml


def load_config():
    config_path = Path(__file__).parent / "training_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── Representative dataset for quantisation calibration ─────────────────────
def representative_dataset_gen(data_dir: Path, img_size: int, n_samples: int = 100):
    """Yield random validation images for INT8 calibration."""
    import random
    from PIL import Image

    images = (
        list((data_dir / "val").rglob("*.jpg")) +
        list((data_dir / "val").rglob("*.jpeg")) +
        list((data_dir / "val").rglob("*.png"))
    )
    random.shuffle(images)

    for img_path in images[:n_samples]:
        try:
            img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
            arr = np.array(img, dtype=np.float32) / 255.0
            yield [arr[np.newaxis, ...]]
        except Exception:
            continue


# ─── TFLite conversion ───────────────────────────────────────────────────────
def convert_to_tflite(model_path: str, output_path: str, cfg: dict, data_dir: Path):
    img_size = cfg["model"]["input_size"]

    print("📦 Loading Keras model…")
    model = tf.keras.models.load_model(model_path)
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # ── Full integer quantisation (INT8) ──────────────────────────────────
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(data_dir, img_size)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,    # fallback for unsupported ops
    ]
    # Keep input/output as float32 for easier JS integration
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    print("\n⚙️  Converting to TFLite (INT8 quantisation)…")
    tflite_model = converter.convert()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✅ TFLite model saved → {out_path}")
    print(f"   Model size: {size_mb:.2f} MB")

    if size_mb > 5:
        print("⚠️  Model exceeds 5 MB target. Consider reducing alpha or input size.")

    return str(out_path)


# ─── TF.js conversion ────────────────────────────────────────────────────────
def convert_to_tfjs(model_path: str, output_dir: str):
    """Convert to TF.js graph model format for browser use."""
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("\n[WARN] tensorflowjs not installed. Skipping TF.js export.")
        print("  Install: pip install tensorflowjs")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n⚙️  Converting to TF.js graph model…")
    model = tf.keras.models.load_model(model_path)
    tfjs.converters.save_keras_model(model, str(out_dir))

    # Check shard sizes
    shards = list(out_dir.glob("group1-shard*"))
    total_size_mb = sum(s.stat().st_size for s in shards) / (1024 * 1024)
    print(f"✅ TF.js model saved → {out_dir}")
    print(f"   Total weight shards: {len(shards)}, size: {total_size_mb:.2f} MB")


# ─── Benchmark ───────────────────────────────────────────────────────────────
def benchmark_tflite(tflite_path: str, img_size: int, n_runs: int = 50):
    """Measure average inference latency using TFLite interpreter."""
    import time

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)

    # Warmup
    for _ in range(5):
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    p95_ms = np.percentile(times, 95)
    print(f"\n⚡ Inference benchmark ({n_runs} runs):")
    print(f"   Average latency: {avg_ms:.1f} ms")
    print(f"   P95 latency:     {p95_ms:.1f} ms")

    if avg_ms > 2000:
        print("⚠️  Average latency >2s – may be slow on low-end devices.")
    else:
        print("✅ Latency is within the 2s target for low-end devices.")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KhetRak model to TFLite + TF.js")
    parser.add_argument("--model-path",  default="./checkpoints/best_model.keras", help="Trained Keras model path")
    parser.add_argument("--data-dir",    default="./data/processed",               help="Processed data dir (for calibration)")
    parser.add_argument("--tflite-out",  default="../model.tflite",                help="TFLite output path")
    parser.add_argument("--tfjs-out",    default="../public/tfjs_model",           help="TF.js output dir")
    parser.add_argument("--no-tfjs",     action="store_true",                     help="Skip TF.js export")
    parser.add_argument("--benchmark",   action="store_true",                     help="Run latency benchmark after conversion")
    args = parser.parse_args()

    cfg = load_config()

    print("🌾 KhetRak – Model Conversion Pipeline")
    print("="*50)

    tflite_path = convert_to_tflite(
        args.model_path, args.tflite_out, cfg, Path(args.data_dir)
    )

    if not args.no_tfjs:
        convert_to_tfjs(args.model_path, args.tfjs_out)

    if args.benchmark:
        benchmark_tflite(tflite_path, cfg["model"]["input_size"])

    print("\n🎉 All done!")
    print(f"   TFLite → {args.tflite_out}")
    if not args.no_tfjs:
        print(f"   TF.js  → {args.tfjs_out}")
    print("\nDeploy by running: npm run dev")
