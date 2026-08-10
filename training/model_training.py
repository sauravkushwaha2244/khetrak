"""
model_training.py – KhetRak MobileNetV2 Training Script
─────────────────────────────────────────────────────────────────────────────
Trains a MobileNetV2 (width=0.75) classifier on the prepared dataset.
Performs two-phase training:
  Phase 1: Train only the new classification head (backbone frozen)
  Phase 2: Fine-tune top layers of backbone at lower LR

Usage:
    python model_training.py [--data-dir ./data/processed] [--config training_config.yaml]
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ─── Config ──────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Dataset ─────────────────────────────────────────────────────────────────
def build_dataset(data_dir: Path, split: str, cfg: dict) -> tf.data.Dataset:
    """Load images from <data_dir>/<split>/ using keras image_dataset_from_directory."""
    img_size   = cfg["model"]["input_size"]
    batch_size = cfg["training"]["batch_size"]
    split_dir  = data_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    ds = keras.utils.image_dataset_from_directory(
        str(split_dir),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=(split == "train"),
        seed=42,
    )

    # Normalize [0,255] → [0,1] inside the pipeline
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─── Augmentation layer (TF-native, runs on GPU) ─────────────────────────────
def build_augmentation_layer(cfg: dict):
    aug_cfg = cfg["augmentation"]
    return keras.Sequential([
        layers.RandomFlip("horizontal") if aug_cfg.get("random_flip") else layers.Lambda(lambda x: x),
        layers.RandomRotation(aug_cfg["random_rotation"] / 360),
        layers.RandomZoom(aug_cfg["random_zoom"]),
        layers.RandomBrightness(aug_cfg["random_brightness"]),
        layers.RandomContrast(aug_cfg["random_contrast"]),
    ], name="augmentation")


# ─── Model ────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, cfg: dict) -> keras.Model:
    img_size = cfg["model"]["input_size"]
    alpha    = cfg["model"]["alpha"]
    dropout  = cfg["model"]["dropout"]

    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Freeze for Phase 1

    inputs   = keras.Input(shape=(img_size, img_size, 3))
    aug_out  = build_augmentation_layer(cfg)(inputs, training=True)
    x        = base(aug_out, training=False)
    x        = layers.GlobalAveragePooling2D()(x)
    x        = layers.BatchNormalization()(x)
    x        = layers.Dropout(dropout)(x)
    x        = layers.Dense(128, activation="relu")(x)
    x        = layers.Dropout(dropout * 0.7)(x)
    outputs  = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return keras.Model(inputs, outputs, name="khetrak_mobilenetv2")


def unfreeze_top_layers(model: keras.Model, n_layers: int, lr: float):
    """Unfreeze top n_layers of the backbone for fine-tuning."""
    base = model.get_layer("mobilenetv2_0.75_224")
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"  Unfroze top {n_layers} backbone layers (LR={lr})")


# ─── Training ────────────────────────────────────────────────────────────────
def train(args):
    cfg      = load_config(args.config)
    data_dir = Path(args.data_dir)
    t_cfg    = cfg["training"]
    out_cfg  = cfg["output"]

    # Load class map
    class_map_path = data_dir / "class_map.json"
    with open(class_map_path) as f:
        class_map = json.load(f)
    num_classes = len(class_map["classes"])
    print(f"\n🌾 KhetRak Training | {num_classes} classes")
    print("="*50)

    # Datasets
    train_ds = build_dataset(data_dir, "train", cfg)
    val_ds   = build_dataset(data_dir, "val",   cfg)

    # Model
    model = build_model(num_classes, cfg)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=t_cfg["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    ckpt_dir = Path(out_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = Path(out_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    cb_list = [
        callbacks.ModelCheckpoint(
            str(ckpt_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            patience=t_cfg["early_stopping"]["patience"],
            monitor=t_cfg["early_stopping"]["monitor"],
            restore_best_weights=t_cfg["early_stopping"]["restore_best"],
            verbose=1,
        ),
        callbacks.TensorBoard(log_dir=str(log_dir)),
        callbacks.CSVLogger(str(log_dir / "training_log.csv")),
    ]

    # ── Phase 1: Train head only ───────────────────────────────────────────
    phase1_epochs = t_cfg.get("finetune_at_epoch", 30)
    print(f"\n📌 Phase 1: Training classification head ({phase1_epochs} epochs)…")
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=cb_list,
    )

    # ── Phase 2: Fine-tune top backbone layers ─────────────────────────────
    total_epochs  = t_cfg["epochs"]
    ft_layers     = t_cfg.get("finetune_layers", 50)
    ft_lr         = t_cfg.get("finetune_lr", 1e-4)

    if total_epochs > phase1_epochs:
        print(f"\n🔧 Phase 2: Fine-tuning top {ft_layers} backbone layers…")
        unfreeze_top_layers(model, ft_layers, ft_lr)
        hist2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=total_epochs,
            initial_epoch=phase1_epochs,
            callbacks=cb_list,
        )
    else:
        hist2 = None

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("\n📊 Evaluating on validation set…")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"   Val accuracy: {val_acc*100:.1f}%  |  Val loss: {val_loss:.4f}")

    # Classification report
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))

    print("\n" + classification_report(y_true, y_pred, target_names=class_map["classes"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                xticklabels=[c.split("_")[-1] for c in class_map["classes"]],
                yticklabels=[c.split("_")[-1] for c in class_map["classes"]],
                cmap="Greens")
    ax.set_title("Confusion Matrix – KhetRak")
    plt.tight_layout()
    plt.savefig(str(log_dir / "confusion_matrix.png"), dpi=150)
    print(f"   Confusion matrix saved → {log_dir / 'confusion_matrix.png'}")

    # Save final model
    final_path = str(ckpt_dir / "final_model.keras")
    model.save(final_path)
    print(f"\n✅ Training complete. Model saved → {final_path}")
    print("   Next step: run convert_model.py to export TFLite")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KhetRak crop disease classifier")
    parser.add_argument("--data-dir", default="./data/processed",     help="Processed dataset directory")
    parser.add_argument("--config",   default="./training_config.yaml", help="Training config YAML")
    train(parser.parse_args())
