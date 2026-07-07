"""
CNN Classifier for Handwritten Medicine Name Recognition
=========================================================

Fine-tunes an EfficientNet-B0 on the Kaggle "Doctor's Handwritten
Prescription BD Dataset" for 78-class medicine name classification.

Training strategy:
  Phase 1 (warmup):  Freeze backbone, train classifier head only (5 epochs, LR=1e-3)
  Phase 2 (finetune): Unfreeze all layers, train end-to-end (30 epochs, LR=1e-4, cosine annealing)
  Early stopping with patience=10 on validation accuracy.

Outputs:
  checkpoints/efficientnet_b0_best.pth  — model weights + label mapping + metadata
  logs/classifier_training_history.csv  — per-epoch metrics
  logs/classifier_training_curves.png   — loss & accuracy plots

Usage:
    python train_classifier.py
    python train_classifier.py --epochs-warmup 5 --epochs-finetune 30 --batch-size 32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Fix Unicode on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add script directory to path for local imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import (
    PrescriptionDataset,
    PadToSquare,
    find_dataset_dir,
    get_split_paths,
    build_label_map,
    build_generic_map,
)

# ─── Defaults ──────────────────────────────────────────────────────────────────

DATA_DIR = SCRIPT_DIR.parent / "data" / "raw" / "prescriptions"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
LOG_DIR = SCRIPT_DIR / "logs"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = IMAGE_SIZE):
    """Training transforms with augmentation for handwriting robustness."""
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=5,
        ),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE):
    """Validation/test transforms — deterministic, no augmentation."""
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Model ─────────────────────────────────────────────────────────────────────

def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create EfficientNet-B0 with a custom classifier head.
    
    Uses higher dropout (0.3) than default (0.2) to reduce
    overfitting on our small dataset (40 samples/class).
    """
    if pretrained:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        model = efficientnet_b0(weights=None)

    # Replace classifier head
    num_features = model.classifier[1].in_features  # 1280 for B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all layers for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ─── Training & Evaluation ─────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model. Returns (avg_loss, accuracy, all_preds, all_labels, all_probs)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Compute top-k accuracy from probability arrays."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(1 for i, label in enumerate(labels) if label in top_k_preds[i])
    return correct / len(labels)


# ─── Logging & Visualization ──────────────────────────────────────────────────

def save_training_curves(history: list, save_path: str):
    """Plot and save training curves (loss + accuracy)."""
    df = pd.DataFrame(history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(df["epoch"], df["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(df["epoch"], df["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(df["epoch"], df["train_acc"], "b-o", markersize=3, label="Train Acc")
    ax2.plot(df["epoch"], df["val_acc"], "r-o", markersize=3, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Mark phase boundary
    if len(df) > 0 and "phase" in df.columns:
        phase1_end = df[df["phase"] == "warmup"]["epoch"].max()
        if not np.isnan(phase1_end):
            for ax in (ax1, ax2):
                ax.axvline(x=phase1_end + 0.5, color="gray", linestyle="--",
                           alpha=0.5, label="Unfreeze backbone")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Training curves saved to {save_path}")


# ─── Main Training Pipeline ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 classifier on handwritten prescription dataset.",
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Path to the prescriptions data directory")
    parser.add_argument("--epochs-warmup", type=int, default=5,
                        help="Epochs for Phase 1 (frozen backbone)")
    parser.add_argument("--epochs-finetune", type=int, default=30,
                        help="Max epochs for Phase 2 (full fine-tune)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr-warmup", type=float, default=1e-3,
                        help="Learning rate for warmup phase")
    parser.add_argument("--lr-finetune", type=float, default=1e-4,
                        help="Initial learning rate for fine-tune phase")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 65}")
    print(f"  ClariRx — Prescription Classifier Training (EfficientNet-B0)")
    print(f"{'=' * 65}")
    print(f"  Device          : {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" (⚠ No GPU detected — training will be slow)")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Warmup epochs   : {args.epochs_warmup} (LR={args.lr_warmup})")
    print(f"  Finetune epochs : {args.epochs_finetune} (LR={args.lr_finetune})")
    print(f"  Label smoothing : {args.label_smoothing}")
    print(f"  Early stopping  : patience={args.patience}")
    print(f"{'=' * 65}\n")

    # Create output directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Dataset ───────────────────────────────────────────────────────
    print("📂 Loading dataset...")
    dataset_dir = find_dataset_dir(args.data_dir)
    print(f"  Found dataset: {os.path.basename(dataset_dir)}")

    train_csv, train_img_dir = get_split_paths(dataset_dir, "Training")
    val_csv, val_img_dir = get_split_paths(dataset_dir, "Validation")
    test_csv, test_img_dir = get_split_paths(dataset_dir, "Testing")

    # Build label mapping from training set
    label_to_idx, idx_to_label = build_label_map(train_csv)
    medicine_to_generic = build_generic_map(train_csv)
    num_classes = len(label_to_idx)
    print(f"  Classes: {num_classes} medicine names")

    # Create datasets
    train_dataset = PrescriptionDataset(
        train_csv, train_img_dir, label_to_idx,
        transform=get_train_transforms(),
    )
    val_dataset = PrescriptionDataset(
        val_csv, val_img_dir, label_to_idx,
        transform=get_val_transforms(),
    )
    test_dataset = PrescriptionDataset(
        test_csv, test_img_dir, label_to_idx,
        transform=get_val_transforms(),
    )

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Create Model ───────────────────────────────────────────────────────
    print("\n🏗️  Creating EfficientNet-B0 model...")
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── Training History ───────────────────────────────────────────────────
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    global_epoch = 0

    # ── Phase 1: Warmup (frozen backbone) ──────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  Phase 1: Warmup — Frozen backbone, training classifier head")
    print(f"{'─' * 65}")

    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,} / {total_params:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_warmup,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.epochs_warmup + 1):
        global_epoch += 1
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": global_epoch, "phase": "warmup",
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc, "lr": lr,
        })

        print(
            f"  Epoch {global_epoch:3d} │ "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} │ "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} │ "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = global_epoch

    # ── Phase 2: Full Fine-Tuning ──────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  Phase 2: Full Fine-Tuning — All layers unfrozen")
    print(f"{'─' * 65}")

    unfreeze_all(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,} / {total_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr_finetune,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_finetune, eta_min=1e-6,
    )

    for epoch in range(1, args.epochs_finetune + 1):
        global_epoch += 1
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": global_epoch, "phase": "finetune",
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc, "lr": lr,
        })

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = global_epoch
            epochs_no_improve = 0
            improved = " ✓ BEST"

            # Save best checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "medicine_to_generic": medicine_to_generic,
                "num_classes": num_classes,
                "best_val_accuracy": best_val_acc,
                "epoch": best_epoch,
                "model_name": "efficientnet_b0",
                "image_size": IMAGE_SIZE,
            }
            ckpt_path = CHECKPOINT_DIR / "efficientnet_b0_best.pth"
            torch.save(checkpoint, ckpt_path)
        else:
            epochs_no_improve += 1

        print(
            f"  Epoch {global_epoch:3d} │ "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} │ "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} │ "
            f"LR={lr:.2e} │ {elapsed:.1f}s{improved}"
        )

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\n  ⏹  Early stopping triggered (no improvement for {args.patience} epochs)")
            break

    # ── Save Training History ──────────────────────────────────────────────
    history_path = LOG_DIR / "classifier_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\n  📄 Training history saved to {history_path}")

    curves_path = LOG_DIR / "classifier_training_curves.png"
    save_training_curves(history, str(curves_path))

    # ── Final Test Evaluation ──────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  Final Evaluation on Test Set")
    print(f"{'─' * 65}")

    # Load best checkpoint
    ckpt_path = CHECKPOINT_DIR / "efficientnet_b0_best.pth"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded best model from epoch {checkpoint['epoch']} "
              f"(val_acc={checkpoint['best_val_accuracy']:.4f})")
    else:
        print("  ⚠ No checkpoint found, using current model weights")

    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device,
    )
    top5_acc = compute_topk_accuracy(test_probs, test_labels, k=5)

    print(f"\n  🎯 Test Results:")
    print(f"     Top-1 Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"     Top-5 Accuracy : {top5_acc:.4f} ({top5_acc * 100:.2f}%)")
    print(f"     Test Loss      : {test_loss:.4f}")

    # Per-class accuracy summary
    from sklearn.metrics import classification_report
    class_names = [idx_to_label[i] for i in range(num_classes)]
    report = classification_report(
        test_labels, test_preds,
        target_names=class_names,
        output_dict=True,
    )
    report_text = classification_report(
        test_labels, test_preds,
        target_names=class_names,
    )

    # Save classification report
    report_path = LOG_DIR / "classifier_test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EfficientNet-B0 Classification Report\n")
        f.write(f"{'=' * 65}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Test Top-1 Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Top-5 Accuracy: {top5_acc:.4f}\n\n")
        f.write(report_text)
    print(f"\n  📄 Classification report saved to {report_path}")

    # Worst performing classes
    per_class_acc = {}
    for i in range(num_classes):
        mask = test_labels == i
        if mask.sum() > 0:
            per_class_acc[idx_to_label[i]] = (test_preds[mask] == i).mean()

    worst_classes = sorted(per_class_acc.items(), key=lambda x: x[1])[:5]
    if worst_classes:
        print(f"\n  ⚠ Worst performing classes:")
        for name, acc in worst_classes:
            generic = medicine_to_generic.get(name, "?")
            print(f"     {name:20s} ({generic:30s}): {acc:.2%}")

    # ── Final Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  ✅ Training Complete!")
    print(f"{'=' * 65}")
    print(f"  Best epoch        : {best_epoch}")
    print(f"  Best val accuracy : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"  Test top-1 acc    : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Test top-5 acc    : {top5_acc:.4f} ({top5_acc * 100:.2f}%)")
    print(f"  Model saved to    : {CHECKPOINT_DIR / 'efficientnet_b0_best.pth'}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
