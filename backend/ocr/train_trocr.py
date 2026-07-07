"""
TrOCR Fine-Tuning for Handwritten Medicine Name Recognition
============================================================

Fine-tunes microsoft/trocr-base-handwritten on the Kaggle dataset.
Because this is an OCR model (image-to-text sequence), it predicts
the actual characters rather than a fixed set of classes.

Optimized for 4GB VRAM (RTX 3050):
  - batch_size: 4
  - gradient_accumulation_steps: 4 (effective batch size 16)
  - fp16 mixed precision
  - frozen encoder (only fine-tune the decoder)

Outputs:
  checkpoints/trocr_best/  — HuggingFace model weights
  logs/trocr_training_history.csv
"""

import argparse
import os
import sys
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# Fix Unicode on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import PrescriptionOCRDataset, find_dataset_dir, get_split_paths

# ─── Defaults ──────────────────────────────────────────────────────────────────

DATA_DIR = SCRIPT_DIR.parent / "data" / "raw" / "prescriptions"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints" / "trocr_best"
LOG_DIR = SCRIPT_DIR / "logs"


def freeze_encoder(model):
    """Freeze the Vision (ViT) encoder to save VRAM and prevent overfitting."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    return model


def compute_metrics(pred, processor, cer_metric):
    """Compute Character Error Rate (CER) and exact match accuracy."""
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Replace -100 (padding) with pad_token_id so we can decode
    pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
    labels_ids = np.where(labels_ids != -100, labels_ids, processor.tokenizer.pad_token_id)

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=labels_str)
    
    # Exact match accuracy
    correct = sum([1 for p, l in zip(pred_str, labels_str) if p.strip().lower() == l.strip().lower()])
    acc = correct / len(labels_str)

    return {"cer": cer, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'=' * 65}")
    print(f"  ClariRx — TrOCR Fine-Tuning")
    print(f"{'=' * 65}")

    # ── Processor & Model ──────────────────────────────────────────────────
    print("🧠 Loading TrOCR processor and model...")
    from transformers import RobertaTokenizer, ViTImageProcessor
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
    image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Optimization for 4GB VRAM
    model = freeze_encoder(model)
    
    # Set special tokens used for generating text
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # Ensure vocabulary size matches
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set beam search parameters correctly in generation_config
    model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.sep_token_id
    model.generation_config.max_length = 32
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    # ── Dataset ────────────────────────────────────────────────────────────
    print("📂 Loading datasets...")
    dataset_dir = find_dataset_dir(args.data_dir)
    train_csv, train_img_dir = get_split_paths(dataset_dir, "Training")
    val_csv, val_img_dir = get_split_paths(dataset_dir, "Validation")

    train_dataset = PrescriptionOCRDataset(train_csv, train_img_dir, processor)
    val_dataset = PrescriptionOCRDataset(val_csv, val_img_dir, processor)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Metric ─────────────────────────────────────────────────────────────
    cer_metric = evaluate.load("cer")

    # ── Training Args ──────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        fp16=True, 
        output_dir=str(CHECKPOINT_DIR),
        logging_dir=str(LOG_DIR),
        logging_steps=10,
        save_total_limit=1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        seed=args.seed,
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor.image_processor, # using image_processor avoids deprecation warning
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, processor, cer_metric),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    print("\n🚀 Starting Training...")
    # Check if a checkpoint exists
    import os
    checkpoint = None
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint = str(CHECKPOINT_DIR / checkpoints[-1])
            print(f"  Resuming from checkpoint: {checkpoint}")
    
    trainer.train(resume_from_checkpoint=checkpoint)

    print("\n✅ Training Complete. Saving best model...")
    trainer.save_model(str(CHECKPOINT_DIR))
    processor.save_pretrained(str(CHECKPOINT_DIR))

    # Save log history
    history = trainer.state.log_history
    pd.DataFrame(history).to_csv(LOG_DIR / "trocr_training_history.csv", index=False)
    print(f"  Model saved to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()
