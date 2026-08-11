"""
BioBERT NER Fine-Tuning for Prescription Entity Extraction
============================================================

Fine-tunes dmis-lab/biobert-v1.1 for token-level NER on prescription text.
Entity types: DRUG, DOSAGE, FREQUENCY, DURATION, INSTRUCTION

Since real labeled NER data is scarce, this script includes a built-in
synthetic prescription text generator that creates BIO-tagged training
samples from common Indian prescription patterns.

Optimized for limited hardware:
  - Gradient accumulation for effective larger batch sizes
  - fp16 mixed precision when GPU is available
  - Early stopping with patience on validation F1

Outputs:
  checkpoints/biobert_ner_best/  — HuggingFace model + tokenizer + label map
  logs/ner_training_history.csv  — per-epoch metrics
  logs/ner_classification_report.txt  — final test classification report

Usage:
    python train.py
    python train.py --epochs 10 --batch-size 16 --lr 3e-5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ─── Path Setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = SCRIPT_DIR.parent
BACKEND_DIR = EXTRACTION_DIR.parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints" / "biobert_ner_best"
LOG_DIR = EXTRACTION_DIR / "logs"


# ─── NER Label Scheme (BIO tagging) ───────────────────────────────────────────

ENTITY_TYPES = ["DRUG", "DOSAGE", "FREQUENCY", "DURATION", "INSTRUCTION"]

def build_label_list() -> List[str]:
    """Build the full BIO label list: O + B-/I- for each entity type."""
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    return labels

LABEL_LIST = build_label_list()
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


# ─── Synthetic Training Data Generator ────────────────────────────────────────

# Common Indian prescription drug names (brand + generic)
DRUG_NAMES = [
    "Amoxicillin 500mg", "Amoxicillin 250mg", "Azithromycin 500mg",
    "Paracetamol 650mg", "Paracetamol 500mg", "Ibuprofen 400mg",
    "Cetirizine 10mg", "Omeprazole 20mg", "Pantoprazole 40mg",
    "Metformin 500mg", "Metformin 850mg", "Amlodipine 5mg",
    "Atorvastatin 10mg", "Atorvastatin 20mg", "Losartan 50mg",
    "Clopidogrel 75mg", "Aspirin 75mg", "Ranitidine 150mg",
    "Montelukast 10mg", "Levocetrizine 5mg", "Doxycycline 100mg",
    "Ciprofloxacin 500mg", "Metronidazole 400mg", "Fluconazole 150mg",
    "Aceclofenac 100mg", "Diclofenac 50mg", "Tramadol 50mg",
    "Gabapentin 300mg", "Pregabalin 75mg", "Telmisartan 40mg",
    "Olmesartan 20mg", "Rosuvastatin 10mg", "Cefixime 200mg",
    "Levofloxacin 500mg", "Ofloxacin 200mg", "Domperidone 10mg",
    "Ondansetron 4mg", "Rabeprazole 20mg", "Esomeprazole 40mg",
    "Cephalexin 500mg", "Clindamycin 300mg", "Sertraline 50mg",
    "Escitalopram 10mg", "Alprazolam 0.5mg", "Clonazepam 0.5mg",
    "Baclofen 10mg", "Chlorpheniramine 4mg", "Fexofenadine 120mg",
    # Common OCR-corrupted versions
    "Am0xicillin 500mg", "Paracetam0l 650mg", "Cetirizne 10mg",
    "Omeprazol 20mg", "Metf0rmin 500mg", "Azithromycn 500mg",
]

DRUG_PREFIXES = ["Tab", "Cap", "Syr", "Inj", "Tab.", "Cap.", "Syr.", "Drops"]

FREQUENCIES = [
    "1-0-1", "1-1-1", "0-0-1", "1-0-0", "0-1-0",
    "1-0-1-1", "SOS", "BD", "TDS", "OD", "HS",
    "twice daily", "once daily", "thrice daily",
    "morning and night", "at bedtime",
]

DURATIONS = [
    "3 days", "5 days", "7 days", "10 days", "14 days",
    "1 week", "2 weeks", "1 month", "3 months",
    "x 3d", "x 5d", "x 7d", "x 10d", "x 14d",
    "for 5 days", "for 7 days", "for 2 weeks",
    "As needed", "Continue",
]

INSTRUCTIONS = [
    "after food", "before food", "with food", "after meals",
    "before meals", "on empty stomach", "with warm water",
    "with milk", "at bedtime", "in the morning",
    "after breakfast", "after dinner",
]


def _generate_single_prescription_line() -> Tuple[List[str], List[str]]:
    """
    Generate a single prescription line with BIO tags.

    Returns:
        (tokens, bio_tags) — aligned lists of word tokens and their BIO tags.
    """
    tokens = []
    tags = []

    # Optionally add a line number prefix (e.g., "1.", "2)")
    if random.random() < 0.5:
        prefix = f"{random.randint(1, 10)}" + random.choice([".", ")", ":"])
        tokens.append(prefix)
        tags.append("O")

    # Optionally add drug form prefix (Tab, Cap, etc.)
    if random.random() < 0.7:
        prefix = random.choice(DRUG_PREFIXES)
        tokens.append(prefix)
        tags.append("O")

    # Drug name (may be multi-word, e.g., "Amoxicillin 500mg")
    drug = random.choice(DRUG_NAMES)
    drug_tokens = drug.split()
    for i, tok in enumerate(drug_tokens):
        tokens.append(tok)
        tags.append("B-DRUG" if i == 0 else "I-DRUG")

    # Frequency (optional)
    if random.random() < 0.85:
        freq = random.choice(FREQUENCIES)
        freq_tokens = freq.split()
        for i, tok in enumerate(freq_tokens):
            tokens.append(tok)
            tags.append("B-FREQUENCY" if i == 0 else "I-FREQUENCY")

    # Instructions (optional)
    if random.random() < 0.6:
        instr = random.choice(INSTRUCTIONS)
        instr_tokens = instr.split()
        for i, tok in enumerate(instr_tokens):
            tokens.append(tok)
            tags.append("B-INSTRUCTION" if i == 0 else "I-INSTRUCTION")

    # Duration (optional)
    if random.random() < 0.7:
        dur = random.choice(DURATIONS)
        dur_tokens = dur.split()
        for i, tok in enumerate(dur_tokens):
            tokens.append(tok)
            tags.append("B-DURATION" if i == 0 else "I-DURATION")

    return tokens, tags


def generate_synthetic_dataset(
    n_samples: int = 500,
    min_lines: int = 1,
    max_lines: int = 5,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """
    Generate a synthetic BIO-tagged prescription NER dataset.

    Each sample is a multi-line prescription with 1-5 medication lines.

    Args:
        n_samples: Number of prescription samples to generate.
        min_lines: Minimum medication lines per prescription.
        max_lines: Maximum medication lines per prescription.
        seed: Random seed for reproducibility.

    Returns:
        List of (tokens, tags) tuples.
    """
    random.seed(seed)
    dataset = []

    for _ in range(n_samples):
        all_tokens = []
        all_tags = []
        n_lines = random.randint(min_lines, max_lines)

        # Optionally add a header
        if random.random() < 0.4:
            header_options = [
                ["Rx"], ["Rx", ":"], ["Prescription"],
                ["Patient", ":", "Mr.", "Sharma"],
                ["Dr.", "Kumar", "-", "Prescription"],
            ]
            header = random.choice(header_options)
            all_tokens.extend(header)
            all_tags.extend(["O"] * len(header))

        for _ in range(n_lines):
            line_tokens, line_tags = _generate_single_prescription_line()
            all_tokens.extend(line_tokens)
            all_tags.extend(line_tags)

        dataset.append((all_tokens, all_tags))

    return dataset


# ─── HuggingFace Dataset Preparation ──────────────────────────────────────────

def prepare_hf_dataset(dataset: List[Tuple[List[str], List[str]]]):
    """Convert synthetic data to a HuggingFace Dataset with train/val/test splits."""
    from datasets import Dataset, DatasetDict

    records = []
    for idx, (tokens, tags) in enumerate(dataset):
        tag_ids = [LABEL_TO_ID[t] for t in tags]
        records.append({
            "id": idx,
            "tokens": tokens,
            "ner_tags": tag_ids,
        })

    full_dataset = Dataset.from_list(records)

    # 70/15/15 split
    train_test = full_dataset.train_test_split(test_size=0.3, seed=42)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=42)

    return DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })


def tokenize_and_align_labels(examples, tokenizer, max_length: int = 128):
    """
    Tokenize inputs and align BIO labels to wordpiece tokens.

    Uses the word_ids() method to propagate labels from word-level
    to sub-word token level with proper B-/I- handling.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        is_split_into_words=True,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First sub-word of a new word → use original label
                label_ids.append(labels[word_idx])
            else:
                # Continuation sub-word → convert B- to I-, keep I- and O
                original_label = labels[word_idx]
                label_name = ID_TO_LABEL[original_label]
                if label_name.startswith("B-"):
                    # Convert B- to I- for sub-word continuation
                    i_label = "I-" + label_name[2:]
                    label_ids.append(LABEL_TO_ID[i_label])
                else:
                    label_ids.append(original_label)

            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


# ─── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute seqeval NER metrics (P/R/F1) for the HuggingFace Trainer."""
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    # Convert IDs back to label strings, ignoring padding (-100)
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_clean = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_seq.append(ID_TO_LABEL[label_id])
            pred_seq_clean.append(ID_TO_LABEL[pred_id])

        true_labels.append(true_seq)
        pred_labels.append(pred_seq_clean)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    """Main training function."""
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    print("=" * 60)
    print("BioBERT NER Training for Prescription Entity Extraction")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Generate / load synthetic dataset ───────────────────────────────────
    print(f"\nGenerating {args.n_samples} synthetic prescription samples...")
    raw_dataset = generate_synthetic_dataset(
        n_samples=args.n_samples,
        seed=args.seed,
    )
    hf_dataset = prepare_hf_dataset(raw_dataset)

    print(f"  Train: {len(hf_dataset['train'])} samples")
    print(f"  Val:   {len(hf_dataset['validation'])} samples")
    print(f"  Test:  {len(hf_dataset['test'])} samples")

    # ── Load tokenizer and model ────────────────────────────────────────────
    model_name = args.model_name
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Tokenize dataset ────────────────────────────────────────────────────
    print("\nTokenizing dataset...")
    tokenized_dataset = hf_dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=hf_dataset["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # ── Training arguments ──────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=(device == "cuda"),
        logging_steps=50,
        report_to="none",  # Disable wandb / tensorboard
        seed=args.seed,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # ── Train ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Starting training...")
    print("─" * 60)

    train_result = trainer.train()

    # ── Save best model ─────────────────────────────────────────────────────
    print(f"\nSaving best model to {CHECKPOINT_DIR}...")
    trainer.save_model(str(CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))

    # Save label mapping alongside the model
    label_map_path = CHECKPOINT_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({
            "label_list": LABEL_LIST,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
        }, f, indent=2)

    # ── Evaluate on test set ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Evaluating on test set...")
    print("─" * 60)

    test_results = trainer.evaluate(tokenized_dataset["test"])

    print(f"\nTest Results:")
    print(f"  Precision: {test_results['eval_precision']:.4f}")
    print(f"  Recall:    {test_results['eval_recall']:.4f}")
    print(f"  F1-Score:  {test_results['eval_f1']:.4f}")

    # ── Generate detailed classification report ─────────────────────────────
    _save_detailed_report(trainer, tokenized_dataset["test"])

    # ── Save training history ───────────────────────────────────────────────
    _save_training_history(trainer)

    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {CHECKPOINT_DIR}")
    print(f"   Logs saved to:  {LOG_DIR}")

    return test_results


def _save_detailed_report(trainer, test_dataset):
    """Generate and save a detailed seqeval classification report."""
    from seqeval.metrics import classification_report as seqeval_report

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(preds, labels):
        true_seq = []
        pred_seq_clean = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_seq.append(ID_TO_LABEL[label_id])
            pred_seq_clean.append(ID_TO_LABEL[pred_id])

        true_labels.append(true_seq)
        pred_labels.append(pred_seq_clean)

    report = seqeval_report(true_labels, pred_labels, digits=4)

    report_path = LOG_DIR / "ner_classification_report.txt"
    with open(report_path, "w") as f:
        f.write("BioBERT NER Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    print(f"\nClassification report saved to: {report_path}")
    print(report)


def _save_training_history(trainer):
    """Save per-epoch training metrics to CSV."""
    history_path = LOG_DIR / "ner_training_history.csv"

    log_history = trainer.state.log_history
    epochs_data = []

    for entry in log_history:
        if "eval_f1" in entry:
            epochs_data.append({
                "epoch": entry.get("epoch", 0),
                "eval_loss": entry.get("eval_loss", 0),
                "eval_precision": entry.get("eval_precision", 0),
                "eval_recall": entry.get("eval_recall", 0),
                "eval_f1": entry.get("eval_f1", 0),
            })

    if epochs_data:
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=epochs_data[0].keys())
            writer.writeheader()
            writer.writerows(epochs_data)

        print(f"Training history saved to: {history_path}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BioBERT for prescription NER"
    )
    parser.add_argument(
        "--model-name", type=str, default="dmis-lab/biobert-v1.1",
        help="HuggingFace model name or local path (default: dmis-lab/biobert-v1.1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Training batch size per device (default: 16)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=2,
        help="Gradient accumulation steps (default: 2, effective batch=32)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-5,
        help="Learning rate (default: 3e-5)"
    )
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="Max token sequence length (default: 128)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="Number of synthetic training samples to generate (default: 500)"
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience in epochs (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
