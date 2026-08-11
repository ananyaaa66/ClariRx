"""
BioBERT NER Inference for Prescription Entity Extraction
==========================================================

Loads a fine-tuned BioBERT NER checkpoint and performs token-level
entity recognition on raw OCR text, then groups BIO tags back into
structured PrescriptionItem objects.

Usage (standalone test):
    python predict.py

Usage (as module):
    from extraction.biobert_ner.predict import extract_with_biobert
    result = extract_with_biobert(raw_text, DocType.PRESCRIPTION)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ─── Path Setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints" / "biobert_ner_best"


# ─── Model Loader (singleton-ish) ─────────────────────────────────────────────

_cached_model = None
_cached_tokenizer = None
_cached_label_map = None


def _load_model():
    """Load the fine-tuned BioBERT NER model, tokenizer, and label map."""
    global _cached_model, _cached_tokenizer, _cached_label_map

    if _cached_model is not None:
        return _cached_model, _cached_tokenizer, _cached_label_map

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"BioBERT NER checkpoint not found at {CHECKPOINT_DIR}. "
            f"Run 'python train.py' first to fine-tune the model."
        )

    logger.info(f"Loading BioBERT NER model from {CHECKPOINT_DIR}...")

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load label map
    label_map_path = CHECKPOINT_DIR / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_data = json.load(f)
            label_map = {int(k): v for k, v in label_data["id_to_label"].items()}
    else:
        # Fallback: use model config
        label_map = model.config.id2label

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_label_map = label_map

    logger.info(f"Model loaded on {device}. Labels: {len(label_map)}")
    return model, tokenizer, label_map


# ─── Token-Level NER Prediction ───────────────────────────────────────────────

def predict_entities(
    text: str,
    confidence_threshold: float = 0.3,
) -> List[Dict]:
    """
    Run token-level NER on a raw text string.

    Args:
        text: Raw OCR text from a prescription.
        confidence_threshold: Minimum softmax probability to accept an entity tag.

    Returns:
        List of entity dicts: {"text": str, "label": str, "confidence": float}
    """
    model, tokenizer, label_map = _load_model()
    device = next(model.parameters()).device

    # Tokenize
    words = text.split()
    inputs = tokenizer(
        words,
        truncation=True,
        padding=True,
        max_length=128,
        is_split_into_words=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert to probabilities and predictions
    probs = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    max_probs = probs.squeeze().cpu().numpy().max(axis=-1)

    # Align back to original words
    word_ids = inputs.get("word_ids", None)
    if word_ids is None:
        # Manually reconstruct word_ids from tokenizer
        encoding = tokenizer(
            words,
            truncation=True,
            padding=True,
            max_length=128,
            is_split_into_words=True,
        )
        word_ids_list = encoding.word_ids()
    else:
        word_ids_list = word_ids.squeeze().cpu().numpy().tolist()

    # Use tokenizer's word_ids method
    encoding = tokenizer(
        words,
        truncation=True,
        padding=True,
        max_length=128,
        is_split_into_words=True,
    )
    word_ids_list = encoding.word_ids()

    # Group predictions by word
    word_preds = {}  # word_idx -> (label, confidence)
    for token_idx, word_idx in enumerate(word_ids_list):
        if word_idx is None:
            continue
        if word_idx not in word_preds:
            label = label_map.get(int(pred_ids[token_idx]), "O")
            conf = float(max_probs[token_idx])
            word_preds[word_idx] = (label, conf)

    # Build entity list
    entities = []
    for word_idx in sorted(word_preds.keys()):
        if word_idx >= len(words):
            break
        label, conf = word_preds[word_idx]
        if label != "O" and conf >= confidence_threshold:
            entities.append({
                "text": words[word_idx],
                "label": label,
                "confidence": round(conf, 4),
                "word_idx": word_idx,
            })

    return entities


# ─── Entity Grouping (BIO → Structured Items) ─────────────────────────────────

def group_entities(entities: List[Dict]) -> List[Dict]:
    """
    Group consecutive BIO-tagged entities into structured spans.

    E.g., [B-DRUG "Amoxicillin", I-DRUG "500mg"] → {"DRUG": "Amoxicillin 500mg"}

    Returns a list of entity groups, each a dict of {entity_type: text}.
    """
    if not entities:
        return []

    groups = []
    current_group = {}
    current_type = None
    current_text_parts = []
    current_confidences = []

    for ent in entities:
        label = ent["label"]
        text = ent["text"]
        conf = ent["confidence"]

        if label.startswith("B-"):
            # Save previous entity span
            if current_type:
                current_group[current_type] = " ".join(current_text_parts)
                current_group[f"{current_type}_confidence"] = (
                    sum(current_confidences) / len(current_confidences)
                )

            entity_type = label[2:]  # Remove "B-"
            current_type = entity_type
            current_text_parts = [text]
            current_confidences = [conf]

        elif label.startswith("I-"):
            entity_type = label[2:]
            if entity_type == current_type:
                # Continue current span
                current_text_parts.append(text)
                current_confidences.append(conf)
            else:
                # Mismatched I- tag; save current and start new
                if current_type:
                    current_group[current_type] = " ".join(current_text_parts)
                    current_group[f"{current_type}_confidence"] = (
                        sum(current_confidences) / len(current_confidences)
                    )
                current_type = entity_type
                current_text_parts = [text]
                current_confidences = [conf]

        else:
            # O tag — flush current entity span
            if current_type:
                current_group[current_type] = " ".join(current_text_parts)
                current_group[f"{current_type}_confidence"] = (
                    sum(current_confidences) / len(current_confidences)
                )
                current_type = None
                current_text_parts = []
                current_confidences = []

    # Flush final entity
    if current_type:
        current_group[current_type] = " ".join(current_text_parts)
        current_group[f"{current_type}_confidence"] = (
            sum(current_confidences) / len(current_confidences)
        )

    # If we have accumulated entities for a prescription line, save the group
    if current_group:
        groups.append(current_group)

    # Post-process: split groups by DRUG boundaries
    # (each DRUG entity signals a new prescription line)
    refined_groups = _split_by_drug(entities)

    return refined_groups if refined_groups else ([current_group] if current_group else [])


def _split_by_drug(entities: List[Dict]) -> List[Dict]:
    """Split entity list into groups, with each B-DRUG starting a new group."""
    groups = []
    current = {}
    current_confs = {}

    for ent in entities:
        label = ent["label"]
        text = ent["text"]
        conf = ent["confidence"]

        if label == "B-DRUG":
            # Save previous group if it has a DRUG
            if "DRUG" in current:
                for key in current_confs:
                    current[f"{key}_confidence"] = (
                        sum(current_confs[key]) / len(current_confs[key])
                    )
                groups.append(current)

            # Start new group
            current = {"DRUG": text}
            current_confs = {"DRUG": [conf]}

        elif label.startswith("I-"):
            entity_type = label[2:]
            if entity_type in current:
                current[entity_type] += " " + text
                current_confs.setdefault(entity_type, []).append(conf)
            else:
                current[entity_type] = text
                current_confs[entity_type] = [conf]

        elif label.startswith("B-"):
            entity_type = label[2:]
            current[entity_type] = text
            current_confs[entity_type] = [conf]

    # Save last group
    if "DRUG" in current:
        for key in current_confs:
            current[f"{key}_confidence"] = (
                sum(current_confs[key]) / len(current_confs[key])
            )
        groups.append(current)

    return groups


# ─── Main Extraction Interface ─────────────────────────────────────────────────

def extract_with_biobert(raw_text: str, doc_type) -> "ExtractionResult":
    """
    Extract structured medical entities from raw OCR text using BioBERT NER.

    Args:
        raw_text: Raw OCR text.
        doc_type: DocType enum (prescription or lab_report).

    Returns:
        ExtractionResult with structured entities.
    """
    # Import here to avoid circular imports
    sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
    from extraction.extract import (
        DocType,
        ExtractionMethod,
        ExtractionResult,
        PrescriptionItem,
    )

    # BioBERT NER is designed for prescriptions
    # Lab reports are better handled by LLM due to their tabular structure
    if doc_type == DocType.LAB_REPORT:
        logger.warning(
            "BioBERT NER is optimized for prescriptions, not lab reports. "
            "Returning empty result — use LLM extraction for lab reports."
        )
        return ExtractionResult(
            doc_type=doc_type,
            method_used=ExtractionMethod.BIOBERT,
            raw_text=raw_text,
            success=True,
        )

    # Run NER prediction
    entities = predict_entities(raw_text)
    entity_groups = group_entities(entities)

    # Convert to PrescriptionItem objects
    items = []
    for group in entity_groups:
        drug_name = group.get("DRUG", "Unknown")
        if drug_name == "Unknown":
            continue

        avg_conf = np.mean([
            group.get("DRUG_confidence", 0.5),
            group.get("DOSAGE_confidence", 0.5),
        ])

        item = PrescriptionItem(
            drug_name=drug_name,
            dosage=group.get("DOSAGE"),
            frequency=group.get("FREQUENCY"),
            duration=group.get("DURATION"),
            instructions=group.get("INSTRUCTION"),
            confidence=round(float(avg_conf), 4),
        )
        items.append(item)

    return ExtractionResult(
        doc_type=doc_type,
        method_used=ExtractionMethod.BIOBERT,
        prescription_items=items,
        raw_text=raw_text,
        success=True,
    )


# ─── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

    sample = (
        "Rx\n"
        "1. Tab Amoxicillin 500mg 1-0-1 after food 5 days\n"
        "2. Cap Omeprazole 20mg 1-0-0 before food 7 days\n"
        "3. Syr Cetirizine 5ml 0-0-1 3 days\n"
    )

    print("=" * 60)
    print("BioBERT NER Entity Extraction Test")
    print("=" * 60)

    try:
        entities = predict_entities(sample)
        print("\nRaw entities:")
        for ent in entities:
            print(f"  [{ent['label']}] {ent['text']} (conf: {ent['confidence']:.3f})")

        groups = group_entities(entities)
        print(f"\nGrouped into {len(groups)} prescription items:")
        for i, g in enumerate(groups):
            print(f"  Item {i+1}: {json.dumps(g, indent=4)}")

    except FileNotFoundError as e:
        print(f"\n⚠ {e}")
        print("Run 'python train.py' first to fine-tune the model.")
