"""
Shared dataset utilities for prescription recognition models.
==============================================================

Provides:
  - PrescriptionDataset: classification dataset (image → label index)
  - PrescriptionOCRDataset: OCR dataset (image → text string)
  - PadToSquare: transform that pads images to square with white background
  - Helper functions for finding the dataset directory and building label maps
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


# ─── Helper Functions ──────────────────────────────────────────────────────────

def find_dataset_dir(base_path: str) -> str:
    """
    Find the Kaggle prescription dataset directory.
    Handles special characters in folder name ("Doctor's Handwritten...").
    
    Args:
        base_path: Path to the prescriptions data directory.
        
    Returns:
        Full path to the dataset directory.
    """
    base = Path(base_path)
    for d in base.iterdir():
        if d.is_dir() and "Prescription" in d.name:
            return str(d)
    raise FileNotFoundError(
        f"No prescription dataset directory found in {base_path}. "
        f"Expected a folder containing 'Prescription' in its name."
    )


def get_split_paths(dataset_dir: str, split: str) -> Tuple[str, str]:
    """
    Get CSV and image directory paths for a given split.
    
    Args:
        dataset_dir: Root of the Kaggle dataset.
        split: One of "Training", "Testing", "Validation".
        
    Returns:
        (csv_path, img_dir) tuple.
    """
    split_dir = os.path.join(dataset_dir, split)
    csv_path = os.path.join(split_dir, f"{split.lower()}_labels.csv")
    img_dir = os.path.join(split_dir, f"{split.lower()}_words")
    return csv_path, img_dir


def build_label_map(
    csv_path: str,
    label_column: str = "MEDICINE_NAME",
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build bidirectional label-to-index mapping from a CSV file.
    Labels are sorted alphabetically for reproducibility.
    
    Returns:
        (label_to_idx, idx_to_label) dictionaries.
    """
    df = pd.read_csv(csv_path)
    labels = sorted(df[label_column].unique())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def build_generic_map(csv_path: str) -> Dict[str, str]:
    """
    Build medicine brand name → generic name mapping from CSV.
    """
    df = pd.read_csv(csv_path)
    return dict(
        df.drop_duplicates("MEDICINE_NAME")[["MEDICINE_NAME", "GENERIC_NAME"]]
        .set_index("MEDICINE_NAME")["GENERIC_NAME"]
    )


# ─── Transforms ────────────────────────────────────────────────────────────────

class PadToSquare:
    """
    Pad an image to make it square, filling with white (255, 255, 255).
    Centers the original image in the padded result.
    
    This preserves the aspect ratio of handwritten word images
    before resizing to the model's expected input size.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_dim = max(w, h)
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
        return padded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fill=white)"


# ─── Datasets ──────────────────────────────────────────────────────────────────

class PrescriptionDataset(Dataset):
    """
    PyTorch Dataset for prescription handwriting classification.
    
    Each item returns (image_tensor, label_index) where label_index
    maps to a medicine name via the label_to_idx dictionary.
    
    All images are converted to RGB regardless of source mode
    (handles RGBA, L, P mode images in the dataset).
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        label_to_idx: Dict[str, int],
        transform=None,
        label_column: str = "MEDICINE_NAME",
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.label_column = label_column

        # Validate all labels exist in mapping
        unknown = set(self.df[label_column].unique()) - set(label_to_idx.keys())
        if unknown:
            print(f"⚠ WARNING: {len(unknown)} unknown labels: {unknown}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["IMAGE"])

        # Load and convert to RGB (handles RGBA, L, P modes)
        image = Image.open(img_path).convert("RGB")
        label = self.label_to_idx[row[self.label_column]]

        if self.transform:
            image = self.transform(image)

        return image, label


class PrescriptionOCRDataset(Dataset):
    """
    PyTorch Dataset for TrOCR-style sequence prediction.
    
    Returns (image, text_label) pairs where text_label is the
    raw medicine name string for tokenization by the TrOCR processor.
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        processor=None,
        label_column: str = "MEDICINE_NAME",
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.processor = processor
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["IMAGE"])
        image = Image.open(img_path).convert("RGB")
        text = row[self.label_column]

        if self.processor:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
            labels = self.processor.tokenizer(
                text, padding="max_length", max_length=32, truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}

        return image, text
