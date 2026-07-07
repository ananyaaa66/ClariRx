"""
ClariRx — Model Evaluation & Comparison
=========================================

Evaluates both the CNN Classifier and the TrOCR model on the Test set.
Generates a side-by-side comparison report with accuracy, CER, and speed.
"""

import argparse
import json
import time
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torchvision.transforms as T
import torch.nn as nn
from torchvision.models import efficientnet_b0

from dataset import PrescriptionDataset, PrescriptionOCRDataset, PadToSquare, find_dataset_dir, get_split_paths

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "raw" / "prescriptions"

def evaluate_cnn(csv_path, img_dir, model_path, device):
    """Evaluate the EfficientNet-B0 CNN."""
    print("\n[1/2] Evaluating CNN Classifier...")
    # Load dataset to get class count
    df = pd.read_csv(csv_path)
    classes = sorted(df["MEDICINE_NAME"].unique())
    label_to_idx = {name: idx for idx, name in enumerate(classes)}
    
    transform = T.Compose([
        PadToSquare(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = PrescriptionDataset(csv_path, img_dir, label_to_idx, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load Model
    model = efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="CNN Inferencing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
    inf_time = time.time() - start_time
    accuracy = correct / total
    
    return {
        "accuracy": accuracy,
        "time_per_image_ms": (inf_time / total) * 1000
    }

def evaluate_trocr(csv_path, img_dir, model_path, device):
    """Evaluate the TrOCR fine-tuned model."""
    print("\n[2/2] Evaluating TrOCR Model...")
    
    processor = TrOCRProcessor.from_pretrained(model_path, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    dataset = PrescriptionOCRDataset(csv_path, img_dir, processor=None)
    
    cer_metric = evaluate.load("cer")
    
    correct = 0
    total = len(dataset)
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in tqdm(range(total), desc="TrOCR Inferencing"):
            image, label = dataset[i]
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            
            generated_ids = model.generate(pixel_values, max_length=32)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            all_preds.append(generated_text)
            all_labels.append(label)
            
            if generated_text.strip().lower() == label.strip().lower():
                correct += 1
                
    inf_time = time.time() - start_time
    accuracy = correct / total
    cer = cer_metric.compute(predictions=all_preds, references=all_labels)
    
    return {
        "accuracy": accuracy,
        "cer": cer,
        "time_per_image_ms": (inf_time / total) * 1000
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--cnn-model", type=str, default=str(SCRIPT_DIR / "checkpoints" / "efficientnet_b0_best.pth"))
    parser.add_argument("--trocr-model", type=str, default=str(SCRIPT_DIR / "checkpoints" / "trocr_best"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    dataset_dir = find_dataset_dir(args.data_dir)
    test_csv, test_img_dir = get_split_paths(dataset_dir, "Testing")
    
    # 1. Evaluate CNN
    cnn_results = None
    if Path(args.cnn_model).exists():
        cnn_results = evaluate_cnn(test_csv, test_img_dir, args.cnn_model, device)
    else:
        print(f"CNN model not found at {args.cnn_model}")
        
    # 2. Evaluate TrOCR
    trocr_results = None
    if Path(args.trocr_model).exists():
        trocr_results = evaluate_trocr(test_csv, test_img_dir, args.trocr_model, device)
    else:
        print(f"TrOCR model not found at {args.trocr_model}")
        
    # 3. Compare and Output
    print("\n" + "="*50)
    print("  ClariRx — OCR Models Comparison (Test Set)")
    print("="*50)
    
    if cnn_results:
        print(f"\n[CNN Classifier - EfficientNet-B0]")
        print(f"  Exact Match Accuracy : {cnn_results['accuracy']:.4f} ({cnn_results['accuracy']*100:.2f}%)")
        print(f"  Inference Speed      : {cnn_results['time_per_image_ms']:.1f} ms/image")
        
    if trocr_results:
        print(f"\n[TrOCR - microsoft/trocr-base-handwritten]")
        print(f"  Exact Match Accuracy : {trocr_results['accuracy']:.4f} ({trocr_results['accuracy']*100:.2f}%)")
        print(f"  Character Error Rate : {trocr_results['cer']:.4f} ({trocr_results['cer']*100:.2f}%)")
        print(f"  Inference Speed      : {trocr_results['time_per_image_ms']:.1f} ms/image")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
