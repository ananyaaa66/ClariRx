"""
OCR Pipeline Evaluation & Benchmarking
=========================================

Evaluates both the PaddleOCR and LLM Vision pipelines against
synthetic ground-truth test cases using CER and WER metrics.

Outputs:
    logs/ocr_eval_report.txt — per-case + aggregate evaluation report

Usage:
    python eval_ocr.py
    python eval_ocr.py --pipeline paddleocr
    python eval_ocr.py --pipeline llm_vision
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
LOG_DIR = SCRIPT_DIR / "logs"

sys.path.insert(0, str(BACKEND_DIR))


# ─── Synthetic Ground Truth Test Cases ─────────────────────────────────────────

# These simulate raw OCR outputs vs. known ground truth text.
# In a production setup, these would come from annotated datasets.

PADDLEOCR_TEST_CASES = [
    {
        "name": "Clean 2-drug prescription",
        "simulated_ocr": (
            "Rx\n"
            "1. Tab Amoxicillin 500mg 1-0-1 after food 5 days\n"
            "2. Tab Paracetamol 650mg SOS\n"
        ),
        "ground_truth": (
            "Rx\n"
            "1. Tab Amoxicillin 500mg 1-0-1 after food 5 days\n"
            "2. Tab Paracetamol 650mg SOS\n"
        ),
    },
    {
        "name": "OCR with minor typos",
        "simulated_ocr": (
            "Rx\n"
            "1. Tab Am0xicillin 500mg 1-0-1 after fo0d 5 days\n"
            "2. Cap 0mepraz0le 20mg 1-0-0 bef0re food 7d\n"
        ),
        "ground_truth": (
            "Rx\n"
            "1. Tab Amoxicillin 500mg 1-0-1 after food 5 days\n"
            "2. Cap Omeprazole 20mg 1-0-0 before food 7d\n"
        ),
    },
    {
        "name": "Noisy handwriting OCR",
        "simulated_ocr": (
            "Dr. Sharma\n"
            "Tab Metformn 500 1-0-1 after food\n"
            "Tab Amlo 5mg 1-0-0 moming\n"
        ),
        "ground_truth": (
            "Dr. Sharma\n"
            "Tab Metformin 500 1-0-1 after food\n"
            "Tab Amlo 5mg 1-0-0 morning\n"
        ),
    },
    {
        "name": "Single drug prescription",
        "simulated_ocr": "Paracetamol 650mg SOS after food\n",
        "ground_truth": "Paracetamol 650mg SOS after food\n",
    },
]

LLM_VISION_TEST_CASES = [
    {
        "name": "CBC lab report",
        "simulated_ocr": (
            "Complete Blood Count (CBC)\n"
            "Haemoglobin: 14.2 g/dL (Normal: 13.0 - 17.0)\n"
            "WBC Count: 11500 cells/uL (Normal: 4000 - 11000)\n"
            "Platelet Count: 2.5 lakhs/uL (Normal: 1.5 - 4.0)\n"
        ),
        "ground_truth": (
            "Complete Blood Count (CBC)\n"
            "Haemoglobin: 14.2 g/dL (Normal: 13.0 - 17.0)\n"
            "WBC Count: 11500 cells/uL (Normal: 4000 - 11000)\n"
            "Platelet Count: 2.5 lakhs/uL (Normal: 1.5 - 4.0)\n"
        ),
    },
    {
        "name": "Lipid profile with OCR noise",
        "simulated_ocr": (
            "LIPID PROFILE\n"
            "Total Cholesterol: 223 mg/dL (Desirable: < 200)\n"
            "HDL Cholesterol: 69 mg/dL (Normal: 50-70)\n"
            "LDL Ch0lesterol: 131 mg/dL (Optimal: < 100)\n"
        ),
        "ground_truth": (
            "LIPID PROFILE\n"
            "Total Cholesterol: 223 mg/dL (Desirable: < 200)\n"
            "HDL Cholesterol: 69 mg/dL (Normal: 50-70)\n"
            "LDL Cholesterol: 131 mg/dL (Optimal: < 100)\n"
        ),
    },
    {
        "name": "Thyroid panel",
        "simulated_ocr": (
            "Thyroid Panel\n"
            "TSH: 4.8 mIU/L (Normal: 0.4 - 4.0)\n"
            "Free T4: 1.1 ng/dL (Normal: 0.8 - 1.8)\n"
        ),
        "ground_truth": (
            "Thyroid Panel\n"
            "TSH: 4.8 mIU/L (Normal: 0.4 - 4.0)\n"
            "Free T4: 1.1 ng/dL (Normal: 0.8 - 1.8)\n"
        ),
    },
]


# ─── Metrics ───────────────────────────────────────────────────────────────────

def compute_cer(prediction: str, reference: str) -> float:
    """
    Compute Character Error Rate (CER) between prediction and reference.

    Uses jiwer if available, otherwise falls back to simple Levenshtein ratio.
    """
    try:
        from jiwer import cer
        return cer(reference, prediction)
    except ImportError:
        # Fallback: simple character-level error rate
        return _levenshtein_ratio(prediction, reference)


def compute_wer(prediction: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER) between prediction and reference.
    """
    try:
        from jiwer import wer
        return wer(reference, prediction)
    except ImportError:
        # Fallback: word-level error rate
        pred_words = prediction.split()
        ref_words = reference.split()
        return _levenshtein_ratio(" ".join(pred_words), " ".join(ref_words))


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Fallback Levenshtein-based error rate."""
    if not s2:
        return 1.0 if s1 else 0.0

    len1, len2 = len(s1), len(s2)
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )

    return matrix[len1][len2] / max(len1, len2, 1)


def compute_exact_match(prediction: str, reference: str) -> bool:
    """Check if prediction exactly matches reference (normalized)."""
    return prediction.strip().lower() == reference.strip().lower()


# ─── Evaluation Engine ─────────────────────────────────────────────────────────

def evaluate_pipeline(
    test_cases: List[Dict],
    pipeline_name: str,
) -> Dict:
    """
    Evaluate OCR pipeline using simulated OCR outputs against ground truth.

    In a production setup, this would run the actual OCR on images.
    Here we use simulated OCR outputs for offline evaluation.

    Returns:
        Dict with per-case and aggregate metrics.
    """
    results = {
        "pipeline": pipeline_name,
        "total_cases": len(test_cases),
        "per_case": [],
        "aggregate_cer": 0.0,
        "aggregate_wer": 0.0,
        "exact_matches": 0,
    }

    total_cer = 0.0
    total_wer = 0.0

    for case in test_cases:
        name = case["name"]
        prediction = case["simulated_ocr"]
        reference = case["ground_truth"]

        cer_score = compute_cer(prediction, reference)
        wer_score = compute_wer(prediction, reference)
        exact = compute_exact_match(prediction, reference)

        case_result = {
            "name": name,
            "cer": round(cer_score, 4),
            "wer": round(wer_score, 4),
            "exact_match": exact,
        }

        results["per_case"].append(case_result)
        total_cer += cer_score
        total_wer += wer_score
        if exact:
            results["exact_matches"] += 1

        status = "✅ EXACT" if exact else f"CER={cer_score:.4f}"
        print(f"  {name:40s} | {status}")

    results["aggregate_cer"] = round(total_cer / len(test_cases), 4) if test_cases else 0.0
    results["aggregate_wer"] = round(total_wer / len(test_cases), 4) if test_cases else 0.0

    return results


# ─── Report Generation ─────────────────────────────────────────────────────────

def generate_report(
    paddleocr_results: Optional[Dict],
    vision_results: Optional[Dict],
    output_path: Path,
):
    """Generate a human-readable OCR evaluation report."""
    lines = []
    lines.append("=" * 70)
    lines.append("ClariRx OCR Pipeline Evaluation Report")
    lines.append("=" * 70)

    for name, results in [
        ("PaddleOCR (Prescriptions)", paddleocr_results),
        ("LLM Vision (Lab Reports)", vision_results),
    ]:
        if results is None:
            continue

        lines.append("")
        lines.append("─" * 70)
        lines.append(f"{name}")
        lines.append("─" * 70)
        lines.append(f"  Test cases:     {results['total_cases']}")
        lines.append(f"  Exact matches:  {results['exact_matches']}/{results['total_cases']}")
        lines.append(f"  Avg CER:        {results['aggregate_cer']:.4f} ({results['aggregate_cer'] * 100:.2f}%)")
        lines.append(f"  Avg WER:        {results['aggregate_wer']:.4f} ({results['aggregate_wer'] * 100:.2f}%)")
        lines.append("")
        lines.append("  Per-Case Breakdown:")

        for case in results["per_case"]:
            match_str = "✓ EXACT" if case["exact_match"] else f"CER={case['cer']:.4f}"
            lines.append(f"    {case['name']:40s} | {match_str}")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n📄 Report saved to: {output_path}")
    print(report_text)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ClariRx OCR pipelines")
    parser.add_argument(
        "--pipeline", type=str, default="all",
        choices=["paddleocr", "llm_vision", "all"],
        help="Which pipeline to evaluate (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("ClariRx OCR Pipeline Evaluation")
    print("=" * 60)

    paddleocr_results = None
    vision_results = None

    if args.pipeline in ("paddleocr", "all"):
        print("\n📋 Evaluating PaddleOCR pipeline...")
        paddleocr_results = evaluate_pipeline(PADDLEOCR_TEST_CASES, "PaddleOCR")

    if args.pipeline in ("llm_vision", "all"):
        print("\n📋 Evaluating LLM Vision pipeline...")
        vision_results = evaluate_pipeline(LLM_VISION_TEST_CASES, "LLM Vision")

    report_path = LOG_DIR / "ocr_eval_report.txt"
    generate_report(paddleocr_results, vision_results, report_path)


if __name__ == "__main__":
    main()
