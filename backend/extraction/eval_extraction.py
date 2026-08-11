"""
Extraction Evaluation Benchmark
=================================

Evaluates the extraction pipeline by running both LLM and BioBERT
methods on test prescription/lab report texts and comparing against
ground truth annotations.

Metrics:
  - Per-entity Precision, Recall, F1-Score
  - Fuzzy drug name matching using rapidfuzz (tolerates OCR typos)
  - Overall extraction accuracy

Outputs:
  logs/extraction_eval_report.txt — detailed evaluation report

Usage:
    python eval_extraction.py
    python eval_extraction.py --method llm --provider gemini
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Path Setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
LOG_DIR = SCRIPT_DIR / "logs"

sys.path.insert(0, str(BACKEND_DIR))


# ─── Test Cases (synthetic ground truth) ───────────────────────────────────────

# Since prescriptions_gt.csv is empty, we define inline test cases
# Each case: (raw_ocr_text, expected_items)

PRESCRIPTION_TEST_CASES = [
    {
        "name": "Simple 2-drug prescription",
        "raw_text": (
            "Rx\n"
            "1. Tab Amoxicillin 500mg  1-0-1  after food  5 days\n"
            "2. Tab Paracetamol 650mg  SOS\n"
        ),
        "expected": [
            {"drug_name": "Amoxicillin 500mg", "frequency": "1-0-1", "duration": "5 days"},
            {"drug_name": "Paracetamol 650mg", "frequency": "SOS"},
        ],
    },
    {
        "name": "3-drug with OCR typos",
        "raw_text": (
            "Tab Am0xicillin 500  1-0-1  after food  5d\n"
            "Cap Omepraz0le 20mg  1-0-0  before food  7d\n"
            "Syr Cetirizne 5ml  0-0-1  3 days\n"
        ),
        "expected": [
            {"drug_name": "Amoxicillin 500mg", "frequency": "1-0-1"},
            {"drug_name": "Omeprazole 20mg", "frequency": "1-0-0"},
            {"drug_name": "Cetirizine 5ml", "frequency": "0-0-1"},
        ],
    },
    {
        "name": "Single drug SOS",
        "raw_text": "Paracetamol 650mg SOS after food\n",
        "expected": [
            {"drug_name": "Paracetamol 650mg", "frequency": "SOS"},
        ],
    },
    {
        "name": "Complex multi-line",
        "raw_text": (
            "Dr. Sharma — Prescription\n"
            "Patient: Mrs. Gupta\n"
            "1) Tab Metformin 500mg  1-0-1  after food  Continue\n"
            "2) Tab Amlodipine 5mg  1-0-0  morning  Continue\n"
            "3) Tab Atorvastatin 10mg  0-0-1  at bedtime  Continue\n"
            "4) Tab Aspirin 75mg  1-0-0  after breakfast\n"
        ),
        "expected": [
            {"drug_name": "Metformin 500mg", "frequency": "1-0-1"},
            {"drug_name": "Amlodipine 5mg", "frequency": "1-0-0"},
            {"drug_name": "Atorvastatin 10mg", "frequency": "0-0-1"},
            {"drug_name": "Aspirin 75mg", "frequency": "1-0-0"},
        ],
    },
    {
        "name": "Messy OCR with noise",
        "raw_text": (
            "Rx:\n"
            "Tab Azithromycn 500mg  OD  x 3d\n"
            "Tab Montlukast 10mg  0-0-1  14 days\n"
        ),
        "expected": [
            {"drug_name": "Azithromycin 500mg", "frequency": "OD", "duration": "3 days"},
            {"drug_name": "Montelukast 10mg", "frequency": "0-0-1", "duration": "14 days"},
        ],
    },
]

LAB_REPORT_TEST_CASES = [
    {
        "name": "CBC Report",
        "raw_text": (
            "Complete Blood Count (CBC)\n"
            "Haemoglobin: 14.2 g/dL  (Normal: 13.0 - 17.0)\n"
            "WBC Count: 11500 cells/uL  (Normal: 4000 - 11000)\n"
            "Platelet Count: 2.5 lakhs/uL  (Normal: 1.5 - 4.0)\n"
        ),
        "expected": [
            {"test_name": "Haemoglobin", "value": "14.2", "is_abnormal": False},
            {"test_name": "WBC Count", "value": "11500", "is_abnormal": True},
            {"test_name": "Platelet Count", "value": "2.5", "is_abnormal": False},
        ],
    },
    {
        "name": "Lipid Profile",
        "raw_text": (
            "LIPID PROFILE\n"
            "Total Cholesterol: 223 mg/dL  (Desirable: < 200)\n"
            "HDL Cholesterol: 69 mg/dL  (Normal: 50-70)\n"
            "LDL Cholesterol: 131 mg/dL  (Optimal: < 100)\n"
        ),
        "expected": [
            {"test_name": "Total Cholesterol", "value": "223", "is_abnormal": True},
            {"test_name": "HDL Cholesterol", "value": "69", "is_abnormal": False},
            {"test_name": "LDL Cholesterol", "value": "131", "is_abnormal": True},
        ],
    },
]


# ─── Fuzzy Matching Utilities ─────────────────────────────────────────────────

def fuzzy_match_drug(predicted: str, expected: str, threshold: int = 75) -> bool:
    """Check if predicted drug name matches expected using fuzzy string matching."""
    from rapidfuzz import fuzz

    # Normalize: lowercase, strip whitespace
    p = predicted.lower().strip()
    e = expected.lower().strip()

    # Exact match
    if p == e:
        return True

    # Fuzzy partial ratio (handles OCR typos like Am0x → Amox)
    score = fuzz.partial_ratio(p, e)
    return score >= threshold


def fuzzy_match_test(predicted: str, expected: str, threshold: int = 75) -> bool:
    """Check if predicted lab test name matches expected using fuzzy matching."""
    from rapidfuzz import fuzz

    p = predicted.lower().strip()
    e = expected.lower().strip()

    if p == e or e in p or p in e:
        return True

    return fuzz.partial_ratio(p, e) >= threshold


# ─── Evaluation Engine ─────────────────────────────────────────────────────────

def evaluate_prescription_extraction(
    test_cases: List[Dict],
    method: str = "llm",
    model_provider: str = "gemini",
) -> Dict:
    """
    Evaluate prescription extraction against ground truth test cases.

    Returns metrics dict with per-field and overall scores.
    """
    from extraction.extract import run_extraction

    results = {
        "total_cases": len(test_cases),
        "total_expected_items": 0,
        "total_predicted_items": 0,
        "drug_matches": 0,
        "frequency_matches": 0,
        "duration_matches": 0,
        "per_case": [],
    }

    for case in test_cases:
        case_name = case["name"]
        raw_text = case["raw_text"]
        expected_items = case["expected"]

        print(f"\n  📝 {case_name}...")

        try:
            result = run_extraction(
                raw_text,
                doc_type="prescription",
                method=method,
                model_provider=model_provider,
                fallback=False,
            )
            predicted_items = result.prescription_items
        except Exception as e:
            logger.error(f"  ❌ Extraction failed: {e}")
            predicted_items = []

        results["total_expected_items"] += len(expected_items)
        results["total_predicted_items"] += len(predicted_items)

        # Match predicted items to expected items
        case_result = {
            "name": case_name,
            "expected_count": len(expected_items),
            "predicted_count": len(predicted_items),
            "drug_matched": 0,
            "freq_matched": 0,
            "dur_matched": 0,
        }

        matched_expected = set()
        for pred in predicted_items:
            best_match_idx = None
            best_score = 0

            for idx, exp in enumerate(expected_items):
                if idx in matched_expected:
                    continue
                if fuzzy_match_drug(pred.drug_name, exp["drug_name"]):
                    best_match_idx = idx
                    break

            if best_match_idx is not None:
                matched_expected.add(best_match_idx)
                exp = expected_items[best_match_idx]

                results["drug_matches"] += 1
                case_result["drug_matched"] += 1

                # Check frequency
                if pred.frequency and "frequency" in exp:
                    if pred.frequency.strip() == exp["frequency"].strip():
                        results["frequency_matches"] += 1
                        case_result["freq_matched"] += 1

                # Check duration
                if pred.duration and "duration" in exp:
                    from rapidfuzz import fuzz
                    if fuzz.partial_ratio(
                        pred.duration.lower(), exp["duration"].lower()
                    ) >= 70:
                        results["duration_matches"] += 1
                        case_result["dur_matched"] += 1

        results["per_case"].append(case_result)

    # Compute aggregate metrics
    total_exp = results["total_expected_items"]
    drug_m = results["drug_matches"]

    results["drug_precision"] = (
        drug_m / results["total_predicted_items"]
        if results["total_predicted_items"] > 0 else 0.0
    )
    results["drug_recall"] = drug_m / total_exp if total_exp > 0 else 0.0
    results["drug_f1"] = (
        2 * results["drug_precision"] * results["drug_recall"]
        / (results["drug_precision"] + results["drug_recall"])
        if (results["drug_precision"] + results["drug_recall"]) > 0 else 0.0
    )

    return results


def evaluate_lab_report_extraction(
    test_cases: List[Dict],
    method: str = "llm",
    model_provider: str = "gemini",
) -> Dict:
    """Evaluate lab report extraction against ground truth test cases."""
    from extraction.extract import run_extraction

    results = {
        "total_cases": len(test_cases),
        "total_expected_items": 0,
        "total_predicted_items": 0,
        "test_name_matches": 0,
        "abnormality_correct": 0,
        "per_case": [],
    }

    for case in test_cases:
        case_name = case["name"]
        raw_text = case["raw_text"]
        expected_items = case["expected"]

        print(f"\n  📝 {case_name}...")

        try:
            result = run_extraction(
                raw_text,
                doc_type="lab_report",
                method=method,
                model_provider=model_provider,
                fallback=False,
            )
            predicted_items = result.lab_report_items
        except Exception as e:
            logger.error(f"  ❌ Extraction failed: {e}")
            predicted_items = []

        results["total_expected_items"] += len(expected_items)
        results["total_predicted_items"] += len(predicted_items)

        case_result = {
            "name": case_name,
            "expected_count": len(expected_items),
            "predicted_count": len(predicted_items),
            "name_matched": 0,
            "abnormality_correct": 0,
        }

        matched_expected = set()
        for pred in predicted_items:
            for idx, exp in enumerate(expected_items):
                if idx in matched_expected:
                    continue
                if fuzzy_match_test(pred.test_name, exp["test_name"]):
                    matched_expected.add(idx)
                    results["test_name_matches"] += 1
                    case_result["name_matched"] += 1

                    if pred.is_abnormal == exp.get("is_abnormal", False):
                        results["abnormality_correct"] += 1
                        case_result["abnormality_correct"] += 1
                    break

        results["per_case"].append(case_result)

    total_exp = results["total_expected_items"]
    name_m = results["test_name_matches"]

    results["name_precision"] = (
        name_m / results["total_predicted_items"]
        if results["total_predicted_items"] > 0 else 0.0
    )
    results["name_recall"] = name_m / total_exp if total_exp > 0 else 0.0
    results["name_f1"] = (
        2 * results["name_precision"] * results["name_recall"]
        / (results["name_precision"] + results["name_recall"])
        if (results["name_precision"] + results["name_recall"]) > 0 else 0.0
    )

    results["abnormality_accuracy"] = (
        results["abnormality_correct"] / name_m if name_m > 0 else 0.0
    )

    return results


# ─── Report Generation ─────────────────────────────────────────────────────────

def generate_report(
    rx_results: Optional[Dict],
    lab_results: Optional[Dict],
    method: str,
    output_path: Path,
):
    """Generate a human-readable evaluation report."""
    lines = []
    lines.append("=" * 70)
    lines.append("ClariRx Extraction Evaluation Report")
    lines.append(f"Method: {method.upper()}")
    lines.append("=" * 70)

    if rx_results:
        lines.append("")
        lines.append("─" * 70)
        lines.append("PRESCRIPTION EXTRACTION")
        lines.append("─" * 70)
        lines.append(f"  Test cases:        {rx_results['total_cases']}")
        lines.append(f"  Expected items:    {rx_results['total_expected_items']}")
        lines.append(f"  Predicted items:   {rx_results['total_predicted_items']}")
        lines.append("")
        lines.append("  Drug Name Matching (fuzzy):")
        lines.append(f"    Matches:    {rx_results['drug_matches']}")
        lines.append(f"    Precision:  {rx_results['drug_precision']:.4f}")
        lines.append(f"    Recall:     {rx_results['drug_recall']:.4f}")
        lines.append(f"    F1-Score:   {rx_results['drug_f1']:.4f}")
        lines.append("")
        lines.append("  Per-Case Breakdown:")
        for case in rx_results["per_case"]:
            lines.append(
                f"    {case['name']:40s} | "
                f"Drug: {case['drug_matched']}/{case['expected_count']}  "
                f"Freq: {case['freq_matched']}/{case['expected_count']}  "
                f"Dur: {case['dur_matched']}/{case['expected_count']}"
            )

    if lab_results:
        lines.append("")
        lines.append("─" * 70)
        lines.append("LAB REPORT EXTRACTION")
        lines.append("─" * 70)
        lines.append(f"  Test cases:        {lab_results['total_cases']}")
        lines.append(f"  Expected items:    {lab_results['total_expected_items']}")
        lines.append(f"  Predicted items:   {lab_results['total_predicted_items']}")
        lines.append("")
        lines.append("  Test Name Matching (fuzzy):")
        lines.append(f"    Matches:    {lab_results['test_name_matches']}")
        lines.append(f"    Precision:  {lab_results['name_precision']:.4f}")
        lines.append(f"    Recall:     {lab_results['name_recall']:.4f}")
        lines.append(f"    F1-Score:   {lab_results['name_f1']:.4f}")
        lines.append(f"    Abnormality Accuracy: {lab_results['abnormality_accuracy']:.4f}")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"\n📄 Report saved to: {output_path}")
    print(report_text)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ClariRx extraction pipeline")
    parser.add_argument(
        "--method", type=str, default="llm", choices=["llm", "biobert"],
        help="Extraction method to evaluate (default: llm)"
    )
    parser.add_argument(
        "--provider", type=str, default="gemini", choices=["gemini", "groq"],
        help="LLM provider for LLM method (default: gemini)"
    )
    parser.add_argument(
        "--skip-rx", action="store_true",
        help="Skip prescription evaluation"
    )
    parser.add_argument(
        "--skip-lab", action="store_true",
        help="Skip lab report evaluation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BACKEND_DIR, ".env"))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    print("=" * 60)
    print(f"ClariRx Extraction Evaluation — Method: {args.method.upper()}")
    print("=" * 60)

    rx_results = None
    lab_results = None

    if not args.skip_rx:
        print("\n📋 Evaluating PRESCRIPTION extraction...")
        rx_results = evaluate_prescription_extraction(
            PRESCRIPTION_TEST_CASES,
            method=args.method,
            model_provider=args.provider,
        )

    if not args.skip_lab:
        print("\n📋 Evaluating LAB REPORT extraction...")
        lab_results = evaluate_lab_report_extraction(
            LAB_REPORT_TEST_CASES,
            method=args.method,
            model_provider=args.provider,
        )

    report_path = LOG_DIR / "extraction_eval_report.txt"
    generate_report(rx_results, lab_results, args.method, report_path)
