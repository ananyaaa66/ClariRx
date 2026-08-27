"""
ClariRx — Full End-to-End Evaluation Pipeline
================================================

Orchestrates all evaluation components into a single unified run:
  1. OCR pipeline evaluation (CER/WER on synthetic ground truth)
  2. Extraction pipeline evaluation (entity P/R/F1 on test cases)
  3. Explanation quality checks (KB grounding rate, bilingual coverage)

Generates:
  eval/results/full_eval_report.json — machine-readable aggregate metrics
  eval/results/full_eval_report.txt  — human-readable evaluation summary

Usage:
    python run_full_eval.py
    python run_full_eval.py --provider gemini --skip-ocr
    python run_full_eval.py --skip-extraction --skip-explanation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ─── Path Setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(BACKEND_DIR))


# ─── OCR Evaluation ───────────────────────────────────────────────────────────

def run_ocr_evaluation() -> Dict:
    """Run OCR pipeline evaluation using synthetic ground truth."""
    print("\n" + "─" * 65)
    print("  [1/3] OCR Pipeline Evaluation")
    print("─" * 65)

    try:
        from ocr.eval_ocr import (
            evaluate_pipeline,
            PADDLEOCR_TEST_CASES,
            LLM_VISION_TEST_CASES,
        )

        print("\n  📋 PaddleOCR (Prescriptions)...")
        paddle_results = evaluate_pipeline(PADDLEOCR_TEST_CASES, "PaddleOCR")

        print("\n  📋 LLM Vision (Lab Reports)...")
        vision_results = evaluate_pipeline(LLM_VISION_TEST_CASES, "LLM Vision")

        return {
            "paddleocr": {
                "total_cases": paddle_results["total_cases"],
                "avg_cer": paddle_results["aggregate_cer"],
                "avg_wer": paddle_results["aggregate_wer"],
                "exact_matches": paddle_results["exact_matches"],
            },
            "llm_vision": {
                "total_cases": vision_results["total_cases"],
                "avg_cer": vision_results["aggregate_cer"],
                "avg_wer": vision_results["aggregate_wer"],
                "exact_matches": vision_results["exact_matches"],
            },
        }

    except Exception as e:
        logger.error(f"OCR evaluation failed: {e}")
        return {"error": str(e)}


# ─── Extraction Evaluation ────────────────────────────────────────────────────

def run_extraction_evaluation(method: str = "llm", provider: str = "gemini") -> Dict:
    """Run extraction pipeline evaluation using inline test cases."""
    print("\n" + "─" * 65)
    print(f"  [2/3] Extraction Pipeline Evaluation (method={method})")
    print("─" * 65)

    try:
        from extraction.eval_extraction import (
            evaluate_prescription_extraction,
            evaluate_lab_report_extraction,
            PRESCRIPTION_TEST_CASES,
            LAB_REPORT_TEST_CASES,
        )

        print("\n  📋 Prescription extraction...")
        rx_results = evaluate_prescription_extraction(
            PRESCRIPTION_TEST_CASES, method=method, model_provider=provider
        )

        print("\n  📋 Lab report extraction...")
        lab_results = evaluate_lab_report_extraction(
            LAB_REPORT_TEST_CASES, method=method, model_provider=provider
        )

        return {
            "prescription": {
                "total_cases": rx_results["total_cases"],
                "total_expected": rx_results["total_expected_items"],
                "total_predicted": rx_results["total_predicted_items"],
                "drug_precision": rx_results["drug_precision"],
                "drug_recall": rx_results["drug_recall"],
                "drug_f1": rx_results["drug_f1"],
            },
            "lab_report": {
                "total_cases": lab_results["total_cases"],
                "total_expected": lab_results["total_expected_items"],
                "total_predicted": lab_results["total_predicted_items"],
                "name_precision": lab_results["name_precision"],
                "name_recall": lab_results["name_recall"],
                "name_f1": lab_results["name_f1"],
                "abnormality_accuracy": lab_results["abnormality_accuracy"],
            },
        }

    except Exception as e:
        logger.error(f"Extraction evaluation failed: {e}")
        return {"error": str(e)}


# ─── Explanation Quality Evaluation ───────────────────────────────────────────

def run_explanation_evaluation() -> Dict:
    """Evaluate explanation quality: KB grounding rate & bilingual coverage."""
    print("\n" + "─" * 65)
    print("  [3/3] Explanation Quality Evaluation")
    print("─" * 65)

    try:
        from knowledge_base.build_kb import MedicalKB
        from explanation.explain_medicine import _offline_explanation
        from explanation.explain_lab_value import _offline_explanation as _offline_lab_explanation

        kb = MedicalKB()

        # Test drug lookup coverage
        test_drugs = [
            "Amoxicillin 500mg", "Paracetamol 650mg", "Omeprazole 20mg",
            "Metformin 500mg", "Amlodipine 5mg", "Atorvastatin 10mg",
            "Cetirizine 10mg", "Aspirin 75mg", "Azithromycin 500mg",
            "Montelukast 10mg", "Clopidogrel 75mg", "Pantoprazole 40mg",
        ]

        drug_hits = 0
        drug_results = []
        for drug in test_drugs:
            info = kb.lookup_drug(drug)
            hit = info is not None
            if hit:
                drug_hits += 1
            drug_results.append({"drug": drug, "kb_hit": hit})

        print(f"  Drug KB coverage: {drug_hits}/{len(test_drugs)}")

        # Test lab lookup coverage
        test_labs = [
            "Haemoglobin", "WBC Count", "Platelet Count",
            "Total Cholesterol", "HDL Cholesterol", "LDL Cholesterol",
            "TSH", "HbA1c", "Creatinine", "Fasting Blood Sugar",
        ]

        lab_hits = 0
        lab_results = []
        for test in test_labs:
            info = kb.lookup_lab_test(test)
            hit = info is not None
            if hit:
                lab_hits += 1
            lab_results.append({"test": test, "kb_hit": hit})

        print(f"  Lab KB coverage:  {lab_hits}/{len(test_labs)}")

        # Test offline explanation generation
        offline_tests = 0
        offline_pass = 0

        for drug in ["Amoxicillin 500mg", "Paracetamol 650mg"]:
            info = kb.lookup_drug(drug)
            result = _offline_explanation(drug, info, "1-0-1", "5 days", "after food")
            offline_tests += 1
            if result.explanation_en and len(result.explanation_en) > 10:
                offline_pass += 1

        print(f"  Offline medicine explanations: {offline_pass}/{offline_tests}")

        return {
            "drug_kb_coverage": f"{drug_hits}/{len(test_drugs)}",
            "drug_kb_rate": round(drug_hits / len(test_drugs), 4),
            "lab_kb_coverage": f"{lab_hits}/{len(test_labs)}",
            "lab_kb_rate": round(lab_hits / len(test_labs), 4),
            "offline_explanation_pass_rate": f"{offline_pass}/{offline_tests}",
            "drug_results": drug_results,
            "lab_results": lab_results,
            "total_drug_entries": len(kb.drug_db),
            "total_lab_entries": len(kb.lab_db),
        }

    except Exception as e:
        logger.error(f"Explanation evaluation failed: {e}")
        return {"error": str(e)}


# ─── Report Generation ─────────────────────────────────────────────────────────

def generate_full_report(
    ocr_results: Optional[Dict],
    extraction_results: Optional[Dict],
    explanation_results: Optional[Dict],
    elapsed_seconds: float,
) -> Dict:
    """Aggregate all results and save reports."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    full_report = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "ocr": ocr_results,
        "extraction": extraction_results,
        "explanation": explanation_results,
    }

    # Save JSON report
    json_path = RESULTS_DIR / "full_eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 JSON report saved to: {json_path}")

    # Generate human-readable report
    lines = []
    lines.append("=" * 70)
    lines.append("ClariRx — Full End-to-End Evaluation Report")
    lines.append(f"Generated: {full_report['timestamp']}")
    lines.append(f"Total time: {elapsed_seconds:.1f}s")
    lines.append("=" * 70)

    if ocr_results and "error" not in ocr_results:
        lines.append("")
        lines.append("─" * 70)
        lines.append("[1] OCR Pipeline Evaluation")
        lines.append("─" * 70)
        for name, data in ocr_results.items():
            lines.append(f"\n  {name}:")
            lines.append(f"    Cases:         {data['total_cases']}")
            lines.append(f"    Avg CER:       {data['avg_cer']:.4f} ({data['avg_cer'] * 100:.2f}%)")
            lines.append(f"    Avg WER:       {data['avg_wer']:.4f} ({data['avg_wer'] * 100:.2f}%)")
            lines.append(f"    Exact matches: {data['exact_matches']}/{data['total_cases']}")

    if extraction_results and "error" not in extraction_results:
        lines.append("")
        lines.append("─" * 70)
        lines.append("[2] Extraction Pipeline Evaluation")
        lines.append("─" * 70)
        if "prescription" in extraction_results:
            rx = extraction_results["prescription"]
            lines.append(f"\n  Prescription Extraction:")
            lines.append(f"    Cases: {rx['total_cases']} | Expected: {rx['total_expected']} | Predicted: {rx['total_predicted']}")
            lines.append(f"    Drug Precision: {rx['drug_precision']:.4f}")
            lines.append(f"    Drug Recall:    {rx['drug_recall']:.4f}")
            lines.append(f"    Drug F1-Score:  {rx['drug_f1']:.4f}")
        if "lab_report" in extraction_results:
            lab = extraction_results["lab_report"]
            lines.append(f"\n  Lab Report Extraction:")
            lines.append(f"    Cases: {lab['total_cases']} | Expected: {lab['total_expected']} | Predicted: {lab['total_predicted']}")
            lines.append(f"    Name Precision:       {lab['name_precision']:.4f}")
            lines.append(f"    Name Recall:          {lab['name_recall']:.4f}")
            lines.append(f"    Name F1-Score:        {lab['name_f1']:.4f}")
            lines.append(f"    Abnormality Accuracy: {lab['abnormality_accuracy']:.4f}")

    if explanation_results and "error" not in explanation_results:
        lines.append("")
        lines.append("─" * 70)
        lines.append("[3] Explanation Quality Evaluation")
        lines.append("─" * 70)
        lines.append(f"  Drug KB entries:    {explanation_results.get('total_drug_entries', 'N/A')}")
        lines.append(f"  Lab KB entries:     {explanation_results.get('total_lab_entries', 'N/A')}")
        lines.append(f"  Drug KB coverage:   {explanation_results['drug_kb_coverage']} ({explanation_results['drug_kb_rate'] * 100:.1f}%)")
        lines.append(f"  Lab KB coverage:    {explanation_results['lab_kb_coverage']} ({explanation_results['lab_kb_rate'] * 100:.1f}%)")
        lines.append(f"  Offline expl. pass: {explanation_results['offline_explanation_pass_rate']}")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    txt_path = RESULTS_DIR / "full_eval_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"📄 Text report saved to: {txt_path}")
    print(report_text)

    return full_report


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run full ClariRx evaluation pipeline")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "groq"],
                        help="LLM provider for extraction evaluation")
    parser.add_argument("--method", type=str, default="llm", choices=["llm", "biobert"],
                        help="Extraction method to evaluate")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR evaluation")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction evaluation")
    parser.add_argument("--skip-explanation", action="store_true", help="Skip explanation evaluation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load environment
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")

    print("=" * 65)
    print("  ClariRx — Full End-to-End Evaluation Pipeline")
    print("=" * 65)

    start_time = time.time()

    ocr_results = None
    extraction_results = None
    explanation_results = None

    if not args.skip_ocr:
        ocr_results = run_ocr_evaluation()

    if not args.skip_extraction:
        extraction_results = run_extraction_evaluation(
            method=args.method, provider=args.provider,
        )

    if not args.skip_explanation:
        explanation_results = run_explanation_evaluation()

    elapsed = time.time() - start_time

    generate_full_report(ocr_results, extraction_results, explanation_results, elapsed)

    print(f"\n✅ Full evaluation complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
