"""
LLM-Based Medical Information Extraction
==========================================

Wraps the LLM Orchestrator (Gemini / Groq) with:
  - Separate system prompts for prescriptions vs. lab reports
  - Pydantic validation and normalization of LLM JSON output
  - Graceful handling of partial / malformed responses

Usage (standalone test):
    python llm_extraction.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import List

logger = logging.getLogger(__name__)

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.extract import (
    DocType,
    ExtractionMethod,
    ExtractionResult,
    LabReportItem,
    PrescriptionItem,
)

# ─── System Prompts ────────────────────────────────────────────────────────────

PRESCRIPTION_SYSTEM_PROMPT = """
You are a medical extraction assistant for ClariRx, an app that helps elderly patients in India understand their prescriptions.
Your task is to take raw, messy OCR text from a handwritten prescription and extract the medications.

You must output ONLY valid JSON matching this exact structure, with no markdown formatting or extra text:
{
  "type": "prescription",
  "items": [
    {
      "drugName": "Cleaned up drug name with dosage (e.g., Amoxicillin 500mg)",
      "dosage": "Dosage strength (e.g., 500mg)",
      "frequency": "Dosage frequency (e.g., 1-0-1 or SOS)",
      "duration": "Duration (e.g., 5 Days, or 'As needed')",
      "instructions": "Any specific instructions (e.g., Take after meals)",
      "explanationEn": "A simple, jargon-free 1-2 sentence explanation of what the medicine does in English.",
      "explanationHi": "The exact same explanation translated into simple Hindi."
    }
  ]
}

Rules:
- If no medications are found, return {"type": "prescription", "items": []}.
- Be highly forgiving of OCR typos (e.g., "Am0x" -> "Amoxicillin").
- "1-0-1" means morning-afternoon-evening doses. Translate to human-readable text.
- Always include both English and Hindi explanations.
"""

LAB_REPORT_SYSTEM_PROMPT = """
You are a medical extraction assistant for ClariRx, an app that helps elderly patients in India understand their lab reports.
Your task is to take raw OCR text from a lab report and extract test results.

You must output ONLY valid JSON matching this exact structure, with no markdown formatting or extra text:
{
  "type": "lab_report",
  "items": [
    {
      "testName": "Full name of the lab test (e.g., Haemoglobin (Hb))",
      "value": "The measured value as a string (e.g., '14.2')",
      "unit": "Unit of measurement (e.g., g/dL, mg/dL, cells/µL)",
      "normalRangeLow": 13.0,
      "normalRangeHigh": 17.0,
      "isAbnormal": false
    }
  ]
}

Rules:
- If no test results are found, return {"type": "lab_report", "items": []}.
- Be forgiving of OCR typos and formatting issues.
- Mark isAbnormal as true if the value is outside the normal range.
- normalRangeLow and normalRangeHigh should be numbers (not strings). Use null if unknown.
- Common Indian lab report formats use lakhs/µL for platelets — preserve this.
"""


# ─── Core Extraction Function ─────────────────────────────────────────────────

def extract_with_llm(
    raw_text: str,
    doc_type: DocType,
    model_provider: str = "gemini",
) -> ExtractionResult:
    """
    Extract structured medical data from raw OCR text using an LLM.

    Args:
        raw_text: Raw OCR text from the document.
        doc_type: Type of document being processed.
        model_provider: LLM provider — "gemini" or "groq".

    Returns:
        ExtractionResult with structured entities.
    """
    from extraction.llm_orchestrator import LLMOrchestrator

    orchestrator = LLMOrchestrator()

    # Select the appropriate system prompt
    system_prompt = (
        PRESCRIPTION_SYSTEM_PROMPT
        if doc_type == DocType.PRESCRIPTION
        else LAB_REPORT_SYSTEM_PROMPT
    )

    # Call the LLM with the correct prompt
    raw_response = _call_llm(orchestrator, raw_text, system_prompt, model_provider)

    # Parse and validate the response
    if doc_type == DocType.PRESCRIPTION:
        items = _parse_prescription_response(raw_response)
        return ExtractionResult(
            doc_type=doc_type,
            method_used=ExtractionMethod.LLM,
            prescription_items=items,
            raw_text=raw_text,
            success=True,
        )
    else:
        items = _parse_lab_report_response(raw_response)
        return ExtractionResult(
            doc_type=doc_type,
            method_used=ExtractionMethod.LLM,
            lab_report_items=items,
            raw_text=raw_text,
            success=True,
        )


# ─── LLM Call Helpers ──────────────────────────────────────────────────────────

def _call_llm(
    orchestrator,
    raw_text: str,
    system_prompt: str,
    model_provider: str,
) -> dict:
    """
    Call the LLM orchestrator with a custom system prompt.
    Temporarily patches the system prompt for lab report extraction.
    """
    import extraction.llm_orchestrator as orch_module

    # Save original prompt and patch with the appropriate one
    original_prompt = orch_module.SYSTEM_PROMPT
    orch_module.SYSTEM_PROMPT = system_prompt

    try:
        response = orchestrator.extract(raw_text, model_provider=model_provider)
        return response
    finally:
        # Restore original prompt
        orch_module.SYSTEM_PROMPT = original_prompt


# ─── Response Parsers ──────────────────────────────────────────────────────────

def _parse_prescription_response(response: dict) -> List[PrescriptionItem]:
    """Parse and validate the LLM JSON response for prescriptions."""
    items = []
    raw_items = response.get("items", [])

    for raw in raw_items:
        try:
            item = PrescriptionItem(
                drug_name=raw.get("drugName", raw.get("drug_name", "Unknown")),
                dosage=raw.get("dosage"),
                frequency=raw.get("frequency"),
                duration=raw.get("duration"),
                instructions=raw.get("instructions"),
                explanation_en=raw.get("explanationEn", raw.get("explanation_en")),
                explanation_hi=raw.get("explanationHi", raw.get("explanation_hi")),
                confidence=1.0,  # LLM extraction is assumed high-confidence
            )
            items.append(item)
        except Exception as e:
            logger.warning(f"Skipping malformed prescription item: {raw} — {e}")

    return items


def _parse_lab_report_response(response: dict) -> List[LabReportItem]:
    """Parse and validate the LLM JSON response for lab reports."""
    items = []
    raw_items = response.get("items", [])

    for raw in raw_items:
        try:
            # Parse numeric values, handling strings and nulls
            value_str = str(raw.get("value", ""))
            range_low = _safe_float(raw.get("normalRangeLow", raw.get("normal_range_low")))
            range_high = _safe_float(raw.get("normalRangeHigh", raw.get("normal_range_high")))

            # Determine abnormality
            is_abnormal = raw.get("isAbnormal", raw.get("is_abnormal", False))
            if isinstance(is_abnormal, str):
                is_abnormal = is_abnormal.lower() in ("true", "yes", "1")

            # Auto-detect abnormality if not provided but ranges are available
            if range_low is not None and range_high is not None:
                try:
                    numeric_value = float(value_str.replace(",", ""))
                    is_abnormal = numeric_value < range_low or numeric_value > range_high
                except (ValueError, TypeError):
                    pass

            item = LabReportItem(
                test_name=raw.get("testName", raw.get("test_name", "Unknown")),
                value=value_str,
                unit=raw.get("unit"),
                normal_range_low=range_low,
                normal_range_high=range_high,
                is_abnormal=is_abnormal,
                confidence=1.0,
            )
            items.append(item)
        except Exception as e:
            logger.warning(f"Skipping malformed lab report item: {raw} — {e}")

    return items


def _safe_float(value) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ─── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sample_prescription = (
        "Rx\n"
        "1. Tab Amox 500  1-0-1  after food  5d\n"
        "2. Cap Omeprazole 20mg  1-0-0  before food  7d\n"
        "3. Syr Cetzine 5ml  0-0-1  3 days\n"
    )

    sample_lab_report = (
        "LIPID PROFILE\n"
        "Total Cholesterol: 223 mg/dL  (Desirable: < 200)\n"
        "Triglycerides: 173 mg/dL  (Normal: < 150)\n"
        "HDL Cholesterol: 69 mg/dL  (Normal: 50-70)\n"
        "LDL Cholesterol: 131 mg/dL  (Optimal: < 100)\n"
    )

    print("=" * 60)
    print("Testing PRESCRIPTION extraction via LLM")
    print("=" * 60)
    try:
        result = extract_with_llm(sample_prescription, DocType.PRESCRIPTION)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Testing LAB REPORT extraction via LLM")
    print("=" * 60)
    try:
        result = extract_with_llm(sample_lab_report, DocType.LAB_REPORT)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")
