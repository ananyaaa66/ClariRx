"""
Unified Extraction Entry Point for ClariRx
============================================

Routes raw OCR text to the appropriate extraction backend (LLM or BioBERT)
and returns structured medical entities in a unified Pydantic schema.

Usage (standalone test):
    python extract.py

Usage (as module):
    from extraction.extract import run_extraction
    result = run_extraction(raw_text, doc_type="prescription", method="llm")
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Enums ─────────────────────────────────────────────────────────────────────

class DocType(str, Enum):
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"


class ExtractionMethod(str, Enum):
    LLM = "llm"
    BIOBERT = "biobert"


# ─── Pydantic Models (unified output schema) ──────────────────────────────────

class PrescriptionItem(BaseModel):
    """A single medication extracted from a prescription."""
    drug_name: str = Field(..., description="Cleaned drug name with strength, e.g. 'Amoxicillin 500mg'")
    dosage: Optional[str] = Field(None, description="Dosage strength if separate from name, e.g. '500mg'")
    frequency: Optional[str] = Field(None, description="Dosage frequency, e.g. '1-0-1' or 'SOS'")
    duration: Optional[str] = Field(None, description="Course duration, e.g. '5 Days'")
    instructions: Optional[str] = Field(None, description="Timing / special instructions, e.g. 'Take after meals'")
    explanation_en: Optional[str] = Field(None, description="Plain-English explanation of the medicine")
    explanation_hi: Optional[str] = Field(None, description="Plain-Hindi explanation of the medicine")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence score")


class LabReportItem(BaseModel):
    """A single test result extracted from a lab report."""
    test_name: str = Field(..., description="Name of the lab test, e.g. 'Haemoglobin (Hb)'")
    value: str = Field(..., description="Measured value, e.g. '14.2'")
    unit: Optional[str] = Field(None, description="Unit of measurement, e.g. 'g/dL'")
    normal_range_low: Optional[float] = Field(None, description="Lower bound of normal range")
    normal_range_high: Optional[float] = Field(None, description="Upper bound of normal range")
    is_abnormal: bool = Field(False, description="Whether the value is outside the normal range")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence score")


class ExtractionResult(BaseModel):
    """Unified extraction output returned by all backends."""
    doc_type: DocType
    method_used: ExtractionMethod
    prescription_items: List[PrescriptionItem] = Field(default_factory=list)
    lab_report_items: List[LabReportItem] = Field(default_factory=list)
    raw_text: str = Field("", description="Original OCR text that was processed")
    success: bool = Field(True)
    error_message: Optional[str] = None


# ─── Main Extraction Router ───────────────────────────────────────────────────

def run_extraction(
    raw_text: str,
    doc_type: str = "prescription",
    method: str = "llm",
    model_provider: str = "gemini",
    fallback: bool = True,
) -> ExtractionResult:
    """
    Main extraction entry point. Routes to LLM or BioBERT based on `method`.

    Args:
        raw_text: Raw OCR text from the document.
        doc_type: One of "prescription" or "lab_report".
        method: Extraction method — "llm" or "biobert".
        model_provider: LLM provider for LLM method — "gemini" or "groq".
        fallback: If True and BioBERT fails, automatically retry with LLM.

    Returns:
        ExtractionResult with structured medical entities.
    """
    dtype = DocType(doc_type)
    preferred_method = ExtractionMethod(method)

    # ── Try preferred method ────────────────────────────────────────────────
    if preferred_method == ExtractionMethod.LLM:
        return _extract_with_llm(raw_text, dtype, model_provider)

    elif preferred_method == ExtractionMethod.BIOBERT:
        result = _extract_with_biobert(raw_text, dtype)

        # Fallback to LLM if BioBERT produced empty or low-confidence results
        if fallback and _should_fallback(result):
            logger.warning(
                "BioBERT extraction returned low-confidence results. "
                "Falling back to LLM extraction."
            )
            return _extract_with_llm(raw_text, dtype, model_provider)

        return result

    else:
        return ExtractionResult(
            doc_type=dtype,
            method_used=preferred_method,
            raw_text=raw_text,
            success=False,
            error_message=f"Unknown extraction method: {method}",
        )


# ─── LLM Extraction ───────────────────────────────────────────────────────────

def _extract_with_llm(
    raw_text: str,
    doc_type: DocType,
    model_provider: str = "gemini",
) -> ExtractionResult:
    """Run extraction using the LLM pipeline."""
    try:
        from extraction.llm_extraction import extract_with_llm
        return extract_with_llm(raw_text, doc_type, model_provider)
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return ExtractionResult(
            doc_type=doc_type,
            method_used=ExtractionMethod.LLM,
            raw_text=raw_text,
            success=False,
            error_message=str(e),
        )


# ─── BioBERT Extraction ───────────────────────────────────────────────────────

def _extract_with_biobert(
    raw_text: str,
    doc_type: DocType,
) -> ExtractionResult:
    """Run extraction using the fine-tuned BioBERT NER model."""
    try:
        from extraction.biobert_ner.predict import extract_with_biobert
        return extract_with_biobert(raw_text, doc_type)
    except Exception as e:
        logger.error(f"BioBERT extraction failed: {e}")
        return ExtractionResult(
            doc_type=doc_type,
            method_used=ExtractionMethod.BIOBERT,
            raw_text=raw_text,
            success=False,
            error_message=str(e),
        )


# ─── Fallback Heuristic ───────────────────────────────────────────────────────

def _should_fallback(result: ExtractionResult) -> bool:
    """Determine if BioBERT results are too poor and we should fallback to LLM."""
    if not result.success:
        return True

    # No items extracted at all
    items = result.prescription_items or result.lab_report_items
    if not items:
        return True

    # Average confidence below threshold
    confidences = [item.confidence for item in items]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    if avg_confidence < 0.5:
        return True

    return False


# ─── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    import os

    # Allow running from backend/ directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sample_prescription = (
        "Rx\n"
        "Patient: Mr. Sharma\n"
        "1. Tab Amoxicillin 500mg  1-0-1  after food  x 5 days\n"
        "2. Tab Paracetamol 650mg  SOS\n"
        "3. Syr Cetirizine 5ml  0-0-1  x 3 days\n"
    )

    sample_lab_report = (
        "Complete Blood Count (CBC)\n"
        "Haemoglobin: 14.2 g/dL  (Normal: 13.0 - 17.0)\n"
        "WBC Count: 11500 cells/uL  (Normal: 4000 - 11000)\n"
        "Platelet Count: 2.5 lakhs/uL  (Normal: 1.5 - 4.0)\n"
    )

    print("=" * 60)
    print("Testing PRESCRIPTION extraction (LLM)")
    print("=" * 60)
    result = run_extraction(sample_prescription, doc_type="prescription", method="llm")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Testing LAB REPORT extraction (LLM)")
    print("=" * 60)
    result = run_extraction(sample_lab_report, doc_type="lab_report", method="llm")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
