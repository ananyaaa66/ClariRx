"""
Lab Value Explanation Engine for ClariRx
==========================================

Generates simple, bilingual (English + Hindi) explanations for
lab test results by grounding LLM output in the curated Medical
Knowledge Base, with severity classification.

Flow:
  1. Look up extracted lab test name in KB (fuzzy match)
  2. Determine severity: Normal / Borderline / Abnormal
  3. Build a grounded prompt with KB context + patient values
  4. Call the LLM Orchestrator to generate the explanation
  5. Return structured LabExplanation with severity + bilingual text

Usage (standalone test):
    python explain_lab_value.py

Usage (as module):
    from explanation.explain_lab_value import explain_lab_value
    result = explain_lab_value("Haemoglobin", value="14.2", unit="g/dL")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(BACKEND_DIR))


# ─── Output Schema ─────────────────────────────────────────────────────────────

class LabExplanation(BaseModel):
    """Structured explanation for a lab test result."""
    test_name: str = Field(..., description="Lab test name")
    patient_value: str = Field(..., description="Patient's measured value")
    unit: str = Field("", description="Unit of measurement")
    normal_range: str = Field("", description="Normal range as string, e.g. '13.0 - 17.0 g/dL'")
    severity: str = Field("normal", description="normal | borderline | abnormal")
    explanation_en: str = Field(..., description="Simple English explanation")
    explanation_hi: str = Field("", description="Simple Hindi explanation")
    kb_grounded: bool = Field(False, description="Whether explanation was grounded in KB data")


# ─── Severity Classification ──────────────────────────────────────────────────

def classify_severity(
    value: float,
    range_low: float,
    range_high: float,
    borderline_pct: float = 0.10,
) -> str:
    """
    Classify a lab value into severity categories.

    Args:
        value: Patient's measured value.
        range_low: Lower bound of normal range.
        range_high: Upper bound of normal range.
        borderline_pct: Percentage beyond range to consider "borderline" (default 10%).

    Returns:
        "normal", "borderline", or "abnormal"
    """
    if range_low <= value <= range_high:
        return "normal"

    range_width = range_high - range_low
    borderline_margin = range_width * borderline_pct

    # Check borderline zones
    if value < range_low:
        if value >= range_low - borderline_margin:
            return "borderline"
        return "abnormal"
    else:  # value > range_high
        if value <= range_high + borderline_margin:
            return "borderline"
        return "abnormal"


def _get_status_text(severity: str, value: float, range_low: float, range_high: float) -> str:
    """Generate human-readable status text."""
    if severity == "normal":
        return "Normal"
    elif severity == "borderline":
        if value < range_low:
            return "Borderline Low"
        else:
            return "Borderline High"
    else:
        if value < range_low:
            return "Low"
        else:
            return "High"


# ─── Prompt Builder ────────────────────────────────────────────────────────────

LAB_PROMPT_TEMPLATE = """
You are a friendly medical assistant for ClariRx, an app for elderly patients and families in India.
Your job is to explain a lab test result in simple, everyday language.

Here is verified information about this test from our medical database:
- Test Name: {test_name}
- What it measures: {what_it_measures}
- Normal Range: {normal_range_low} - {normal_range_high} {unit}
- What high values mean: {high_means}
- What low values mean: {low_means}

The patient's result:
- Value: {patient_value} {unit}
- Status: {status}

Please generate a clear, caring explanation in the following JSON format only, no extra text:
{{
  "explanation_en": "A 3-4 sentence explanation in simple English. Include: what this test checks, whether the value is normal or not, what it might mean, and one suggestion.",
  "explanation_hi": "The same explanation in simple Hindi (Devanagari script).",
  "severity": "{severity}"
}}

Rules:
- Do NOT use medical jargon.
- Do NOT diagnose conditions.
- For abnormal values, ALWAYS recommend consulting the doctor.
- Be warm and caring.
"""

FALLBACK_LAB_PROMPT = """
You are a friendly medical assistant for ClariRx, an app for elderly patients in India.

The patient's lab test result:
- Test: {test_name}
- Value: {patient_value} {unit}

We don't have reference ranges for this test. Please provide a careful, general explanation.

Generate ONLY valid JSON:
{{
  "explanation_en": "A 2-3 sentence explanation. Mention that the patient should discuss this result with their doctor. Be helpful but cautious.",
  "explanation_hi": "The same explanation in simple Hindi.",
  "severity": "normal"
}}
"""


# ─── Main Explanation Function ─────────────────────────────────────────────────

def explain_lab_value(
    test_name: str,
    value: str,
    unit: str = "",
    normal_range_low: Optional[float] = None,
    normal_range_high: Optional[float] = None,
    gender: str = "male",
    model_provider: str = "gemini",
) -> LabExplanation:
    """
    Generate a simple, bilingual explanation for a lab test result.

    Args:
        test_name: Name of the lab test.
        value: Patient's measured value (string).
        unit: Unit of measurement.
        normal_range_low: Override for lower normal range (from extraction).
        normal_range_high: Override for upper normal range (from extraction).
        gender: "male" or "female" for gender-specific ranges.
        model_provider: LLM provider.

    Returns:
        LabExplanation with severity + bilingual text.
    """
    from knowledge_base.build_kb import MedicalKB
    from extraction.llm_orchestrator import LLMOrchestrator

    kb = MedicalKB()
    test_info = kb.lookup_lab_test(test_name)

    # Try to parse numeric value
    try:
        numeric_value = float(value.replace(",", ""))
    except (ValueError, TypeError):
        numeric_value = None

    if test_info:
        # Use KB ranges, with overrides from extraction
        range_key = f"normal_range_{gender}"
        kb_range = test_info.get(range_key, test_info.get("normal_range_male", [0, 0]))
        range_low = normal_range_low if normal_range_low is not None else kb_range[0]
        range_high = normal_range_high if normal_range_high is not None else kb_range[1]

        # Classify severity
        if numeric_value is not None:
            severity = classify_severity(numeric_value, range_low, range_high)
            status = _get_status_text(severity, numeric_value, range_low, range_high)
        else:
            severity = "normal"
            status = "Unable to determine (non-numeric value)"

        use_unit = unit or test_info.get("unit", "")
        normal_range_str = f"{range_low} - {range_high} {use_unit}"

        # Build grounded prompt
        prompt = LAB_PROMPT_TEMPLATE.format(
            test_name=test_info["test_name"],
            what_it_measures=test_info["what_it_measures"],
            normal_range_low=range_low,
            normal_range_high=range_high,
            unit=use_unit,
            high_means=test_info.get("high_means", "Consult your doctor."),
            low_means=test_info.get("low_means", "Consult your doctor."),
            patient_value=value,
            status=status,
            severity=severity,
        )
        kb_grounded = True
        logger.info(f"KB match: '{test_name}' -> {test_info['test_name']} | Status: {status}")

    else:
        # Fallback: no KB match
        prompt = FALLBACK_LAB_PROMPT.format(
            test_name=test_name,
            patient_value=value,
            unit=unit,
        )
        severity = "normal"
        normal_range_str = "Not available"
        use_unit = unit
        kb_grounded = False
        logger.warning(f"No KB match for lab test '{test_name}' - using fallback")

    # Call LLM
    orchestrator = LLMOrchestrator()
    try:
        response = _call_llm_for_explanation(orchestrator, prompt, model_provider)
        return LabExplanation(
            test_name=test_name,
            patient_value=value,
            unit=use_unit,
            normal_range=normal_range_str,
            severity=response.get("severity", severity),
            explanation_en=response.get("explanation_en", "Please consult your doctor for details about this test result."),
            explanation_hi=response.get("explanation_hi", ""),
            kb_grounded=kb_grounded,
        )
    except Exception as e:
        logger.error(f"LLM explanation failed: {e}")
        return _offline_explanation(test_name, value, use_unit, normal_range_str, severity, test_info)


def _call_llm_for_explanation(orchestrator, prompt: str, model_provider: str) -> dict:
    """Call LLM with explanation prompt and parse JSON response."""
    import extraction.llm_orchestrator as orch_module

    original_prompt = orch_module.SYSTEM_PROMPT
    orch_module.SYSTEM_PROMPT = prompt

    try:
        response = orchestrator.extract("Generate the explanation now.", model_provider=model_provider)
        return response
    finally:
        orch_module.SYSTEM_PROMPT = original_prompt


def _offline_explanation(
    test_name: str,
    value: str,
    unit: str,
    normal_range: str,
    severity: str,
    test_info: Optional[dict],
) -> LabExplanation:
    """Generate a basic explanation without LLM, using only KB data."""
    if test_info:
        if severity == "normal":
            en = (
                f"Your {test_info['test_name']} is {value} {unit}, which is within the normal range ({normal_range}). "
                f"This test measures {test_info['what_it_measures'].lower()} "
                f"Your result looks good."
            )
        elif severity == "borderline":
            en = (
                f"Your {test_info['test_name']} is {value} {unit}, which is slightly outside the normal range ({normal_range}). "
                f"This is borderline and may not be a concern, but please discuss it with your doctor "
                f"during your next visit."
            )
        else:
            en = (
                f"Your {test_info['test_name']} is {value} {unit}, which is outside the normal range ({normal_range}). "
                f"Please consult your doctor to discuss this result. "
                f"They can advise you on any needed follow-up."
            )
    else:
        en = (
            f"Your {test_name} result is {value} {unit}. "
            f"Please discuss this result with your doctor for proper interpretation."
        )

    return LabExplanation(
        test_name=test_name,
        patient_value=value,
        unit=unit,
        normal_range=normal_range,
        severity=severity,
        explanation_en=en,
        explanation_hi="",
        kb_grounded=test_info is not None,
    )


# ─── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BACKEND_DIR, ".env"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_cases = [
        {"test_name": "Haemoglobin", "value": "14.2", "unit": "g/dL"},
        {"test_name": "WBC Count", "value": "11500", "unit": "cells/uL"},
        {"test_name": "Total Cholesterol", "value": "223", "unit": "mg/dL"},
        {"test_name": "HbA1c", "value": "6.8", "unit": "%"},
        {"test_name": "Platelet Count", "value": "0.88", "unit": "lakhs/uL"},
    ]

    for case in test_cases:
        print("=" * 60)
        print(f"Test: {case['test_name']} = {case['value']} {case['unit']}")
        print("=" * 60)
        try:
            result = explain_lab_value(**case)
            print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {e}")
        print()
