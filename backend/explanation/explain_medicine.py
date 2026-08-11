"""
Medicine Explanation Engine for ClariRx
=========================================

Generates simple, bilingual (English + Hindi) explanations for
prescribed medicines by grounding LLM output in the curated
Medical Knowledge Base.

Flow:
  1. Look up extracted drug name in KB (fuzzy match)
  2. Build a grounded prompt with KB context + prescription details
  3. Call the LLM Orchestrator to generate the explanation
  4. Return structured MedicineExplanation

Usage (standalone test):
    python explain_medicine.py

Usage (as module):
    from explanation.explain_medicine import explain_medicine
    result = explain_medicine("Amoxicillin 500mg", frequency="1-0-1", duration="5 days")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROMPT_PATH = SCRIPT_DIR / "prompts" / "explanation_prompt.txt"

sys.path.insert(0, str(BACKEND_DIR))


# ─── Output Schema ─────────────────────────────────────────────────────────────

class MedicineExplanation(BaseModel):
    """Structured explanation for a prescribed medicine."""
    drug_name: str = Field(..., description="Drug name as prescribed")
    generic_name: Optional[str] = Field(None, description="Generic/scientific name from KB")
    drug_class: Optional[str] = Field(None, description="Drug class from KB")
    explanation_en: str = Field(..., description="Simple English explanation")
    explanation_hi: str = Field("", description="Simple Hindi explanation")
    severity: str = Field("safe", description="Always 'safe' for medicines")
    kb_grounded: bool = Field(False, description="Whether explanation was grounded in KB data")


# ─── Prompt Builder ────────────────────────────────────────────────────────────

MEDICINE_PROMPT_TEMPLATE = """
You are a friendly medical assistant for ClariRx, an app for elderly patients and families in India.
Your job is to explain a prescribed medicine in simple, everyday language.

Here is verified information about this medicine from our medical database:
- Generic Name: {generic_name}
- Drug Class: {drug_class}
- What it does: {use}
- Common Side Effects: {side_effects}
- Important Warnings: {warnings}

The patient has been prescribed:
- Medicine: {prescribed_drug}
- Dosage/Frequency: {frequency}
- Duration: {duration}
- Special Instructions: {instructions}

Please generate a clear, caring explanation in the following JSON format only, no extra text:
{{
  "explanation_en": "A 3-4 sentence explanation in simple English.",
  "explanation_hi": "The same explanation in simple Hindi (Devanagari script).",
  "severity": "safe"
}}

Rules:
- Do NOT use medical jargon.
- Do NOT invent information beyond what is provided above.
- Be warm and caring.
"""

FALLBACK_PROMPT_TEMPLATE = """
You are a friendly medical assistant for ClariRx, an app for elderly patients in India.

The patient has been prescribed: {prescribed_drug}
- Dosage/Frequency: {frequency}
- Duration: {duration}
- Special Instructions: {instructions}

We don't have this medicine in our database. Please provide a careful, general explanation.

Generate ONLY valid JSON:
{{
  "explanation_en": "A 2-3 sentence general explanation. Mention that the patient should confirm details with their doctor or pharmacist. Be helpful but cautious.",
  "explanation_hi": "The same explanation in simple Hindi.",
  "severity": "safe"
}}

Do NOT guess what the medicine treats. Be honest that you're providing general guidance.
"""


def _build_grounded_prompt(
    drug_info: dict,
    prescribed_drug: str,
    frequency: str = "",
    duration: str = "",
    instructions: str = "",
) -> str:
    """Build a KB-grounded explanation prompt."""
    return MEDICINE_PROMPT_TEMPLATE.format(
        generic_name=drug_info.get("generic_name", "Unknown"),
        drug_class=drug_info.get("drug_class", "Unknown"),
        use=drug_info.get("use", "Not available"),
        side_effects=", ".join(drug_info.get("side_effects", ["None listed"])),
        warnings=" ".join(drug_info.get("warnings", ["None"])),
        prescribed_drug=prescribed_drug,
        frequency=frequency or "As prescribed",
        duration=duration or "As prescribed",
        instructions=instructions or "Follow doctor's instructions",
    )


def _build_fallback_prompt(
    prescribed_drug: str,
    frequency: str = "",
    duration: str = "",
    instructions: str = "",
) -> str:
    """Build a non-grounded fallback prompt for unknown drugs."""
    return FALLBACK_PROMPT_TEMPLATE.format(
        prescribed_drug=prescribed_drug,
        frequency=frequency or "As prescribed",
        duration=duration or "As prescribed",
        instructions=instructions or "Follow doctor's instructions",
    )


# ─── Main Explanation Function ─────────────────────────────────────────────────

def explain_medicine(
    drug_name: str,
    frequency: str = "",
    duration: str = "",
    instructions: str = "",
    model_provider: str = "gemini",
) -> MedicineExplanation:
    """
    Generate a simple, bilingual explanation for a prescribed medicine.

    Args:
        drug_name: Prescribed drug name (may include dosage).
        frequency: Dosage frequency (e.g., "1-0-1").
        duration: Course duration (e.g., "5 days").
        instructions: Special instructions (e.g., "after food").
        model_provider: LLM provider — "gemini" or "groq".

    Returns:
        MedicineExplanation with English + Hindi text.
    """
    from knowledge_base.build_kb import MedicalKB
    from extraction.llm_orchestrator import LLMOrchestrator

    kb = MedicalKB()
    drug_info = kb.lookup_drug(drug_name)

    if drug_info:
        # Grounded explanation
        prompt = _build_grounded_prompt(
            drug_info, drug_name, frequency, duration, instructions
        )
        generic_name = drug_info["generic_name"]
        drug_class = drug_info["drug_class"]
        kb_grounded = True
        logger.info(f"KB match: '{drug_name}' -> {generic_name}")
    else:
        # Fallback: no KB match
        prompt = _build_fallback_prompt(drug_name, frequency, duration, instructions)
        generic_name = None
        drug_class = None
        kb_grounded = False
        logger.warning(f"No KB match for '{drug_name}' - using fallback prompt")

    # Call LLM
    orchestrator = LLMOrchestrator()
    try:
        response = _call_llm_for_explanation(orchestrator, prompt, model_provider)
        return MedicineExplanation(
            drug_name=drug_name,
            generic_name=generic_name,
            drug_class=drug_class,
            explanation_en=response.get("explanation_en", "Please consult your doctor for details about this medicine."),
            explanation_hi=response.get("explanation_hi", ""),
            severity=response.get("severity", "safe"),
            kb_grounded=kb_grounded,
        )
    except Exception as e:
        logger.error(f"LLM explanation failed: {e}")
        # Return a safe fallback
        return _offline_explanation(drug_name, drug_info, frequency, duration, instructions)


def _call_llm_for_explanation(orchestrator, prompt: str, model_provider: str) -> dict:
    """Call LLM with explanation prompt and parse JSON response."""
    import extraction.llm_orchestrator as orch_module

    original_prompt = orch_module.SYSTEM_PROMPT
    orch_module.SYSTEM_PROMPT = prompt

    try:
        # Use empty text since the full prompt is in the system prompt
        response = orchestrator.extract("Generate the explanation now.", model_provider=model_provider)
        return response
    finally:
        orch_module.SYSTEM_PROMPT = original_prompt


def _offline_explanation(
    drug_name: str,
    drug_info: Optional[dict],
    frequency: str,
    duration: str,
    instructions: str,
) -> MedicineExplanation:
    """
    Generate a basic explanation without LLM, using only KB data.
    Used as a fallback when LLM is unavailable.
    """
    if drug_info:
        en = (
            f"{drug_info['generic_name']} is a {drug_info['drug_class'].lower()}. "
            f"{drug_info['use']} "
            f"Take as prescribed by your doctor"
        )
        if frequency:
            en += f" ({frequency})"
        if duration:
            en += f" for {duration}"
        en += "."
        if drug_info.get("warnings"):
            en += f" Important: {drug_info['warnings'][0]}"

        return MedicineExplanation(
            drug_name=drug_name,
            generic_name=drug_info["generic_name"],
            drug_class=drug_info["drug_class"],
            explanation_en=en,
            explanation_hi="",
            severity="safe",
            kb_grounded=True,
        )
    else:
        return MedicineExplanation(
            drug_name=drug_name,
            explanation_en=(
                f"You have been prescribed {drug_name}. "
                f"Please follow your doctor's instructions carefully. "
                f"If you have any questions, ask your pharmacist or doctor."
            ),
            explanation_hi="",
            severity="safe",
            kb_grounded=False,
        )


# ─── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BACKEND_DIR, ".env"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_cases = [
        {
            "drug_name": "Tab Amoxicillin 500mg",
            "frequency": "1-0-1",
            "duration": "5 days",
            "instructions": "after food",
        },
        {
            "drug_name": "Napa 650mg",
            "frequency": "SOS",
            "duration": "",
            "instructions": "",
        },
        {
            "drug_name": "UnknownMedicine XR 200mg",
            "frequency": "1-0-0",
            "duration": "14 days",
            "instructions": "before food",
        },
    ]

    for case in test_cases:
        print("=" * 60)
        print(f"Medicine: {case['drug_name']}")
        print("=" * 60)
        try:
            result = explain_medicine(**case)
            print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {e}")
        print()
