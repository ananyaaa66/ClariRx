"""
Unit Tests for Knowledge Base & Explanation Engine (Phase 4)
==============================================================

Tests:
  - Knowledge base drug lookup (exact, fuzzy, prefix stripping, unknown)
  - Knowledge base lab test lookup (exact, alias, fuzzy, unknown)
  - Severity classification for lab values
  - Offline explanation generation (no LLM needed)
  - Output schema validation

Usage:
    python -m pytest backend/tests/test_explanation.py -v
"""

import sys
import os
from pathlib import Path

import pytest

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge Base Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMedicalKB:
    """Tests for the Medical Knowledge Base."""

    @pytest.fixture
    def kb(self):
        from knowledge_base.build_kb import MedicalKB
        return MedicalKB()

    # ── Drug Lookup ────────────────────────────────────────────────────────

    def test_drug_exact_generic_match(self, kb):
        result = kb.lookup_drug("Amoxicillin")
        assert result is not None
        assert result["generic_name"] == "Amoxicillin"
        assert "Antibiotic" in result["drug_class"]

    def test_drug_exact_brand_match(self, kb):
        result = kb.lookup_drug("Cetzine")
        assert result is not None
        assert result["generic_name"] == "Cetirizine"

    def test_drug_with_dosage(self, kb):
        result = kb.lookup_drug("Amoxicillin 500mg")
        assert result is not None
        assert result["generic_name"] == "Amoxicillin"

    def test_drug_with_tab_prefix(self, kb):
        result = kb.lookup_drug("Tab Paracetamol")
        assert result is not None
        assert result["generic_name"] == "Paracetamol"

    def test_drug_with_cap_prefix_and_dosage(self, kb):
        result = kb.lookup_drug("Cap Omeprazole 20mg")
        assert result is not None
        assert result["generic_name"] == "Omeprazole"

    def test_drug_fuzzy_ocr_typo(self, kb):
        result = kb.lookup_drug("Am0xicillin")
        assert result is not None
        assert result["generic_name"] == "Amoxicillin"

    def test_drug_brand_alatrol(self, kb):
        result = kb.lookup_drug("Alatrol 10mg")
        assert result is not None
        assert result["generic_name"] == "Cetirizine"

    def test_drug_brand_napa(self, kb):
        result = kb.lookup_drug("Napa 650mg")
        assert result is not None
        assert result["generic_name"] == "Paracetamol"

    def test_drug_unknown_returns_none(self, kb):
        result = kb.lookup_drug("UnknownDrug123")
        assert result is None

    def test_drug_has_required_fields(self, kb):
        result = kb.lookup_drug("Amoxicillin")
        assert result is not None
        required_fields = ["generic_name", "brand_names", "drug_class", "use", "side_effects", "warnings"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    # ── Lab Test Lookup ────────────────────────────────────────────────────

    def test_lab_exact_match(self, kb):
        result = kb.lookup_lab_test("Haemoglobin (Hb)")
        assert result is not None
        assert result["test_name"] == "Haemoglobin (Hb)"

    def test_lab_alias_match(self, kb):
        result = kb.lookup_lab_test("HbA1c")
        assert result is not None
        assert result["test_name"] == "HbA1c"

    def test_lab_alias_tsh(self, kb):
        result = kb.lookup_lab_test("TSH")
        assert result is not None
        assert result["category"] == "THYROID"

    def test_lab_partial_match(self, kb):
        result = kb.lookup_lab_test("Total Cholesterol")
        assert result is not None
        assert result["category"] == "LIPID"

    def test_lab_has_required_fields(self, kb):
        result = kb.lookup_lab_test("Haemoglobin (Hb)")
        assert result is not None
        required_fields = [
            "test_name", "category", "what_it_measures", "unit",
            "normal_range_male", "normal_range_female", "high_means", "low_means",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    # ── Database Size ──────────────────────────────────────────────────────

    def test_drug_db_has_entries(self, kb):
        assert len(kb.drug_db) >= 40, "Drug DB should have at least 40 entries"

    def test_lab_db_has_entries(self, kb):
        assert len(kb.lab_db) >= 20, "Lab DB should have at least 20 entries"


# ═══════════════════════════════════════════════════════════════════════════════
# Severity Classification Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeverityClassification:
    """Tests for lab value severity classification."""

    def test_normal_value(self):
        from explanation.explain_lab_value import classify_severity
        assert classify_severity(14.0, 13.0, 17.0) == "normal"

    def test_normal_at_lower_bound(self):
        from explanation.explain_lab_value import classify_severity
        assert classify_severity(13.0, 13.0, 17.0) == "normal"

    def test_normal_at_upper_bound(self):
        from explanation.explain_lab_value import classify_severity
        assert classify_severity(17.0, 13.0, 17.0) == "normal"

    def test_borderline_high(self):
        from explanation.explain_lab_value import classify_severity
        # 17.3 is within 10% of range width (0.4) above the upper bound
        assert classify_severity(17.3, 13.0, 17.0) == "borderline"

    def test_borderline_low(self):
        from explanation.explain_lab_value import classify_severity
        # 12.7 is within 10% of range width (0.4) below the lower bound
        assert classify_severity(12.7, 13.0, 17.0) == "borderline"

    def test_abnormal_high(self):
        from explanation.explain_lab_value import classify_severity
        assert classify_severity(20.0, 13.0, 17.0) == "abnormal"

    def test_abnormal_low(self):
        from explanation.explain_lab_value import classify_severity
        assert classify_severity(8.0, 13.0, 17.0) == "abnormal"


# ═══════════════════════════════════════════════════════════════════════════════
# Offline Explanation Tests (no LLM required)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOfflineExplanation:
    """Tests for offline (non-LLM) explanation generation."""

    def test_medicine_offline_with_kb(self):
        from explanation.explain_medicine import _offline_explanation
        from knowledge_base.build_kb import MedicalKB

        kb = MedicalKB()
        drug_info = kb.lookup_drug("Paracetamol")

        result = _offline_explanation(
            drug_name="Paracetamol 650mg",
            drug_info=drug_info,
            frequency="SOS",
            duration="",
            instructions="after food",
        )
        assert result.drug_name == "Paracetamol 650mg"
        assert result.generic_name == "Paracetamol"
        assert result.kb_grounded is True
        assert len(result.explanation_en) > 20
        assert result.severity == "safe"

    def test_medicine_offline_without_kb(self):
        from explanation.explain_medicine import _offline_explanation

        result = _offline_explanation(
            drug_name="UnknownPill 100mg",
            drug_info=None,
            frequency="1-0-1",
            duration="7 days",
            instructions="",
        )
        assert result.drug_name == "UnknownPill 100mg"
        assert result.kb_grounded is False
        assert "doctor" in result.explanation_en.lower() or "pharmacist" in result.explanation_en.lower()

    def test_lab_offline_normal(self):
        from explanation.explain_lab_value import _offline_explanation
        from knowledge_base.build_kb import MedicalKB

        kb = MedicalKB()
        test_info = kb.lookup_lab_test("Haemoglobin")

        result = _offline_explanation(
            test_name="Haemoglobin",
            value="14.5",
            unit="g/dL",
            normal_range="13.0 - 17.0 g/dL",
            severity="normal",
            test_info=test_info,
        )
        assert result.severity == "normal"
        assert result.kb_grounded is True
        assert "normal" in result.explanation_en.lower()

    def test_lab_offline_abnormal(self):
        from explanation.explain_lab_value import _offline_explanation
        from knowledge_base.build_kb import MedicalKB

        kb = MedicalKB()
        test_info = kb.lookup_lab_test("Haemoglobin")

        result = _offline_explanation(
            test_name="Haemoglobin",
            value="8.5",
            unit="g/dL",
            normal_range="13.0 - 17.0 g/dL",
            severity="abnormal",
            test_info=test_info,
        )
        assert result.severity == "abnormal"
        assert "doctor" in result.explanation_en.lower() or "consult" in result.explanation_en.lower()

    def test_lab_offline_unknown_test(self):
        from explanation.explain_lab_value import _offline_explanation

        result = _offline_explanation(
            test_name="UnknownTest",
            value="42",
            unit="mg/dL",
            normal_range="Not available",
            severity="normal",
            test_info=None,
        )
        assert result.kb_grounded is False
        assert "doctor" in result.explanation_en.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Schema Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemas:
    """Tests for Pydantic output schema validation."""

    def test_medicine_explanation_schema(self):
        from explanation.explain_medicine import MedicineExplanation

        explanation = MedicineExplanation(
            drug_name="Amoxicillin 500mg",
            generic_name="Amoxicillin",
            drug_class="Antibiotic",
            explanation_en="This is a test.",
            explanation_hi="",
            severity="safe",
            kb_grounded=True,
        )
        dumped = explanation.model_dump()
        assert "drug_name" in dumped
        assert "explanation_en" in dumped
        assert dumped["severity"] == "safe"

    def test_lab_explanation_schema(self):
        from explanation.explain_lab_value import LabExplanation

        explanation = LabExplanation(
            test_name="Haemoglobin",
            patient_value="14.2",
            unit="g/dL",
            normal_range="13.0 - 17.0 g/dL",
            severity="normal",
            explanation_en="Your haemoglobin is normal.",
            explanation_hi="",
            kb_grounded=True,
        )
        dumped = explanation.model_dump()
        assert "test_name" in dumped
        assert "severity" in dumped
        assert dumped["severity"] == "normal"
