"""
Unit Tests for Extraction Pipeline (Phase 4)
===============================================

Tests:
  - Extraction router: run_extraction routing, fallback logic, invalid method
  - Pydantic schemas: PrescriptionItem, LabReportItem, ExtractionResult
  - LLM response parsing: prescription & lab report JSON parsing, malformed data
  - BioBERT NER entity grouping: BIO tag grouping, _split_by_drug logic
  - Fallback heuristic: _should_fallback edge cases

All tests use mocking — no real LLM/model calls are made.

Usage:
    python -m pytest backend/tests/test_extraction.py -v
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Schema Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractionSchemas:
    """Tests for extraction Pydantic model validation."""

    def test_prescription_item_required_fields(self):
        """Test PrescriptionItem requires drug_name."""
        from extraction.extract import PrescriptionItem

        item = PrescriptionItem(drug_name="Amoxicillin 500mg")
        assert item.drug_name == "Amoxicillin 500mg"
        assert item.confidence == 1.0  # default

    def test_prescription_item_all_fields(self):
        """Test PrescriptionItem with all optional fields populated."""
        from extraction.extract import PrescriptionItem

        item = PrescriptionItem(
            drug_name="Amoxicillin 500mg",
            dosage="500mg",
            frequency="1-0-1",
            duration="5 Days",
            instructions="Take after meals",
            explanation_en="An antibiotic.",
            explanation_hi="एक एंटीबायोटिक।",
            confidence=0.95,
        )
        assert item.frequency == "1-0-1"
        assert item.duration == "5 Days"
        assert item.confidence == 0.95

    def test_lab_report_item_required_fields(self):
        """Test LabReportItem requires test_name and value."""
        from extraction.extract import LabReportItem

        item = LabReportItem(test_name="Haemoglobin", value="14.2")
        assert item.test_name == "Haemoglobin"
        assert item.value == "14.2"
        assert item.is_abnormal is False  # default

    def test_lab_report_item_abnormal(self):
        """Test LabReportItem with abnormal flag and ranges."""
        from extraction.extract import LabReportItem

        item = LabReportItem(
            test_name="WBC Count",
            value="11500",
            unit="cells/uL",
            normal_range_low=4000.0,
            normal_range_high=11000.0,
            is_abnormal=True,
            confidence=0.98,
        )
        assert item.is_abnormal is True
        assert item.normal_range_low == 4000.0

    def test_extraction_result_defaults(self):
        """Test ExtractionResult default values."""
        from extraction.extract import ExtractionResult, DocType, ExtractionMethod

        result = ExtractionResult(
            doc_type=DocType.PRESCRIPTION,
            method_used=ExtractionMethod.LLM,
        )
        assert result.success is True
        assert result.prescription_items == []
        assert result.lab_report_items == []
        assert result.error_message is None

    def test_extraction_result_with_error(self):
        """Test ExtractionResult with failure state."""
        from extraction.extract import ExtractionResult, DocType, ExtractionMethod

        result = ExtractionResult(
            doc_type=DocType.PRESCRIPTION,
            method_used=ExtractionMethod.BIOBERT,
            success=False,
            error_message="Model checkpoint not found",
        )
        assert result.success is False
        assert "checkpoint" in result.error_message


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction Router Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractionRouter:
    """Tests for the main extraction routing logic."""

    def test_run_extraction_routes_to_llm(self):
        """Test that method='llm' routes to LLM extraction."""
        from extraction.extract import run_extraction, ExtractionMethod

        mock_result = MagicMock()
        mock_result.doc_type = "prescription"
        mock_result.method_used = ExtractionMethod.LLM
        mock_result.success = True

        with patch("extraction.extract._extract_with_llm", return_value=mock_result) as mock_llm:
            result = run_extraction("Rx\nTab Amox 500mg", doc_type="prescription", method="llm")
            mock_llm.assert_called_once()
            assert result == mock_result

    def test_run_extraction_routes_to_biobert(self):
        """Test that method='biobert' routes to BioBERT extraction."""
        from extraction.extract import run_extraction, ExtractionMethod, ExtractionResult, DocType, PrescriptionItem

        mock_result = ExtractionResult(
            doc_type=DocType.PRESCRIPTION,
            method_used=ExtractionMethod.BIOBERT,
            prescription_items=[PrescriptionItem(drug_name="Amoxicillin", confidence=0.9)],
            success=True,
        )

        with patch("extraction.extract._extract_with_biobert", return_value=mock_result):
            result = run_extraction("Rx\nTab Amox 500mg", doc_type="prescription", method="biobert")
            assert result.method_used == ExtractionMethod.BIOBERT
            assert len(result.prescription_items) == 1

    def test_run_extraction_biobert_fallback_to_llm(self):
        """Test that BioBERT falls back to LLM when results are empty."""
        from extraction.extract import run_extraction, ExtractionMethod, ExtractionResult, DocType

        # BioBERT returns empty
        biobert_result = ExtractionResult(
            doc_type=DocType.PRESCRIPTION,
            method_used=ExtractionMethod.BIOBERT,
            prescription_items=[],
            success=True,
        )

        # LLM returns items
        llm_result = ExtractionResult(
            doc_type=DocType.PRESCRIPTION,
            method_used=ExtractionMethod.LLM,
            success=True,
        )

        with patch("extraction.extract._extract_with_biobert", return_value=biobert_result):
            with patch("extraction.extract._extract_with_llm", return_value=llm_result) as mock_llm:
                result = run_extraction(
                    "Rx\nTab Amox 500mg",
                    doc_type="prescription",
                    method="biobert",
                    fallback=True,
                )
                mock_llm.assert_called_once()

    def test_doc_type_enum_validation(self):
        """Test that invalid doc_type raises ValueError."""
        from extraction.extract import run_extraction

        with pytest.raises(ValueError):
            run_extraction("some text", doc_type="invalid_type", method="llm")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Response Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMResponseParsing:
    """Tests for LLM JSON response parsing and normalization."""

    def test_parse_prescription_response_camelcase(self):
        """Test parsing prescription items with camelCase keys (from LLM)."""
        from extraction.llm_extraction import _parse_prescription_response

        response = {
            "items": [
                {
                    "drugName": "Amoxicillin 500mg",
                    "frequency": "1-0-1",
                    "duration": "5 Days",
                    "instructions": "after food",
                    "explanationEn": "An antibiotic.",
                    "explanationHi": "एक एंटीबायोटिक।",
                },
            ]
        }

        items = _parse_prescription_response(response)
        assert len(items) == 1
        assert items[0].drug_name == "Amoxicillin 500mg"
        assert items[0].frequency == "1-0-1"
        assert items[0].explanation_en == "An antibiotic."

    def test_parse_prescription_response_snake_case(self):
        """Test parsing prescription items with snake_case keys."""
        from extraction.llm_extraction import _parse_prescription_response

        response = {
            "items": [
                {
                    "drug_name": "Paracetamol 650mg",
                    "frequency": "SOS",
                    "explanation_en": "Pain reliever.",
                    "explanation_hi": "दर्द निवारक।",
                },
            ]
        }

        items = _parse_prescription_response(response)
        assert len(items) == 1
        assert items[0].drug_name == "Paracetamol 650mg"

    def test_parse_prescription_empty_items(self):
        """Test parsing response with empty items list."""
        from extraction.llm_extraction import _parse_prescription_response

        response = {"items": []}
        items = _parse_prescription_response(response)
        assert items == []

    def test_parse_lab_report_response(self):
        """Test parsing lab report items with abnormality detection."""
        from extraction.llm_extraction import _parse_lab_report_response

        response = {
            "items": [
                {
                    "testName": "Haemoglobin",
                    "value": "14.2",
                    "unit": "g/dL",
                    "normalRangeLow": 13.0,
                    "normalRangeHigh": 17.0,
                    "isAbnormal": False,
                },
                {
                    "testName": "WBC Count",
                    "value": "11500",
                    "unit": "cells/uL",
                    "normalRangeLow": 4000,
                    "normalRangeHigh": 11000,
                    "isAbnormal": True,
                },
            ]
        }

        items = _parse_lab_report_response(response)
        assert len(items) == 2
        assert items[0].test_name == "Haemoglobin"
        assert items[0].is_abnormal is False
        assert items[1].is_abnormal is True

    def test_parse_lab_auto_detect_abnormality(self):
        """Test that abnormality is auto-detected from ranges if not provided."""
        from extraction.llm_extraction import _parse_lab_report_response

        response = {
            "items": [
                {
                    "testName": "WBC",
                    "value": "15000",
                    "unit": "cells/uL",
                    "normalRangeLow": 4000,
                    "normalRangeHigh": 11000,
                    # isAbnormal not provided, should auto-detect
                },
            ]
        }

        items = _parse_lab_report_response(response)
        assert len(items) == 1
        assert items[0].is_abnormal is True  # 15000 > 11000

    def test_parse_malformed_item_skipped(self):
        """Test that malformed items are skipped without crashing."""
        from extraction.llm_extraction import _parse_prescription_response

        response = {
            "items": [
                {"drugName": "Valid Drug 100mg", "frequency": "1-0-1"},
                None,  # This should be skipped gracefully
            ]
        }

        # Should not raise — malformed items are skipped with a warning
        items = _parse_prescription_response(response)
        assert len(items) >= 1

    def test_safe_float_parsing(self):
        """Test _safe_float handles various input types."""
        from extraction.llm_extraction import _safe_float

        assert _safe_float(13.0) == 13.0
        assert _safe_float("13.5") == 13.5
        assert _safe_float(None) is None
        assert _safe_float("not_a_number") is None
        assert _safe_float("") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback Heuristic Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbackHeuristic:
    """Tests for the BioBERT → LLM fallback decision logic."""

    def test_should_fallback_on_failure(self):
        """Test fallback triggers when success=False."""
        from extraction.extract import _should_fallback, ExtractionResult, DocType, ExtractionMethod

        result = ExtractionResult(
            doc_type=DocType.PRESCRIPTION,
            method_used=ExtractionMethod.BIOBERT,
            success=False,
            error_message="Model not found",
        )
        assert _should_fallback(result) is True

    def test_should_fallback_on_empty_items(self):
        """Test fallback triggers when no items are extracted."""
        from extraction.extract import _should_fallback, ExtractionResult, DocType, ExtractionMethod

        result = ExtractionResult(
            doc_type=DocType.PRESCRIPTION,
            method_used=ExtractionMethod.BIOBERT,
            prescription_items=[],
            success=True,
        )
        assert _should_fallback(result) is True

    def test_should_not_fallback_on_good_results(self):
        """Test no fallback when items have high confidence."""
        from extraction.extract import (
            _should_fallback, ExtractionResult, DocType,
            ExtractionMethod, PrescriptionItem,
        )

        result = ExtractionResult(
            doc_type=DocType.PRESCRIPTION,
            method_used=ExtractionMethod.BIOBERT,
            prescription_items=[
                PrescriptionItem(drug_name="Amoxicillin 500mg", confidence=0.9),
                PrescriptionItem(drug_name="Paracetamol 650mg", confidence=0.85),
            ],
            success=True,
        )
        assert _should_fallback(result) is False

    def test_should_fallback_on_low_confidence(self):
        """Test fallback triggers when average confidence is below 0.5."""
        from extraction.extract import (
            _should_fallback, ExtractionResult, DocType,
            ExtractionMethod, PrescriptionItem,
        )

        result = ExtractionResult(
            doc_type=DocType.PRESCRIPTION,
            method_used=ExtractionMethod.BIOBERT,
            prescription_items=[
                PrescriptionItem(drug_name="???", confidence=0.2),
                PrescriptionItem(drug_name="Unknown", confidence=0.3),
            ],
            success=True,
        )
        assert _should_fallback(result) is True


# ═══════════════════════════════════════════════════════════════════════════════
# BioBERT Entity Grouping Tests
# ═══════════════════════════════════════════════════════════════════════════════

# BioBERT predict module requires torch at import time.
# Skip these tests gracefully if torch is not installed.
torch = pytest.importorskip("torch", reason="torch required for BioBERT tests")


class TestBioBERTEntityGrouping:
    """Tests for BIO tag grouping logic in the BioBERT NER predict module."""

    def test_split_by_drug_single_item(self):
        """Test entity grouping with a single drug."""
        from extraction.biobert_ner.predict import _split_by_drug

        entities = [
            {"text": "Amoxicillin", "label": "B-DRUG", "confidence": 0.95},
            {"text": "500mg", "label": "I-DRUG", "confidence": 0.90},
            {"text": "1-0-1", "label": "B-FREQUENCY", "confidence": 0.88},
        ]

        groups = _split_by_drug(entities)
        assert len(groups) == 1
        assert "Amoxicillin 500mg" in groups[0]["DRUG"]
        assert "FREQUENCY" in groups[0]

    def test_split_by_drug_multiple_items(self):
        """Test entity grouping with multiple drugs."""
        from extraction.biobert_ner.predict import _split_by_drug

        entities = [
            {"text": "Amoxicillin", "label": "B-DRUG", "confidence": 0.95},
            {"text": "500mg", "label": "B-DOSAGE", "confidence": 0.90},
            {"text": "1-0-1", "label": "B-FREQUENCY", "confidence": 0.88},
            {"text": "Paracetamol", "label": "B-DRUG", "confidence": 0.92},
            {"text": "SOS", "label": "B-FREQUENCY", "confidence": 0.85},
        ]

        groups = _split_by_drug(entities)
        assert len(groups) == 2
        assert groups[0]["DRUG"] == "Amoxicillin"
        assert groups[1]["DRUG"] == "Paracetamol"

    def test_group_entities_empty(self):
        """Test grouping with no entities returns empty list."""
        from extraction.biobert_ner.predict import group_entities

        groups = group_entities([])
        assert groups == []
