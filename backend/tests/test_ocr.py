"""
Unit Tests for OCR Pipelines (Phase 3)
========================================

Tests:
  - PaddleOCR pipeline: initialization, line sorting, confidence filtering,
    line grouping, full text reconstruction
  - LLM Vision pipeline: initialization, supported formats, prompt construction,
    response handling, error cases
  - OCR evaluation metrics: CER, WER, exact match

All tests use mocking — no real model/API calls are made.

Usage:
    python -m pytest backend/tests/test_ocr.py -v
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# PaddleOCR Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaddleOCRPipeline:
    """Tests for the PaddleOCR-based prescription OCR pipeline."""

    def test_detection_parsing(self):
        """Test that raw PaddleOCR output is parsed into structured detections."""
        from ocr.paddleocr_pipeline import PaddleOCRPipeline

        with patch("ocr.paddleocr_pipeline.PaddleOCRPipeline.__init__", return_value=None):
            pipeline = PaddleOCRPipeline.__new__(PaddleOCRPipeline)
            pipeline.confidence_threshold = 0.5
            pipeline.line_merge_threshold = 15

        raw_results = [
            [[[10, 10], [100, 10], [100, 30], [10, 30]], ("Tab Amoxicillin", 0.95)],
            [[[120, 10], [200, 10], [200, 30], [120, 30]], ("500mg", 0.90)],
        ]

        detections = pipeline._parse_detections(raw_results)
        assert len(detections) == 2
        assert detections[0]["text"] == "Tab Amoxicillin"
        assert detections[0]["confidence"] == 0.95
        assert "y_center" in detections[0]
        assert "x_center" in detections[0]

    def test_confidence_filtering(self):
        """Test that low-confidence detections are filtered out."""
        from ocr.paddleocr_pipeline import PaddleOCRPipeline

        with patch("ocr.paddleocr_pipeline.PaddleOCRPipeline.__init__", return_value=None):
            pipeline = PaddleOCRPipeline.__new__(PaddleOCRPipeline)
            pipeline.confidence_threshold = 0.6
            pipeline.line_merge_threshold = 15

        detections = [
            {"text": "Good", "confidence": 0.95, "y_center": 10, "x_center": 10},
            {"text": "Noisy", "confidence": 0.3, "y_center": 20, "x_center": 10},
            {"text": "OK", "confidence": 0.7, "y_center": 30, "x_center": 10},
        ]

        filtered = pipeline._filter_by_confidence(detections)
        assert len(filtered) == 2
        assert all(d["confidence"] >= 0.6 for d in filtered)

    def test_line_grouping(self):
        """Test that detections are grouped into lines by Y-coordinate."""
        from ocr.paddleocr_pipeline import PaddleOCRPipeline

        with patch("ocr.paddleocr_pipeline.PaddleOCRPipeline.__init__", return_value=None):
            pipeline = PaddleOCRPipeline.__new__(PaddleOCRPipeline)
            pipeline.confidence_threshold = 0.5
            pipeline.line_merge_threshold = 15

        detections = [
            {"text": "Tab", "confidence": 0.9, "y_center": 10, "x_center": 10},
            {"text": "Amoxicillin", "confidence": 0.9, "y_center": 12, "x_center": 50},
            {"text": "500mg", "confidence": 0.9, "y_center": 11, "x_center": 100},
            {"text": "Tab", "confidence": 0.9, "y_center": 50, "x_center": 10},
            {"text": "Paracetamol", "confidence": 0.9, "y_center": 51, "x_center": 50},
        ]

        lines = pipeline._group_into_lines(detections)
        assert len(lines) == 2
        assert "Amoxicillin" in lines[0]
        assert "Paracetamol" in lines[1]

    def test_line_sorting_left_to_right(self):
        """Test that detections within a line are sorted left-to-right."""
        from ocr.paddleocr_pipeline import PaddleOCRPipeline

        with patch("ocr.paddleocr_pipeline.PaddleOCRPipeline.__init__", return_value=None):
            pipeline = PaddleOCRPipeline.__new__(PaddleOCRPipeline)
            pipeline.confidence_threshold = 0.5
            pipeline.line_merge_threshold = 15

        # Deliberately out of order (right word first)
        detections = [
            {"text": "500mg", "confidence": 0.9, "y_center": 10, "x_center": 100},
            {"text": "Tab", "confidence": 0.9, "y_center": 10, "x_center": 10},
            {"text": "Amoxicillin", "confidence": 0.9, "y_center": 10, "x_center": 50},
        ]

        lines = pipeline._group_into_lines(detections)
        assert len(lines) == 1
        assert lines[0] == "Tab Amoxicillin 500mg"

    def test_file_not_found_raises(self):
        """Test that extract raises FileNotFoundError for missing images."""
        from ocr.paddleocr_pipeline import PaddleOCRPipeline

        with patch("ocr.paddleocr_pipeline.PaddleOCRPipeline.__init__", return_value=None):
            pipeline = PaddleOCRPipeline.__new__(PaddleOCRPipeline)
            pipeline._available = True

        with pytest.raises(FileNotFoundError):
            pipeline.extract("/nonexistent/image.jpg")

    def test_unavailable_pipeline_raises(self):
        """Test that extract raises RuntimeError when PaddleOCR isn't installed."""
        from ocr.paddleocr_pipeline import PaddleOCRPipeline

        with patch("ocr.paddleocr_pipeline.PaddleOCRPipeline.__init__", return_value=None):
            pipeline = PaddleOCRPipeline.__new__(PaddleOCRPipeline)
            pipeline._available = False

        with pytest.raises(RuntimeError, match="not installed"):
            pipeline.extract("some_image.jpg")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Vision Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMVisionPipeline:
    """Tests for the Gemini Vision-based lab report OCR pipeline."""

    def test_supported_extensions(self):
        """Test that common image and PDF formats are supported."""
        from ocr.llm_vision_pipeline import LLMVisionPipeline

        supported = LLMVisionPipeline.SUPPORTED_EXTENSIONS
        assert ".jpg" in supported
        assert ".jpeg" in supported
        assert ".png" in supported
        assert ".pdf" in supported
        assert ".tiff" in supported

    def test_unsupported_extension_raises(self):
        """Test that unsupported file types raise ValueError."""
        from ocr.llm_vision_pipeline import LLMVisionPipeline

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("ocr.llm_vision_pipeline.LLMVisionPipeline.__init__", return_value=None):
                pipeline = LLMVisionPipeline.__new__(LLMVisionPipeline)
                pipeline._available = True

        # Create a temporary file with unsupported extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                pipeline.extract(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_file_not_found_raises(self):
        """Test that extract raises FileNotFoundError for missing files."""
        from ocr.llm_vision_pipeline import LLMVisionPipeline

        with patch("ocr.llm_vision_pipeline.LLMVisionPipeline.__init__", return_value=None):
            pipeline = LLMVisionPipeline.__new__(LLMVisionPipeline)
            pipeline._available = True

        with pytest.raises(FileNotFoundError):
            pipeline.extract("/nonexistent/report.pdf")

    def test_unavailable_pipeline_raises(self):
        """Test that extract raises RuntimeError when Gemini is not configured."""
        from ocr.llm_vision_pipeline import LLMVisionPipeline

        with patch("ocr.llm_vision_pipeline.LLMVisionPipeline.__init__", return_value=None):
            pipeline = LLMVisionPipeline.__new__(LLMVisionPipeline)
            pipeline._available = False

        with pytest.raises(RuntimeError, match="not available"):
            pipeline.extract("some_report.pdf")

    def test_system_prompt_is_defined(self):
        """Test that the vision system prompt contains key instructions."""
        from ocr.llm_vision_pipeline import VISION_SYSTEM_PROMPT

        assert "OCR" in VISION_SYSTEM_PROMPT
        assert "extract" in VISION_SYSTEM_PROMPT.lower()
        assert "UNREADABLE" in VISION_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════════
# OCR Evaluation Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOCREvalMetrics:
    """Tests for OCR evaluation metric computation."""

    def test_exact_match_identical(self):
        """Test exact match with identical strings."""
        from ocr.eval_ocr import compute_exact_match

        assert compute_exact_match("Hello World", "hello world") is True

    def test_exact_match_different(self):
        """Test exact match with different strings."""
        from ocr.eval_ocr import compute_exact_match

        assert compute_exact_match("Hello", "World") is False

    def test_cer_identical_strings(self):
        """Test CER is 0 for identical strings."""
        from ocr.eval_ocr import compute_cer

        cer = compute_cer("Amoxicillin 500mg", "Amoxicillin 500mg")
        assert cer == 0.0

    def test_cer_different_strings(self):
        """Test CER is > 0 for different strings."""
        from ocr.eval_ocr import compute_cer

        cer = compute_cer("Am0xicillin", "Amoxicillin")
        assert cer > 0.0

    def test_wer_identical_strings(self):
        """Test WER is 0 for identical strings."""
        from ocr.eval_ocr import compute_wer

        wer = compute_wer("Tab Amoxicillin 500mg", "Tab Amoxicillin 500mg")
        assert wer == 0.0
