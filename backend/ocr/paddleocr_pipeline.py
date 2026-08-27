"""
PaddleOCR Inference Pipeline for Handwritten Prescriptions
============================================================

Wraps PaddleOCR to extract text from handwritten prescription images.
Uses bounding-box spatial sorting (top-to-bottom, left-to-right)
and line grouping to reconstruct proper reading order.

Usage (standalone):
    python paddleocr_pipeline.py --image /path/to/prescription.jpg

Usage (as module):
    from ocr.paddleocr_pipeline import run_paddleocr
    text = run_paddleocr("path/to/image.jpg")
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── PaddleOCR Pipeline ───────────────────────────────────────────────────────

class PaddleOCRPipeline:
    """
    Wrapper around PaddleOCR for prescription image → text extraction.

    Features:
        - Bounding box sorting for reading order reconstruction
        - Confidence thresholding to filter noisy detections
        - Y-coordinate line grouping for multi-line prescriptions
        - Configurable language and detection models
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.3,
        confidence_threshold: float = 0.5,
        line_merge_threshold: int = 15,
    ):
        """
        Initialize the PaddleOCR pipeline.

        Args:
            lang: OCR language — "en" for English, "hi" for Hindi.
            use_angle_cls: Whether to use text angle classification.
            det_db_thresh: Detection DB threshold (lower = more sensitive).
            confidence_threshold: Minimum confidence to accept a text box.
            line_merge_threshold: Max Y-pixel distance to merge boxes into same line.
        """
        self.confidence_threshold = confidence_threshold
        self.line_merge_threshold = line_merge_threshold

        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                det_db_thresh=det_db_thresh,
                show_log=False,
            )
            self._available = True
            logger.info("PaddleOCR engine initialized successfully.")
        except ImportError:
            logger.warning(
                "PaddleOCR is not installed. Install with: "
                "pip install paddleocr paddlepaddle"
            )
            self.ocr = None
            self._available = False

    @property
    def is_available(self) -> bool:
        """Check if PaddleOCR is properly initialized."""
        return self._available

    def extract(self, image_path: str) -> str:
        """
        Extract text from a prescription image.

        Args:
            image_path: Path to the prescription image file.

        Returns:
            Extracted text with lines separated by newlines.

        Raises:
            FileNotFoundError: If the image file does not exist.
            RuntimeError: If PaddleOCR is not available.
        """
        if not self._available:
            raise RuntimeError(
                "PaddleOCR is not installed. Cannot perform OCR extraction."
            )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run PaddleOCR
        results = self.ocr.ocr(image_path, cls=True)

        if not results or not results[0]:
            logger.warning(f"PaddleOCR returned no results for: {image_path}")
            return ""

        # Extract bounding boxes, text, and confidence
        detections = self._parse_detections(results[0])

        # Filter by confidence
        detections = self._filter_by_confidence(detections)

        if not detections:
            return ""

        # Sort and group into lines
        lines = self._group_into_lines(detections)

        # Join lines into final text
        text = "\n".join(lines)
        logger.info(
            f"Extracted {len(lines)} lines from {os.path.basename(image_path)}"
        )
        return text

    def _parse_detections(
        self, raw_results: list
    ) -> List[dict]:
        """Parse raw PaddleOCR output into structured detections."""
        detections = []
        for item in raw_results:
            bbox = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = item[1][0]
            confidence = item[1][1]

            # Calculate center Y for sorting
            y_center = sum(point[1] for point in bbox) / 4
            x_center = sum(point[0] for point in bbox) / 4

            detections.append({
                "bbox": bbox,
                "text": text,
                "confidence": confidence,
                "y_center": y_center,
                "x_center": x_center,
            })

        return detections

    def _filter_by_confidence(self, detections: List[dict]) -> List[dict]:
        """Filter out low-confidence detections."""
        filtered = [
            d for d in detections
            if d["confidence"] >= self.confidence_threshold
        ]
        removed = len(detections) - len(filtered)
        if removed > 0:
            logger.debug(
                f"Filtered {removed} low-confidence detections "
                f"(threshold={self.confidence_threshold})"
            )
        return filtered

    def _group_into_lines(self, detections: List[dict]) -> List[str]:
        """
        Group detections into lines based on Y-coordinate proximity,
        then sort each line left-to-right.
        """
        # Sort by Y center first
        detections.sort(key=lambda d: d["y_center"])

        lines: List[List[dict]] = []
        current_line: List[dict] = []
        current_y = None

        for det in detections:
            if current_y is None:
                current_line = [det]
                current_y = det["y_center"]
            elif abs(det["y_center"] - current_y) <= self.line_merge_threshold:
                current_line.append(det)
            else:
                lines.append(current_line)
                current_line = [det]
                current_y = det["y_center"]

        if current_line:
            lines.append(current_line)

        # Sort each line left-to-right and join text
        result = []
        for line in lines:
            line.sort(key=lambda d: d["x_center"])
            line_text = " ".join(d["text"] for d in line)
            result.append(line_text)

        return result


# ─── Convenience Function ──────────────────────────────────────────────────────

# Module-level singleton (lazy init)
_pipeline: Optional[PaddleOCRPipeline] = None


def run_paddleocr(image_path: str, **kwargs) -> str:
    """
    Run PaddleOCR on a prescription image and return extracted text.

    This is the main entry point for the OCR pipeline. It lazily
    initializes the PaddleOCR engine on first call.

    Args:
        image_path: Path to the prescription image file.
        **kwargs: Additional arguments passed to PaddleOCRPipeline.

    Returns:
        Extracted text string.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = PaddleOCRPipeline(**kwargs)
    return _pipeline.extract(image_path)


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text from a prescription image using PaddleOCR."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = PaddleOCRPipeline(confidence_threshold=args.confidence)
    text = pipeline.extract(args.image)

    print(f"\n{'=' * 60}")
    print("  PaddleOCR — Extracted Text")
    print(f"{'=' * 60}")
    print(text)
    print(f"{'=' * 60}\n")
