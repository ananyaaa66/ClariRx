"""
LLM Vision Pipeline for Lab Report Extraction
================================================

Uses Google Gemini's multimodal vision capability to extract
structured text from lab report images and PDFs.

Approach:
  - Sends the document image to Gemini 2.5 Flash with a structured prompt
  - Returns raw text extraction (not JSON — extraction happens downstream)
  - Falls back to base64 inline data if file upload fails

Usage (standalone):
    python llm_vision_pipeline.py --image /path/to/lab_report.jpg

Usage (as module):
    from ocr.llm_vision_pipeline import run_llm_vision
    text = run_llm_vision("path/to/lab_report.pdf")
"""

from __future__ import annotations

import argparse
import base64
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── System Prompt ─────────────────────────────────────────────────────────────

VISION_SYSTEM_PROMPT = """You are an OCR text extraction assistant for ClariRx, a medical assistant app.

Your task is to accurately extract ALL text from this medical lab report image.

Output rules:
1. Extract text EXACTLY as it appears — do not reinterpret or summarize.
2. Preserve the structure: test names, values, units, and reference ranges on each line.
3. Use this format per line: "Test Name: Value Unit (Reference Range: Low - High)"
4. If the document is a prescription instead of a lab report, extract it as-is.
5. If the image is unclear or unreadable, output "UNREADABLE" for that section.
6. Do NOT add any commentary, headers, or formatting beyond the raw text.
7. Output ONLY the extracted text, nothing else."""


# ─── LLM Vision Pipeline ──────────────────────────────────────────────────────

class LLMVisionPipeline:
    """
    Multimodal LLM-based text extraction for lab reports using Google Gemini.

    Features:
        - Supports image files (JPEG, PNG, TIFF, BMP, WebP) and PDFs
        - Uses Gemini 2.5 Flash for fast, accurate vision extraction
        - Falls back to base64 inline data if file upload fails
        - Structured prompting for consistent output format
    """

    # Supported file types and their MIME types
    SUPPORTED_EXTENSIONS = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
    }

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the LLM Vision pipeline.

        Args:
            model_name: Gemini model to use for vision extraction.

        Raises:
            ValueError: If GEMINI_API_KEY is not set.
        """
        self.model_name = model_name
        self._model = None
        self._available = False

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY not set. LLM Vision pipeline will not be available."
            )
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model_name)
            self._genai = genai
            self._available = True
            logger.info(f"Gemini Vision pipeline initialized with model: {model_name}")
        except ImportError:
            logger.warning(
                "google-generativeai is not installed. Install with: "
                "pip install google-generativeai"
            )

    @property
    def is_available(self) -> bool:
        """Check if the pipeline is properly initialized."""
        return self._available

    def extract(self, image_path: str) -> str:
        """
        Extract text from a lab report image or PDF using Gemini Vision.

        Args:
            image_path: Path to the lab report file (image or PDF).

        Returns:
            Extracted text from the document.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the pipeline is not available.
            ValueError: If the file type is not supported.
        """
        if not self._available:
            raise RuntimeError(
                "LLM Vision pipeline is not available. "
                "Ensure GEMINI_API_KEY is set and google-generativeai is installed."
            )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        ext = Path(image_path).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        mime_type = self.SUPPORTED_EXTENSIONS[ext]

        # Try file upload first, fall back to base64
        try:
            return self._extract_with_upload(image_path, mime_type)
        except Exception as e:
            logger.warning(f"File upload failed ({e}), falling back to base64 inline.")
            return self._extract_with_base64(image_path, mime_type)

    def _extract_with_upload(self, image_path: str, mime_type: str) -> str:
        """Extract using Gemini file upload API."""
        uploaded_file = self._genai.upload_file(
            path=image_path,
            mime_type=mime_type,
        )

        response = self._model.generate_content(
            [VISION_SYSTEM_PROMPT, uploaded_file],
            generation_config=self._genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )

        text = response.text.strip()
        logger.info(
            f"Vision extraction complete (upload): "
            f"{len(text)} chars from {os.path.basename(image_path)}"
        )
        return text

    def _extract_with_base64(self, image_path: str, mime_type: str) -> str:
        """Extract using base64-encoded inline data as fallback."""
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        image_part = {
            "mime_type": mime_type,
            "data": data,
        }

        response = self._model.generate_content(
            [VISION_SYSTEM_PROMPT, image_part],
            generation_config=self._genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )

        text = response.text.strip()
        logger.info(
            f"Vision extraction complete (base64): "
            f"{len(text)} chars from {os.path.basename(image_path)}"
        )
        return text


# ─── Convenience Function ──────────────────────────────────────────────────────

_pipeline: Optional[LLMVisionPipeline] = None


def run_llm_vision(image_path: str, **kwargs) -> str:
    """
    Run LLM Vision extraction on a lab report image and return text.

    This is the main entry point for the vision pipeline. It lazily
    initializes the Gemini model on first call.

    Args:
        image_path: Path to the lab report file.
        **kwargs: Additional arguments passed to LLMVisionPipeline.

    Returns:
        Extracted text string.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = LLMVisionPipeline(**kwargs)
    return _pipeline.extract(image_path)


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv

    SCRIPT_DIR = Path(__file__).resolve().parent
    load_dotenv(SCRIPT_DIR.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Extract text from a lab report using Gemini Vision."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to lab report image/PDF")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = LLMVisionPipeline(model_name=args.model)
    text = pipeline.extract(args.image)

    print(f"\n{'=' * 60}")
    print("  Gemini Vision — Extracted Text")
    print(f"{'=' * 60}")
    print(text)
    print(f"{'=' * 60}\n")
