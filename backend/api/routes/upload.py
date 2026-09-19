"""
Upload Router
===============
Handles image uploads, runs OCR, and extracts structured data.
"""

import os
import tempfile
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Form

from api.schemas import UploadResponse
from extraction.extract import run_extraction, DocType

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/api/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    doc_type: str = Form(default="prescription"),
):
    """
    Upload an image (prescription or lab report) to process.
    """
    if doc_type not in ["prescription", "lab_report"]:
        raise HTTPException(status_code=400, detail="doc_type must be 'prescription' or 'lab_report'")

    # Save the uploaded file to a temporary location
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

        # 1. Run OCR pipeline (PaddleOCR -> LLM Vision fallback)
        raw_text = ""
        try:
            from ocr.paddleocr_pipeline import run_paddleocr
            raw_text = run_paddleocr(tmp_path)
        except Exception as e1:
            logger.warning(f"PaddleOCR pipeline unavailable or failed: {e1}")
            try:
                from ocr.llm_vision_pipeline import run_llm_vision
                raw_text = run_llm_vision(tmp_path)
            except Exception as e2:
                logger.warning(f"LLM Vision pipeline failed: {e2}")

        if not raw_text or not raw_text.strip():
            return UploadResponse(
                success=False,
                doc_type=DocType(doc_type),
                raw_text="",
                items=[],
                error_message="OCR failed to extract any readable text from the image."
            )

        # 2. Extract Entities
        extraction_result = run_extraction(
            raw_text=raw_text,
            doc_type=doc_type,
            fallback=True # use BioBERT -> LLM fallback
        )

        items = []
        if doc_type == "prescription":
            items = extraction_result.prescription_items
        else:
            items = extraction_result.lab_report_items

        return UploadResponse(
            success=extraction_result.success,
            doc_type=DocType(doc_type),
            raw_text=raw_text,
            items=items,
            error_message=extraction_result.error_message
        )

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
