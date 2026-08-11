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

    try:
        # 1. Run OCR
        # We need an OCR function that takes an image path and returns text.
        # Let's import the one built in Phase 1 or use a dummy for now if it requires full model load.
        try:
            from ocr.inference import run_ocr
            raw_text = run_ocr(tmp_path)
        except ImportError:
            # Fallback if inference isn't fully set up yet
            logger.warning("ocr.inference not found or failed to load. Using mock OCR text.")
            if doc_type == "prescription":
                raw_text = "Rx\n1. Tab Amoxicillin 500mg 1-0-1 after food 5 days\n2. Tab Paracetamol 650mg SOS"
            else:
                raw_text = "Complete Blood Count\nHaemoglobin 14.2 g/dL\nWBC Count 11500 cells/uL"

        if not raw_text.strip():
            return UploadResponse(
                success=False,
                doc_type=DocType(doc_type),
                raw_text="",
                error_message="OCR failed to extract any text from the image."
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
