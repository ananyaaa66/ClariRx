"""
FastAPI Request and Response Schemas
======================================
Defines the Pydantic models for the API endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# We reuse the schemas from extraction and explanation modules
from extraction.extract import DocType, PrescriptionItem, LabReportItem, ExtractionResult
from explanation.explain_medicine import MedicineExplanation
from explanation.explain_lab_value import LabExplanation


# ─── Upload & Extraction ───────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response returned when an image is uploaded and processed."""
    success: bool
    doc_type: DocType
    raw_text: str
    items: List[PrescriptionItem | LabReportItem] = Field(default_factory=list)
    error_message: Optional[str] = None


# ─── Explanation Requests ──────────────────────────────────────────────────────

class MedicineExplainRequest(BaseModel):
    """Request body for medicine explanation."""
    drug_name: str
    frequency: str = ""
    duration: str = ""
    instructions: str = ""

class LabExplainRequest(BaseModel):
    """Request body for lab value explanation."""
    test_name: str
    value: str
    unit: str = ""
    normal_range_low: Optional[float] = None
    normal_range_high: Optional[float] = None
    gender: str = "male"


# ─── Reminders ─────────────────────────────────────────────────────────────────

class ReminderCreate(BaseModel):
    """Payload to create a new medication reminder."""
    drug_name: str
    dosage: str = ""
    frequency_times: List[str] = Field(..., description="List of times in HH:MM format (24h)")
    start_date: str = Field(..., description="YYYY-MM-DD")
    duration_days: int = 1
    instructions: str = ""

class ReminderResponse(ReminderCreate):
    """Response representing a saved reminder."""
    id: str
    is_active: bool = True
