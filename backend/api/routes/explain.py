"""
Explain Router
================
Endpoints to generate simple bilingual explanations from the Knowledge Base.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import MedicineExplainRequest, LabExplainRequest
from explanation.explain_medicine import explain_medicine, MedicineExplanation
from explanation.explain_lab_value import explain_lab_value, LabExplanation

router = APIRouter()

@router.post("/api/explain/medicine", response_model=MedicineExplanation)
async def api_explain_medicine(req: MedicineExplainRequest):
    """
    Explain a medicine in simple English and Hindi.
    """
    try:
        explanation = explain_medicine(
            drug_name=req.drug_name,
            frequency=req.frequency,
            duration=req.duration,
            instructions=req.instructions
        )
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/explain/lab", response_model=LabExplanation)
async def api_explain_lab_value(req: LabExplainRequest):
    """
    Explain a lab test result in simple English and Hindi.
    """
    try:
        explanation = explain_lab_value(
            test_name=req.test_name,
            value=req.value,
            unit=req.unit,
            normal_range_low=req.normal_range_low,
            normal_range_high=req.normal_range_high,
            gender=req.gender
        )
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
