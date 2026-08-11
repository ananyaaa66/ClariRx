"""
Integration Tests for FastAPI Endpoints (Phase 5)
===================================================

Tests:
  - Explain Medicine endpoint
  - Explain Lab Value endpoint
  - Reminders CRUD operations
"""

import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# ─── Explain Endpoints ────────────────────────────────────────────────────────

def test_explain_medicine():
    payload = {
        "drug_name": "Amoxicillin 500mg",
        "frequency": "1-0-1",
        "duration": "5 days",
        "instructions": "after food"
    }
    response = client.post("/api/explain/medicine", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["drug_name"] == "Amoxicillin 500mg"
    assert data["generic_name"] == "Amoxicillin"
    assert "explanation_en" in data
    assert "explanation_hi" in data
    assert data["severity"] == "safe"
    assert data["kb_grounded"] is True

def test_explain_lab_value():
    payload = {
        "test_name": "Haemoglobin",
        "value": "14.2",
        "unit": "g/dL",
        "gender": "male"
    }
    response = client.post("/api/explain/lab", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["test_name"] == "Haemoglobin"
    assert data["severity"] == "normal"
    assert "explanation_en" in data
    assert data["kb_grounded"] is True

# ─── Reminders Endpoints ──────────────────────────────────────────────────────

def test_reminders_crud():
    # 1. Create Reminder
    payload = {
        "drug_name": "Paracetamol",
        "dosage": "650mg",
        "frequency_times": ["08:00", "20:00"],
        "start_date": "2026-08-01",
        "duration_days": 5,
        "instructions": "SOS"
    }
    response = client.post("/api/reminders", json=payload)
    assert response.status_code == 200
    created = response.json()
    assert created["drug_name"] == "Paracetamol"
    assert "id" in created
    reminder_id = created["id"]

    # 2. List Reminders
    response = client.get("/api/reminders")
    assert response.status_code == 200
    reminders = response.json()
    assert len(reminders) >= 1
    assert any(r["id"] == reminder_id for r in reminders)

    # 3. Delete Reminder
    response = client.delete(f"/api/reminders/{reminder_id}")
    assert response.status_code == 200

    # 4. Verify Deletion
    response = client.get("/api/reminders")
    assert response.status_code == 200
    reminders = response.json()
    assert not any(r["id"] == reminder_id for r in reminders)
