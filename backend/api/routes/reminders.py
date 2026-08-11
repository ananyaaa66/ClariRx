"""
Reminders Router
==================
Endpoints for creating and managing medication reminders.
Uses an in-memory store for simplicity in this version.
"""

from typing import List
from uuid import uuid4
from fastapi import APIRouter, HTTPException

from api.schemas import ReminderCreate, ReminderResponse

router = APIRouter()

# In-memory storage for reminders
# In a real app, this would be a database (e.g., SQLite or PostgreSQL)
_REMINDERS_DB = {}

@router.post("/api/reminders", response_model=ReminderResponse)
async def create_reminder(reminder: ReminderCreate):
    """
    Create a new medication reminder.
    """
    reminder_id = str(uuid4())
    new_reminder = ReminderResponse(
        id=reminder_id,
        **reminder.model_dump()
    )
    _REMINDERS_DB[reminder_id] = new_reminder
    return new_reminder

@router.get("/api/reminders", response_model=List[ReminderResponse])
async def list_reminders():
    """
    List all active medication reminders.
    """
    return list(_REMINDERS_DB.values())

@router.delete("/api/reminders/{reminder_id}")
async def delete_reminder(reminder_id: str):
    """
    Delete a medication reminder.
    """
    if reminder_id not in _REMINDERS_DB:
        raise HTTPException(status_code=404, detail="Reminder not found")
    del _REMINDERS_DB[reminder_id]
    return {"message": "Reminder deleted successfully"}

def get_all_reminders():
    """Helper function for the background scheduler to access reminders."""
    return list(_REMINDERS_DB.values())
