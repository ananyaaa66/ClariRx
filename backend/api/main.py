"""
ClariRx FastAPI Main Entry Point
=================================
Initializes the FastAPI application, configures CORS, and includes routers.
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Ensure backend directory is in the path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

# Import routers
from api.routes import upload, explain, reminders

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ClariRx")
    logger.info("Starting ClariRx API...")
    
    # Start the reminder scheduler
    try:
        from reminders.scheduler import start_scheduler
        start_scheduler()
        logger.info("Reminder scheduler started.")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
    yield

app = FastAPI(
    title="ClariRx API",
    description="Backend API for the ClariRx medical assistant.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, tags=["Upload"])
app.include_router(explain.router, tags=["Explanation"])
app.include_router(reminders.router, tags=["Reminders"])

@app.get("/", tags=["Health"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "app": "ClariRx"}
