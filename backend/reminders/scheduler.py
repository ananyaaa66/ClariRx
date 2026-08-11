"""
Medication Reminder Scheduler
===============================
Uses APScheduler to check for due medication reminders.
"""

import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from api.routes.reminders import get_all_reminders

logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler = BackgroundScheduler()

def check_reminders():
    """
    Periodic job that checks if any reminder is due right now.
    """
    now = datetime.now()
    current_time_str = now.strftime("%H:%M")

    active_reminders = get_all_reminders()
    for reminder in active_reminders:
        if not reminder.is_active:
            continue
            
        # Very simplistic check for demonstration
        if current_time_str in reminder.frequency_times:
            # In a real app, this would send an FCM push notification or SMS.
            # Here we just log it to the console.
            logger.info("*" * 60)
            logger.info(f"⏰ REMINDER: Time to take {reminder.drug_name}!")
            if reminder.dosage:
                logger.info(f"   Dosage: {reminder.dosage}")
            if reminder.instructions:
                logger.info(f"   Instructions: {reminder.instructions}")
            logger.info("*" * 60)

def start_scheduler():
    """
    Start the background scheduler.
    """
    if not _scheduler.running:
        # Check every 60 seconds
        _scheduler.add_job(
            func=check_reminders,
            trigger=IntervalTrigger(seconds=60),
            id='check_reminders_job',
            name='Check for due medication reminders',
            replace_existing=True,
        )
        _scheduler.start()
        logger.info("Background reminder scheduler started. Checking every 60s.")

def shutdown_scheduler():
    """
    Stop the background scheduler.
    """
    if _scheduler.running:
        _scheduler.shutdown()
        logger.info("Background reminder scheduler stopped.")
