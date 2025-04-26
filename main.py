import os
import sys
import uvicorn
import logging
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import config
from routers import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Prerequisite Checks ---


def check_environment_variables():
    """Checks if required environment variables are set."""
    load_dotenv()

    # Check for Gemini API Key
    if not config.GEMINI_API_KEY:
        logger.error("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
    else:
        logger.info("GEMINI_API_KEY found.")


def check_required_files():
    """Checks if required dataset files exist."""
    required_files = [config.MOVIES_CSV, config.RATINGS_CSV, config.TAGS_CSV]
    all_files_found = True
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Error: Required data file not found: {file_path}")
            logger.error(f"Expected location: {os.path.abspath(file_path)}")
            all_files_found = False

    if not all_files_found:
        sys.exit(1)
    else:
        logger.info("All required data files found.")


def run_prerequisite_checks():
    """Runs all prerequisite checks before starting the application."""
    logger.info("Running prerequisite checks...")

    check_environment_variables()
    check_required_files()

    # Note: Spacy model check is now handled in models.py
    # to avoid duplication of logic

    logger.info("Prerequisite checks passed.")


run_prerequisite_checks()

# --- FastAPI Application Setup ---

app = FastAPI(
    title="Movie Recommendation API",
    description="Provides movie recommendations using LLM and Keyword-based approaches.",
    version="1.0.0",
)

app.include_router(router)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly (for development)...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
