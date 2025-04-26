import os
import sys
import spacy
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


def check_and_download_spacy_model(model_name):
    """Checks if a SpaCy model is installed and downloads it if not."""
    try:
        spacy.load(model_name)
        logger.info(f"SpaCy model '{model_name}' already installed.")
    except OSError:
        logger.warning(f"SpaCy model '{model_name}' not found. Attempting download...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name]
            )
            logger.info(
                f"Successfully downloaded '{model_name}'. Please restart the script if needed."
            )
            spacy.load(model_name)  # Try loading again
        except subprocess.CalledProcessError:
            logger.error(f"ERROR: Failed to download SpaCy model '{model_name}'.")
            logger.error(
                f"Please install it manually: python -m spacy download {model_name}"
            )
            sys.exit(1)
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during SpaCy model download: {e}"
            )
            sys.exit(1)


def run_prerequisite_checks():
    """Runs all prerequisite checks before starting the application."""
    logger.info("Running prerequisite checks...")
    load_dotenv()

    # Check for Gemini API Key
    if not config.GEMINI_API_KEY:
        logger.error("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
    else:
        logger.info("GEMINI_API_KEY found.")

    # Check for essential dataset files
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

    # Perform SpaCy model check
    check_and_download_spacy_model(config.SPACY_MODEL_NAME)

    logger.info("Prerequisite checks passed.")


run_prerequisite_checks()

# --- FastAPI Application Setup ---

app = FastAPI(
    title="Movie Recommendation API",
    description="Provides movie recommendations using LLM and Keyword-based approaches.",
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly (for development)...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
