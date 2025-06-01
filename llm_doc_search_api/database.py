# llm_doc_search_api/database.py
import json
import logging
import os
from typing import (  # Removed List and BaseModel if not used directly here
    Any, Dict)

# from .models import DocumentMetadata # Only if you were type hinting with it directly here

logger = logging.getLogger(__name__)
DB_FILE = "documents_db.json"  # Make sure this is a path Uvicorn can write to

# This is the actual in-memory store that your routes will use
documents_db: Dict[str, Dict[str, Any]] = {}


def load_db_from_file():  # Renamed for clarity
    global documents_db
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    documents_db = loaded_data
                    logger.info(f"Loaded {len(documents_db)} documents from {DB_FILE}")
                else:
                    logger.warning(
                        f"DB_FILE {DB_FILE} did not contain a dictionary. Initializing empty DB."
                    )
                    documents_db = (
                        {}
                    )  # Ensure it's initialized if file content is wrong
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {DB_FILE}. Initializing empty DB.")
            documents_db = {}
        except Exception as e:
            logger.error(f"Error loading {DB_FILE}: {e}. Initializing empty DB.")
            documents_db = {}
    else:
        logger.info(f"{DB_FILE} not found. Initializing empty DB.")
        documents_db = {}  # Ensure it's initialized if file doesn't exist


def save_db_to_file():  # Renamed for clarity
    global documents_db  # Ensure we are saving the global one
    try:
        # Ensure the directory for DB_FILE exists if it's nested
        db_dir = os.path.dirname(DB_FILE)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created directory for DB_FILE: {db_dir}")

        with open(DB_FILE, "w") as f:
            json.dump(documents_db, f, indent=4)
        logger.info(f"Saved {len(documents_db)} documents to {DB_FILE}")
    except Exception as e:
        logger.error(f"Error saving to {DB_FILE}: {e}", exc_info=True)


# Call load_db_from_file() when the module is first imported
# This populates documents_db from the file at import time.
load_db_from_file()
