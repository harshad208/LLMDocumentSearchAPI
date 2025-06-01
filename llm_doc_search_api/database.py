import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)
DB_FILE = "documents_db.json"

documents_db: Dict[str, Dict[str, Any]] = {}


def load_db_from_file():
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
                    documents_db = {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {DB_FILE}. Initializing empty DB.")
            documents_db = {}
        except Exception as e:
            logger.error(f"Error loading {DB_FILE}: {e}. Initializing empty DB.")
            documents_db = {}
    else:
        logger.info(f"{DB_FILE} not found. Initializing empty DB.")
        documents_db = {}


def save_db_to_file():
    global documents_db
    try:
        db_dir = os.path.dirname(DB_FILE)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created directory for DB_FILE: {db_dir}")

        with open(DB_FILE, "w") as f:
            json.dump(documents_db, f, indent=4)
        logger.info(f"Saved {len(documents_db)} documents to {DB_FILE}")
    except Exception as e:
        logger.error(f"Error saving to {DB_FILE}: {e}", exc_info=True)


load_db_from_file()
