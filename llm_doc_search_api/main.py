import logging
import os
# from typing import Any, Dict # No longer needed here for documents_db
from contextlib import asynccontextmanager
from logging.config import dictConfig

from fastapi import FastAPI

from . import config  # Your application's config module
from .database import save_db_to_file  # Use the correct function name
# These are needed for startup event
from .embedding_generator import get_model as get_embedding_model
from .llm_handler import get_llm as get_llm_model_instance
from .routes import router as api_router  # Import the router from routes.py
from .text_extractor import SUPPORTED_EXTENSIONS  # For app description
from .vector_store import get_collection_count

# Import documents_db from the new database module
# from .database import documents_db # No longer need to import it here if routes.py handles it

# Configure logging (this should be done once, early)
dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


@asynccontextmanager  # <--- ADD THIS DECORATOR
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    logger.info("Application startup via lifespan manager...")
    # load_db() # This is called when database.py is imported, so not strictly needed here again
    # unless you want to ensure it's explicitly part of app lifecycle start.
    # For now, relying on module import is fine.

    try:
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)
        logger.info(f"Upload directory '{config.UPLOAD_DIR}' ensured.")

        get_embedding_model()
        logger.info(f"Embedding model '{config.EMBEDDING_MODEL_NAME}' loaded.")

        get_llm_model_instance()
        logger.info(f"LLM model from '{config.LLM_MODEL_PATH}' loaded.")

        logger.info(
            f"Initial ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' count: {get_collection_count()}"
        )
    except Exception as e:
        logger.error(
            f"Lifespan Startup Error: Could not pre-load models or init Chroma: {e}",
            exc_info=True,
        )
    logger.info("Application startup complete (lifespan).")

    yield  # The application runs while the context manager is active

    # --- Shutdown Logic ---
    logger.info("Application shutting down via lifespan manager...")
    save_db_to_file()  # <--- Make sure this matches the function name in database.py
    logger.info("Database saved. Application shutdown complete (lifespan).")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Document Search API",
    description=f"API for uploading documents ({', '.join(SUPPORTED_EXTENSIONS)}) and performing semantic search.",
    version=config.APP_VERSION,
    lifespan=lifespan,  # <--- PASS THE LIFESPAN FUNCTION HERE
)


# Include the API routes
app.include_router(api_router)


# --- Main execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Uvicorn server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        "llm_doc_search_api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_config=None,
    )
