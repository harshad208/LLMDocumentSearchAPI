import logging
import os
from contextlib import asynccontextmanager
from logging.config import dictConfig

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .database import save_db_to_file
from .embedding_generator import get_model as get_embedding_model
from .llm_handler import get_llm as get_llm_model_instance
from .routes import router as api_router
from .text_extractor import SUPPORTED_EXTENSIONS
from .vector_store import get_collection_count

dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    logger.info("Application startup via lifespan manager...")

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

    yield

    # --- Shutdown Logic ---
    logger.info("Application shutting down via lifespan manager...")
    save_db_to_file()
    logger.info("Database saved. Application shutdown complete (lifespan).")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Document Search API",
    description=f"API for uploading documents ({', '.join(SUPPORTED_EXTENSIONS)}) and performing semantic search.",
    version=config.APP_VERSION,
    lifespan=lifespan,
)


# Include the API routes
app.include_router(api_router)

ALLOW_ALL = os.getenv("CORS_ALLOW_ALL", "false").lower() == "true"

if ALLOW_ALL:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    allowed = os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed if origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- Main execution ---
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
