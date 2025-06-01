import os
from pathlib import Path

from dotenv import load_dotenv

from llm_doc_search_api import \
    __version__ as package_version  # Import from __init__.py

BASE_DIR = Path(__file__).resolve().parent.parent

dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

# --- Application Settings ---
APP_VERSION = package_version  # Use the version from the package
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Upload Settings ---
_upload_dir_name = os.getenv("UPLOAD_DIRECTORY", "uploads")
UPLOAD_DIR = str(BASE_DIR / _upload_dir_name)

# --- Chunking Settings ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# --- Embedding Model Settings ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Vector Store Settings (ChromaDB) ---
_chroma_data_path_name = os.getenv("CHROMA_DATA_PATH", "chroma_data")
CHROMA_DATA_PATH = str(BASE_DIR / _chroma_data_path_name)
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "document_chunks")

# --- API Server Settings ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))


# --- Logging Configuration ---
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": LOG_LEVEL,  # Use the LOG_LEVEL from .env
            "stream": "ext://sys.stdout",
        },
        # Optional: File handler
        # "file": {
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "formatter": "standard",
        #     "filename": str(BASE_DIR / "app.log"), # Log file in project root
        #     "maxBytes": 1024*1024*5,  # 5 MB
        #     "backupCount": 5,
        #     "level": LOG_LEVEL,
        # },
    },
    "root": {
        "handlers": ["console"],  # Add "file" here if you enable file handler
        "level": LOG_LEVEL,
    },
    # Configure log levels for specific libraries if needed
    "loggers": {
        "uvicorn.error": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "fastapi": {
            "handlers": ["console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        # Add other library-specific loggers here
        # "sentence_transformers": {
        #     "handlers": ["console"],
        #     "level": "WARNING", # Example: make sentence_transformers less verbose
        #     "propagate": False,
        # },
    },
}

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DATA_PATH, exist_ok=True)

# --- LLM Settings ---
_llm_model_path_str = os.getenv(
    "LLM_MODEL_PATH", "llm_models/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
)
LLM_MODEL_PATH = str(BASE_DIR / _llm_model_path_str)
LLM_N_CTX = int(os.getenv("LLM_N_CTX", 2048))
LLM_N_GPU_LAYERS = int(
    os.getenv("LLM_N_GPU_LAYERS", 0)
)  # 0 for CPU, set appropriately if you have GPU support
LLM_MAX_TOKENS = int(
    os.getenv("LLM_MAX_TOKENS", 256)
)  # Max new tokens for LLM response
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3))

if __name__ == "__main__":
    # Test print loaded configurations
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"APP_VERSION: {APP_VERSION}")
    print(f"LOG_LEVEL: {LOG_LEVEL}")
    print(f"UPLOAD_DIR: {UPLOAD_DIR}")
    print(f"CHUNK_SIZE: {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    print(f"EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
    print(f"CHROMA_DATA_PATH: {CHROMA_DATA_PATH}")
    print(f"CHROMA_COLLECTION_NAME: {CHROMA_COLLECTION_NAME}")
    print(f"API_HOST: {API_HOST}, API_PORT: {API_PORT}")
    print(f"LLM_MODEL_PATH: {LLM_MODEL_PATH}")
    print(f"LLM_N_CTX: {LLM_N_CTX}, LLM_N_GPU_LAYERS: {LLM_N_GPU_LAYERS}")
    print(f"LLM_MAX_TOKENS: {LLM_MAX_TOKENS}, LLM_TEMPERATURE: {LLM_TEMPERATURE}")
