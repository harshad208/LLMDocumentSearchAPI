import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config

logger = logging.getLogger(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        logger.info(
            f"Loading sentence transformer model: {config.EMBEDDING_MODEL_NAME}..."
        )
        try:
            _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            logger.info(
                f"Sentence transformer model '{config.EMBEDDING_MODEL_NAME}' loaded successfully."
            )
        except Exception as e:
            logger.error(
                f"Error loading sentence transformer model {config.EMBEDDING_MODEL_NAME}: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to load SentenceTransformer model: {config.EMBEDDING_MODEL_NAME}"
            ) from e
    return _model


def generate_embeddings(texts: List[str]) -> np.ndarray | None:
    try:
        model = get_model()
        if model is None:
            logger.error("Embedding model is not available.")
            return None

        logger.info(
            f"Generating embeddings for {len(texts)} chunks using {config.EMBEDDING_MODEL_NAME}..."
        )

        embeddings = model.encode(
            texts, show_progress_bar=(logger.getEffectiveLevel() == logging.DEBUG)
        )
        logger.info(
            f"Successfully generated {len(embeddings)} embeddings with dimension {embeddings.shape[1] if embeddings is not None and embeddings.ndim > 1 else 'N/A'}."
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # For standalone testing, ensure logging is configured
    from logging.config import dictConfig

    dictConfig(config.LOGGING_CONFIG)

    logger.info("Testing embedding generator...")
    test_texts = [
        "This is the first document.",
        "This document is the second document.",
    ]
    embeddings_array = generate_embeddings(test_texts)
    if embeddings_array is not None:
        logger.info(f"Shape of embeddings array: {embeddings_array.shape}")
    else:
        logger.error("Failed to generate embeddings in standalone test.")
