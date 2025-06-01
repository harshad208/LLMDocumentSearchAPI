import logging
import uuid
from typing import Any, Dict, List, Sequence

import chromadb

from . import config

logger = logging.getLogger(__name__)

collection = None

try:
    client = chromadb.PersistentClient(path=config.CHROMA_DATA_PATH)
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
    logger.info(
        f"ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' loaded/created at path '{config.CHROMA_DATA_PATH}'. Count: {collection.count()}"
    )
except Exception as e:
    logger.critical(
        f"CRITICAL: Error initializing ChromaDB client or collection: {e}",
        exc_info=True,
    )

    collection = None


def add_chunks_to_collection(
    doc_id: str, chunks: List[str], embeddings: List[Sequence[float]]
):
    if collection is None:
        logger.error("ChromaDB collection is not available. Cannot add chunks.")
        raise RuntimeError("ChromaDB collection not initialized.")

    if not chunks or not embeddings:
        logger.warning("No chunks or embeddings provided to add_chunks_to_collection.")
        return

    if len(chunks) != len(embeddings):
        logger.error("Mismatch between number of chunks and embeddings.")
        raise ValueError("The number of chunks and embeddings must be the same.")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "doc_id": str(doc_id),
            "chunk_text": chunk_text,
            "chunk_index": int(i),
        }
        for i, chunk_text in enumerate(chunks)
    ]

    try:
        collection.add(
            embeddings=[list(e) for e in embeddings],  # type: ignore
            documents=chunks,
            metadatas=metadatas,  # type: ignore
            ids=ids,
        )
        logger.info(
            f"Added {len(chunks)} chunks for doc_id '{doc_id}' to ChromaDB collection '{config.CHROMA_COLLECTION_NAME}'."
        )
    except Exception as e:
        logger.error(
            f"Error adding chunks to ChromaDB for doc_id '{doc_id}': {e}", exc_info=True
        )
        raise


def query_collection(
    query_embedding: Sequence[float], n_results: int = 5, doc_id: str | None = None
) -> List[Dict[str, Any]]:
    if collection is None:
        logger.error("ChromaDB collection is not available. Cannot query.")
        raise RuntimeError("ChromaDB collection not initialized.")

    filter_metadata = {}
    if doc_id:
        filter_metadata = {"doc_id": doc_id}

    logger.debug(
        f"Querying ChromaDB with n_results={n_results}, filter={filter_metadata}"
    )
    try:
        results = collection.query(
            query_embeddings=[list(query_embedding)],  # type: ignore
            n_results=n_results,
            where=filter_metadata if filter_metadata else None,  # type: ignore
            include=["metadatas", "documents", "distances"],
        )

        processed_results = []
        if results and results.get("ids") and results["ids"][0]:
            num_found = len(results["ids"][0])
            logger.debug(f"ChromaDB query returned {num_found} results.")
            for i in range(num_found):
                processed_results.append(
                    {
                        "id": results["ids"][0][i],
                        "document_text": (
                            results["documents"][0][i]
                            if results["documents"] and results["documents"][0]
                            else None
                        ),
                        "metadata": (
                            results["metadatas"][0][i]
                            if results["metadatas"] and results["metadatas"][0]
                            else None
                        ),
                        "distance": (
                            results["distances"][0][i]
                            if results["distances"] and results["distances"][0]
                            else None
                        ),
                    }
                )
        else:
            logger.debug("ChromaDB query returned no results.")
        return processed_results
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}", exc_info=True)
        return []


def delete_document_chunks(doc_id: str):
    if collection is None:
        logger.error("ChromaDB collection is not available. Cannot delete chunks.")

        return

    try:
        logger.info(f"Attempting to delete chunks for doc_id '{doc_id}' from ChromaDB.")
        collection.delete(where={"doc_id": doc_id})
        logger.info(
            f"Successfully deleted chunks for doc_id '{doc_id}' from ChromaDB (if any existed)."
        )
    except Exception as e:
        logger.error(
            f"Error deleting chunks for doc_id '{doc_id}' from ChromaDB: {e}",
            exc_info=True,
        )


def get_collection_count() -> int:
    if collection is None:
        logger.error("ChromaDB collection is not available. Cannot get count.")
        return -1
    return collection.count()
