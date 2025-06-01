import logging
import os
import shutil
import uuid
from logging.config import dictConfig
from typing import Any, Dict, List

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import config
from .llm_handler import construct_prompt_for_qa, generate_llm_response
from .llm_handler import get_llm as get_llm_model_instance

dictConfig(config.LOGGING_CONFIG)

logger = logging.getLogger(__name__)

from .chunking import get_text_chunks_simple
from .embedding_generator import generate_embeddings
from .embedding_generator import get_model as get_embedding_model
from .text_extractor import (SUPPORTED_EXTENSIONS, extract_text_from_file,
                             is_supported_file)
from .vector_store import (add_chunks_to_collection, delete_document_chunks,
                           get_collection_count, query_collection)



# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Document Search API",
    description=f"API for uploading documents ({', '.join(SUPPORTED_EXTENSIONS)}) and performing semantic search.",
    version=config.APP_VERSION,
)

# In-memory storage for document metadata ONLY
documents_db: Dict[str, Dict[str, Any]] = {}


# --- Pydantic Models (no change from previous step) ---
class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    content_type: str
    status: str = "uploaded"
    file_path: str | None = None
    extracted_text_preview: str | None = None
    num_chunks: int | None = None
    embedding_dim: int | None = None
    error_message: str | None = None


class SearchQuery(BaseModel):
    query: str
    doc_id: str | None = None
    top_n: int = 3


class SearchResultItem(BaseModel):
    chunk_id: str
    document_text: str
    distance: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    search_results: List[SearchResultItem]  # Renamed from 'results' for clarity
    llm_answer: str | None = None  # Add field for LLM's answer


# --- Application Startup Logic ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    try:
        get_embedding_model()  # Pre-load embedding model
        logger.info(f"Embedding model '{config.EMBEDDING_MODEL_NAME}' loaded.")

        # Attempt to pre-load LLM model
        get_llm_model_instance()  # This will load the LLM
        logger.info(f"LLM model from '{config.LLM_MODEL_PATH}' loaded.")

        logger.info(
            f"Initial ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' count: {get_collection_count()}"
        )
    except Exception as e:
        logger.error(
            f"Startup Error: Could not pre-load models or init Chroma: {e}",
            exc_info=True,
        )
        # Note: If LLM loading fails here, get_llm() will try again on first request that needs it.
    logger.info("Application startup complete.")


# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "healthy", "message": "Welcome to the LLM Document Search API!"}


@app.post("/upload", response_model=DocumentMetadata, tags=["Document Management"])
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Upload request received for file: {file.filename}")
    if not file:
        logger.warning("Upload attempt with no file.")
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if not is_supported_file(file.filename):
        logger.warning(f"Unsupported file type uploaded: {file.filename}")
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.filename}. Supported types are: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    doc_id = str(uuid.uuid4())
    # Use UPLOAD_DIR from config
    file_location = os.path.join(config.UPLOAD_DIR, f"{doc_id}_{file.filename}")

    metadata = DocumentMetadata(
        doc_id=doc_id,
        filename=file.filename,
        content_type=file.content_type,
        status="saving",
        file_path=file_location,
    )
    documents_db[doc_id] = metadata.model_dump()
    logger.debug(
        f"Initial metadata for doc_id {doc_id}: {metadata.model_dump_json(indent=2)}"
    )

    extracted_text = ""
    chunks = []

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        metadata.status = "saved"
        documents_db[doc_id].update(metadata.model_dump())
        logger.info(f"File '{file.filename}' (ID: {doc_id}) saved to '{file_location}'")

        metadata.status = "extracting_text"
        documents_db[doc_id].update(metadata.model_dump())
        extracted_text = extract_text_from_file(
            file_location
        )  # This function also needs logging

        if extracted_text.startswith("[Error extracting text"):
            metadata.status = "extraction_error"
            metadata.error_message = extracted_text
            logger.error(
                f"Text extraction failed for {file.filename} (ID: {doc_id}): {extracted_text}"
            )
        else:
            metadata.status = "text_extracted"
            metadata.extracted_text_preview = (
                (extracted_text[:200] + "...")
                if len(extracted_text) > 200
                else extracted_text
            )  # Shorter preview for logs
            documents_db[doc_id].update(metadata.model_dump())
            logger.info(
                f"Text extracted successfully for '{file.filename}' (ID: {doc_id})."
            )

            if metadata.status == "text_extracted":
                metadata.status = "chunking"
                documents_db[doc_id].update(metadata.model_dump())
                # Use CHUNK_SIZE, CHUNK_OVERLAP from config
                chunks = get_text_chunks_simple(
                    extracted_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP
                )
                if not chunks:
                    metadata.num_chunks = 0
                    metadata.status = "chunking_failed_empty"
                    logger.warning(
                        f"No chunks generated for {file.filename} (ID: {doc_id})."
                    )
                else:
                    metadata.num_chunks = len(chunks)
                    metadata.status = "chunked"
                    logger.info(
                        f"Text from '{file.filename}' (ID: {doc_id}) split into {metadata.num_chunks} chunks."
                    )
                documents_db[doc_id].update(metadata.model_dump())

            if metadata.status == "chunked" and chunks:
                metadata.status = "embedding"
                documents_db[doc_id].update(metadata.model_dump())

                embeddings_array = generate_embeddings(
                    chunks
                )  # This function also needs logging

                if embeddings_array is not None and len(embeddings_array) > 0:
                    metadata.embedding_dim = embeddings_array.shape[1]
                    try:
                        add_chunks_to_collection(
                            doc_id, chunks, embeddings_array.tolist()
                        )  # Needs logging
                        metadata.status = "ready"
                        logger.info(
                            f"Chunks for '{file.filename}' (ID: {doc_id}) added to vector store. Status: ready."
                        )
                    except Exception as e:
                        metadata.status = "vector_storage_failed"
                        metadata.error_message = (
                            f"Failed to store chunks in vector store: {str(e)}"
                        )
                        logger.error(
                            f"Error storing chunks for {doc_id} in vector store: {e}",
                            exc_info=True,
                        )
                else:
                    metadata.status = "embedding_failed"
                    metadata.error_message = "Failed to generate embeddings."
                    logger.error(
                        f"Embedding generation failed for {file.filename} (ID: {doc_id})."
                    )
                documents_db[doc_id].update(metadata.model_dump())

    except Exception as e:
        metadata.status = "upload_error"
        metadata.error_message = str(e)
        logger.error(
            f"Unhandled error during upload process for {file.filename} (ID: {doc_id}): {e}",
            exc_info=True,
        )
        if os.path.exists(file_location) and file_location.startswith(
            str(config.UPLOAD_DIR)
        ):
            try:
                os.remove(file_location)
            except OSError as ose:
                logger.error(
                    f"Error removing file {file_location} during cleanup: {ose}",
                    exc_info=True,
                )
        metadata.file_path = None
        if metadata.status not in [
            "saving",
            "saved",
            "extracting_text",
            "extraction_error",
        ]:
            logger.info(
                f"Attempting to clean up vector store for failed upload of doc_id {doc_id}"
            )
            delete_document_chunks(doc_id)
    finally:
        if file and not file.file.closed:  # Ensure file is closed
            await file.close()

    documents_db[doc_id].update(metadata.model_dump())
    logger.debug(
        f"Final metadata for doc_id {doc_id}: {metadata.model_dump_json(indent=2)}"
    )
    return metadata


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def semantic_search_and_answer(query_data: SearchQuery):  # Renamed for clarity
    logger.info(
        f"Search & Answer request: Query='{query_data.query}', DocID='{query_data.doc_id}', TopN={query_data.top_n}"
    )
    if not query_data.query:
        logger.warning("Search attempt with empty query.")
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # 1. Generate embedding for the query
    query_embedding_array = generate_embeddings([query_data.query])
    if query_embedding_array is None or len(query_embedding_array) == 0:
        logger.error(f"Could not generate embedding for query: {query_data.query}")
        raise HTTPException(
            status_code=500, detail="Could not generate embedding for the query."
        )
    query_embedding_list = query_embedding_array[0].tolist()

    # 2. Query ChromaDB for relevant chunks
    retrieved_chunks_raw = query_collection(
        query_embedding=query_embedding_list,
        n_results=query_data.top_n,  # Use top_n from query for retrieval
        doc_id=query_data.doc_id,
    )
    logger.debug(f"Raw search results from vector store: {retrieved_chunks_raw}")

    formatted_search_results = [
        SearchResultItem(
            chunk_id=res["id"],
            document_text=res.get("document_text", ""),
            distance=res["distance"],
            metadata=res["metadata"],
        )
        for res in retrieved_chunks_raw
    ]

    llm_response_text = None
    if formatted_search_results:
        # 3. Construct prompt with retrieved chunks as context
        context_texts = [
            result.document_text
            for result in formatted_search_results
            if result.document_text
        ]
        if context_texts:  # Proceed only if there's actual text context
            prompt_for_llm = construct_prompt_for_qa(query_data.query, context_texts)

            # 4. Generate answer using LLM
            logger.info("Sending prompt to LLM for answer generation...")
            llm_response_text = generate_llm_response(prompt_for_llm)
        else:
            logger.info(
                "No text content found in search results to use as context for LLM."
            )
            llm_response_text = "Could not generate an answer as no relevant text content was found in the search results."
    else:
        logger.info("No search results found to provide context to LLM.")
        llm_response_text = "No relevant documents found to answer the question."

    logger.info(
        f"Returning {len(formatted_search_results)} search results and LLM answer for query '{query_data.query}'."
    )
    return SearchResponse(
        query=query_data.query,
        search_results=formatted_search_results,
        llm_answer=llm_response_text,
    )


@app.get("/docs", response_model=List[DocumentMetadata], tags=["Document Management"])
async def list_uploaded_documents():
    logger.info("Request to list all documents.")
    return [DocumentMetadata(**meta) for meta in documents_db.values()]


@app.get(
    "/docs/{doc_id}", response_model=DocumentMetadata, tags=["Document Management"]
)
async def get_document_metadata(doc_id: str):
    logger.info(f"Request for metadata of document ID: {doc_id}")
    if doc_id not in documents_db:
        logger.warning(f"Document ID {doc_id} not found for metadata request.")
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentMetadata(**documents_db[doc_id])


@app.delete("/docs/{doc_id}", status_code=200, tags=["Document Management"])
async def delete_document_and_chunks(doc_id: str):
    logger.info(f"Request to delete document ID: {doc_id}")
    if doc_id not in documents_db:
        logger.warning(f"Document ID {doc_id} not found for deletion.")
        raise HTTPException(status_code=404, detail="Document not found")

    doc_meta = documents_db[doc_id]
    file_path = doc_meta.get("file_path")

    try:
        delete_document_chunks(doc_id)  # This function needs logging
        logger.info(f"Chunks for doc_id {doc_id} deleted from vector store.")
    except Exception as e:
        logger.error(
            f"Error deleting chunks from vector store for doc_id {doc_id}: {e}",
            exc_info=True,
        )

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"File {file_path} deleted successfully.")
        except OSError as e:
            logger.error(f"Error deleting file {file_path}: {e}", exc_info=True)

    del documents_db[doc_id]
    logger.info(f"Metadata for doc_id {doc_id} deleted.")
    return {
        "doc_id": doc_id,
        "message": "Document and associated data deleted successfully.",
    }


@app.get(
    "/docs/{doc_id}/text", response_model=Dict[str, str], tags=["Document Management"]
)
async def get_extracted_document_text(doc_id: str):
    logger.info(f"Request for full text of document ID: {doc_id}")
    # ... (rest of this endpoint's logic, add logging as needed) ...
    if doc_id not in documents_db:
        logger.warning(f"Document ID {doc_id} not found for text request.")
        raise HTTPException(status_code=404, detail="Document not found")

    doc_meta_dict = documents_db[doc_id]
    # Add more specific status checks if necessary based on your flow
    valid_statuses_for_text_extraction = [
        "text_extracted",
        "chunked",
        "embedding",
        "ready",
        "embedding_failed",
        "vector_storage_failed",
        "chunking_failed_empty",
    ]  # Ensure all relevant statuses are covered
    if doc_meta_dict.get("status") not in valid_statuses_for_text_extraction:
        logger.warning(
            f"Text not available for document {doc_id} due to status: {doc_meta_dict.get('status')}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Text not yet extracted or extraction failed for document {doc_id}. Status: {doc_meta_dict.get('status')}",
        )

    file_path = doc_meta_dict.get("file_path")
    if not file_path or not os.path.exists(file_path):
        logger.error(
            f"Source file for document {doc_id} not found at path: {file_path}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Source file for document {doc_id} not found on server.",
        )

    try:
        full_text = extract_text_from_file(file_path)  # This function needs logging
        if full_text.startswith("[Error extracting text"):
            logger.error(f"Failed to re-extract text for doc {doc_id}: {full_text}")
            raise HTTPException(
                status_code=500, detail=f"Failed to re-extract text: {full_text}"
            )
        logger.info(f"Successfully retrieved full text for doc {doc_id}.")
        return {
            "doc_id": doc_id,
            "filename": doc_meta_dict.get("filename"),
            "extracted_text": full_text,
        }
    except Exception as e:
        logger.error(f"Error retrieving text for document {doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving text for document {doc_id}: {str(e)}",
        )

if __name__ == "__main__":
    # Use config for host and port if running directly
    logger.info(f"Starting Uvicorn server on {config.API_HOST}:{config.API_PORT}")
    import uvicorn

    uvicorn.run(
        "llm_doc_search_api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_config=None,
    )
