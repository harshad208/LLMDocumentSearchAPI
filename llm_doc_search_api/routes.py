import logging
import os
import shutil
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from . import config
from .chunking import get_text_chunks_simple
from .database import documents_db, save_db_to_file
from .embedding_generator import generate_embeddings
from .llm_handler import construct_prompt_for_qa, generate_llm_response
from .models import (DocumentMetadata, SearchQuery, SearchResponse,
                     SearchResultItem)
from .text_extractor import (SUPPORTED_EXTENSIONS, extract_text_from_file,
                             is_supported_file)
from .vector_store import (add_chunks_to_collection, delete_document_chunks,
                           query_collection)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "healthy", "message": "Welcome to the LLM Document Search API!"}


@router.post("/upload", response_model=DocumentMetadata, tags=["Document Management"])
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Upload request received for file: {file.filename}")
    if not file or not file.filename:
        logger.warning("Upload attempt with no file or no filename.")
        raise HTTPException(
            status_code=400, detail="No file uploaded or filename missing."
        )

    if not is_supported_file(file.filename):
        logger.warning(f"Unsupported file type uploaded: {file.filename}")
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.filename}. Supported types are: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    doc_id = str(uuid.uuid4())
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    file_location = os.path.join(config.UPLOAD_DIR, f"{doc_id}_{file.filename}")

    metadata_obj = DocumentMetadata(
        doc_id=doc_id,
        filename=file.filename,
        content_type=str(file.content_type),
        status="initializing",
        file_path=file_location,
    )

    documents_db[doc_id] = metadata_obj.model_dump()
    # save_db_to_file() # Optional: save initial state, or wait till end of successful processing

    extracted_text = ""
    chunks = []

    try:
        metadata_obj.status = "saving_file"
        documents_db[doc_id].update(metadata_obj.model_dump())
        # save_db_to_file() # Optional: save at each significant step

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)  # type: ignore
        metadata_obj.status = "file_saved"
        documents_db[doc_id].update(metadata_obj.model_dump())
        logger.info(f"File '{file.filename}' (ID: {doc_id}) saved to '{file_location}'")

        metadata_obj.status = "extracting_text"
        documents_db[doc_id].update(metadata_obj.model_dump())
        extracted_text = extract_text_from_file(file_location)

        if extracted_text.startswith("[Error extracting text"):
            metadata_obj.status = "extraction_error"
            metadata_obj.error_message = extracted_text
            logger.error(f"Text extraction failed for {doc_id}: {extracted_text}")
        else:
            metadata_obj.status = "text_extracted"
            metadata_obj.extracted_text_preview = (
                (extracted_text[:200] + "...")
                if len(extracted_text) > 200
                else extracted_text
            )
            logger.info(f"Text extracted successfully for {doc_id}.")
        documents_db[doc_id].update(metadata_obj.model_dump())

        if metadata_obj.status == "text_extracted":
            metadata_obj.status = "chunking"
            documents_db[doc_id].update(metadata_obj.model_dump())
            chunks = get_text_chunks_simple(
                extracted_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP
            )
            if not chunks:
                metadata_obj.num_chunks = 0
                metadata_obj.status = "chunking_failed_empty"
                logger.warning(f"No chunks generated for {doc_id}.")
            else:
                metadata_obj.num_chunks = len(chunks)
                metadata_obj.status = "chunked"
                logger.info(
                    f"Text for {doc_id} split into {metadata_obj.num_chunks} chunks."
                )
            documents_db[doc_id].update(metadata_obj.model_dump())

        if metadata_obj.status == "chunked" and chunks:
            metadata_obj.status = "embedding"
            documents_db[doc_id].update(metadata_obj.model_dump())
            embeddings_array = generate_embeddings(chunks)

            if embeddings_array is not None and len(embeddings_array) > 0:
                metadata_obj.embedding_dim = embeddings_array.shape[1]
                try:
                    add_chunks_to_collection(doc_id, chunks, embeddings_array.tolist())
                    metadata_obj.status = "ready"
                    logger.info(
                        f"Chunks for {doc_id} added to vector store. Status: ready."
                    )
                except Exception as e:
                    metadata_obj.status = "vector_storage_failed"
                    metadata_obj.error_message = f"Failed to store chunks: {str(e)}"
                    logger.error(
                        f"Error storing chunks for {doc_id}: {e}", exc_info=True
                    )
            else:
                metadata_obj.status = "embedding_failed"
                metadata_obj.error_message = "Failed to generate embeddings."
                logger.error(f"Embedding generation failed for {doc_id}.")
            documents_db[doc_id].update(metadata_obj.model_dump())

    except Exception as e:

        metadata_obj.status = "upload_pipeline_error"
        metadata_obj.error_message = str(e)
        logger.error(f"Unhandled error during upload for {doc_id}: {e}", exc_info=True)
        if os.path.exists(file_location) and file_location.startswith(
            str(config.UPLOAD_DIR)
        ):
            try:
                os.remove(file_location)
                logger.info(
                    f"Cleaned up file {file_location} for failed upload {doc_id}"
                )
            except OSError as ose:
                logger.error(
                    f"Error removing file {file_location} during cleanup: {ose}",
                    exc_info=True,
                )
        metadata_obj.file_path = None

        if metadata_obj.status not in [
            "initializing",
            "saving_file",
            "file_saved",
            "extracting_text",
            "extraction_error",
        ]:
            logger.info(
                f"Attempting to clean up vector store for failed upload {doc_id}"
            )
            try:
                delete_document_chunks(doc_id)
            except Exception as ve:
                logger.error(
                    f"Error cleaning up vector store for {doc_id}: {ve}", exc_info=True
                )
        documents_db[doc_id].update(metadata_obj.model_dump())

    finally:
        if file and hasattr(file, "file") and file.file and not file.file.closed:  # type: ignore
            await file.close()

        if doc_id in documents_db:
            documents_db[doc_id].update(metadata_obj.model_dump())
        else:
            documents_db[doc_id] = metadata_obj.model_dump()

        save_db_to_file()
        logger.info(
            f"Final state for doc_id {doc_id} saved to DB file. Status: {metadata_obj.status}"
        )
        logger.debug(
            f"Final metadata for doc_id {doc_id}: {metadata_obj.model_dump_json(indent=2)}"
        )

    return metadata_obj


@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def semantic_search_and_answer(query_data: SearchQuery):
    logger.info(
        f"Search & Answer request: Query='{query_data.query}', DocID='{query_data.doc_id}', TopN={query_data.top_n}"
    )
    if not query_data.query:
        logger.warning("Search attempt with empty query.")
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    if query_data.doc_id:
        if query_data.doc_id not in documents_db:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{query_data.doc_id}' not found.",
            )
        if documents_db[query_data.doc_id].get("status") != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Document with ID '{query_data.doc_id}' is not ready for search. Status: {documents_db[query_data.doc_id].get('status')}",
            )

    query_embedding_array = generate_embeddings([query_data.query])
    if query_embedding_array is None or len(query_embedding_array) == 0:
        logger.error(f"Could not generate embedding for query: {query_data.query}")
        raise HTTPException(
            status_code=500, detail="Could not generate embedding for the query."
        )
    query_embedding_list = query_embedding_array[0].tolist()

    retrieved_chunks_raw = query_collection(
        query_embedding=query_embedding_list,
        n_results=query_data.top_n,
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
        context_texts = [
            result.document_text
            for result in formatted_search_results
            if result.document_text
        ]
        if context_texts:
            prompt_for_llm = construct_prompt_for_qa(query_data.query, context_texts)
            logger.info("Sending prompt to LLM for answer generation...")
            llm_response_text = generate_llm_response(prompt_for_llm)
        else:
            logger.info(
                "No text content found in search results to use as context for LLM."
            )
            llm_response_text = "Could not generate an answer as no relevant text content was found in the search results."
    else:
        if (
            query_data.doc_id
            and documents_db.get(query_data.doc_id, {}).get("status") == "ready"
        ):

            logger.info(
                f"No relevant chunks found in document {query_data.doc_id} for the query."
            )
            llm_response_text = f"No relevant information found within document '{documents_db[query_data.doc_id].get('filename')}' to answer the question."
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


@router.get(
    "/documents/list",
    response_model=List[DocumentMetadata],
    tags=["Document Management"],
)
async def list_uploaded_documents():
    logger.info("Request to list all documents.")
    logger.info(f"LIST_DOCS: Current documents_db keys: {list(documents_db.keys())}")
    if not documents_db:
        logger.warning(
            "LIST_DOCS: documents_db is empty when /documents/list is called."
        )
    else:
        logger.debug(f"LIST_DOCS: documents_db content: {documents_db}")

    try:
        valid_meta_list = []
        for doc_id, meta_dict in documents_db.items():
            if isinstance(meta_dict, dict):
                try:
                    valid_meta_list.append(DocumentMetadata(**meta_dict))
                except Exception as e_pydantic:
                    logger.error(
                        f"LIST_DOCS: Pydantic validation error for doc_id {doc_id}: {e_pydantic}. Skipping item. Data: {meta_dict}"
                    )
            else:
                logger.error(
                    f"LIST_DOCS: Item for doc_id {doc_id} is not a dict: {type(meta_dict)}. Skipping item."
                )

        logger.info(
            f"LIST_DOCS: Successfully created {len(valid_meta_list)} DocumentMetadata objects."
        )
        return valid_meta_list
    except Exception as e:
        logger.error(
            f"LIST_DOCS: General error creating DocumentMetadata list: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Error retrieving document list.")


@router.get(
    "/docs/{doc_id}", response_model=DocumentMetadata, tags=["Document Management"]
)
async def get_document_metadata_by_id(doc_id: str):
    logger.info(f"Request for metadata of document ID: {doc_id}")
    if doc_id not in documents_db:
        logger.warning(f"Document ID {doc_id} not found for metadata request.")
        raise HTTPException(status_code=404, detail="Document not found")

    meta_dict = documents_db[doc_id]
    if not isinstance(meta_dict, dict):
        logger.error(
            f"Data for doc_id {doc_id} in documents_db is not a dict: {type(meta_dict)}"
        )
        raise HTTPException(
            status_code=500, detail="Internal server error: Invalid document data."
        )
    try:
        return DocumentMetadata(**meta_dict)
    except Exception as e_pydantic:
        logger.error(
            f"GET_DOC_META: Pydantic validation error for doc_id {doc_id}: {e_pydantic}. Data: {meta_dict}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error: Invalid document metadata format.",
        )


@router.delete("/docs/{doc_id}", status_code=200, tags=["Document Management"])
async def delete_document_and_chunks(doc_id: str):
    logger.info(f"Request to delete document ID: {doc_id}")
    if doc_id not in documents_db:
        logger.warning(f"Document ID {doc_id} not found for deletion.")
        raise HTTPException(status_code=404, detail="Document not found")

    doc_meta_dict = documents_db.get(doc_id, {})
    file_path = doc_meta_dict.get("file_path")
    try:
        delete_document_chunks(doc_id)
        logger.info(f"Chunks for doc_id {doc_id} deleted from vector store.")
    except Exception as e:
        logger.error(
            f"Error deleting chunks from vector store for doc_id {doc_id}: {e}",
            exc_info=True,
        )

    if file_path and isinstance(file_path, str) and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"File {file_path} deleted successfully.")
        except OSError as e:
            logger.error(f"Error deleting file {file_path}: {e}", exc_info=True)

    del documents_db[doc_id]
    save_db_to_file()
    logger.info(f"Metadata for doc_id {doc_id} deleted and DB file saved.")

    return {
        "doc_id": doc_id,
        "message": "Document and associated data deleted successfully.",
    }


@router.get(
    "/docs/{doc_id}/text", response_model=Dict[str, str], tags=["Document Management"]
)
async def get_extracted_document_text(doc_id: str):
    logger.info(f"Request for full text of document ID: {doc_id}")
    if doc_id not in documents_db:
        logger.warning(f"Document ID {doc_id} not found for text request.")
        raise HTTPException(status_code=404, detail="Document not found")

    doc_meta_dict = documents_db[doc_id]
    if not isinstance(doc_meta_dict, dict):
        logger.error(
            f"Data for doc_id {doc_id} in documents_db is not a dict: {type(doc_meta_dict)}"
        )
        raise HTTPException(
            status_code=500, detail="Internal server error: Invalid document data."
        )

    valid_statuses_for_text_extraction = [
        "text_extracted",
        "chunked",
        "embedding",
        "ready",
        "embedding_failed",
        "vector_storage_failed",
        "chunking_failed_empty",
        "file_saved",
    ]
    current_status = doc_meta_dict.get("status")
    if (
        current_status not in valid_statuses_for_text_extraction
        and current_status != "extraction_error"
    ):
        logger.warning(
            f"Text not available for document {doc_id} due to status: {current_status}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Text not yet extracted or extraction failed for document {doc_id}. Status: {current_status}",
        )

    file_path = doc_meta_dict.get("file_path")
    if not file_path or not isinstance(file_path, str) or not os.path.exists(file_path):
        logger.error(
            f"Source file for document {doc_id} not found or invalid path: {file_path}"
        )

        if current_status == "extraction_error" and doc_meta_dict.get("error_message"):
            raise HTTPException(
                status_code=404,
                detail=f"Source file for document {doc_id} not found. Extraction previously failed: {doc_meta_dict.get('error_message')}",
            )
        raise HTTPException(
            status_code=404,
            detail=f"Source file for document {doc_id} not found on server.",
        )

    try:
        full_text = extract_text_from_file(file_path)
        if full_text.startswith("[Error extracting text"):
            logger.error(f"Failed to re-extract text for doc {doc_id}: {full_text}")
            # Optionally update status in db if re-extraction fails
            # documents_db[doc_id]["status"] = "extraction_error_on_retry"
            # documents_db[doc_id]["error_message"] = full_text
            # save_db_to_file()
            raise HTTPException(
                status_code=500, detail=f"Failed to re-extract text: {full_text}"
            )

        logger.info(f"Successfully retrieved full text for doc {doc_id}.")
        return {
            "doc_id": doc_id,
            "filename": str(doc_meta_dict.get("filename", "unknown_filename")),
            "extracted_text": full_text,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving text for document {doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving text for document {doc_id}: {str(e)}",
        )
