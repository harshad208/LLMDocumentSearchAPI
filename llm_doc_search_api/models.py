from typing import Any, Dict, List, Optional

from pydantic import BaseModel


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
