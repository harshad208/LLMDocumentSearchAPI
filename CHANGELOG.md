# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
<!-- Keep this section for ongoing development -->

---

## [1.0.0] - 2025-06-01 <!-- Use today's date or your release date -->

### Added
- **File-based Persistence for Document Metadata**: Document metadata (statuses, filenames, etc.) is now persisted to a `documents_db.json` file. This prevents data loss on server restarts. Metadata is loaded on startup and saved on shutdown and after relevant operations (upload, delete).
- **Type Hinting and MyPy Integration**: Added comprehensive type hints across the codebase and integrated MyPy for static type checking, improving code robustness and maintainability.
- **ChromaDB Data Path Creation**: The application now automatically creates the `CHROMA_DATA_PATH` directory if it doesn't exist, simplifying initial setup.
- More robust error handling and logging in API routes, especially for document listing and retrieval.

### Changed
- **API Structure**: Refactored the main API file into `main.py`, `routes.py`, and `models.py` for better organization.
- **FastAPI Event Handling**: Migrated from deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` handlers to the modern `lifespan` context manager for application startup and shutdown logic.
- **Document List Endpoint**: Changed the endpoint for listing all documents from `/docs` to `/documents/list` to avoid conflict with FastAPI's default Swagger UI path.
- **ChromaDB Type Compliance**: Updated `vector_store.py` to use more specific type hints (`Where`, `Embedding`, `Metadatas`, etc.) from the `chromadb` library to satisfy MyPy and ensure correct usage of the ChromaDB API.
- **Upload Process Logic**: Refined the document upload process in `routes.py` to manage document metadata state more consistently and ensure it's saved to the persistence layer after each operation.
- Improved logging messages throughout the application for better traceability.

### Fixed
- **Circular Import Error**: Resolved a circular import dependency between `main.py` and `routes.py` by moving the shared `documents_db` to a separate `database.py` module.
- **MyPy Type Errors**: Addressed various type errors reported by MyPy, particularly in `vector_store.py` related to ChromaDB API calls (embeddings, metadatas, where clauses).
- Potential `NoneType` errors and improved handling of missing data when constructing `DocumentMetadata` objects.

### Deprecated
- The old `@app.on_event` style startup/shutdown handlers (now replaced by `lifespan`).

