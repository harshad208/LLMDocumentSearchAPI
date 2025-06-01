import logging
from typing import List

logger = logging.getLogger(__name__)


def get_text_chunks_simple(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    logger.debug(
        f"Starting chunking. Text length: {len(text)}, Chunk size: {chunk_size}, Overlap: {chunk_overlap}"
    )
    if chunk_overlap >= chunk_size:
        logger.error("Chunk overlap must be less than chunk size.")
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start_index = 0
    text_length = len(text)

    while start_index < text_length:
        end_index = start_index + chunk_size

        if end_index >= text_length:
            chunk = text[start_index:]
            if chunk.strip():
                chunks.append(chunk.strip())
            break

        actual_end_index = text.rfind(" ", start_index, end_index)
        if (
            actual_end_index == -1
            or actual_end_index < start_index + (chunk_size - chunk_overlap) // 2
        ):
            actual_end_index = end_index

        chunk = text[start_index:actual_end_index].strip()
        if chunk:
            chunks.append(chunk)

        next_start_index = actual_end_index - chunk_overlap
        if next_start_index <= start_index:
            next_start_index = actual_end_index

        start_index = next_start_index
        if start_index >= text_length:
            break

    final_chunks = [c for c in chunks if c]
    logger.debug(f"Chunking complete. Generated {len(final_chunks)} chunks.")
    return final_chunks
