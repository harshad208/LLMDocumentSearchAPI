import logging
import os

import pdfplumber
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_text_from_file(file_path: str) -> str:
    logger.debug(f"Attempting to extract text from: {file_path}")
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    try:
        if extension == ".txt":
            return extract_text_from_txt(file_path)
        elif extension == ".pdf":
            return extract_text_from_pdf(file_path)
        elif extension == ".docx":
            return extract_text_from_docx(file_path)
        else:
            logger.warning(f"Unsupported file type for text extraction: {extension}")
            # Return a specific error message string that can be checked
            return f"[Error: Unsupported file type: {extension}]"
    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {e}", exc_info=True)
        return f"[Error extracting text from {os.path.basename(file_path)}: {e}]"


def extract_text_from_txt(file_path: str) -> str:
    logger.debug(f"Extracting text from TXT: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_pdf(file_path: str) -> str:
    logger.debug(f"Extracting text from PDF: {file_path}")
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            logger.debug(f"Extracted text from PDF page {i+1}/{len(pdf.pages)}")
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    logger.debug(f"Extracting text from DOCX: {file_path}")
    doc = DocxDocument(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)


def is_supported_file(filename: str) -> bool:
    _, extension = os.path.splitext(filename)
    return extension.lower() in SUPPORTED_EXTENSIONS
