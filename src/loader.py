"""
Document Loading Module (Loader Module)

This module reads PDF and TXT files and converts them to LangChain Document objects.
Its main function is to read data from disk and prepare it for processing (splitting/embedding).
"""

import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Suppress unnecessary warnings from PDF reader
logging.getLogger("pypdf").setLevel(logging.ERROR)

logger = logging.getLogger("rag.loader")

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_single_document(file_path: str) -> List[Document]:
    """
    Loads a single file (PDF or TXT).

    Args:
        file_path: Full path of the file to load.

    Returns:
        List of loaded Document objects.
    """
    if not os.path.exists(file_path):
        logger.warning("File not found: %s", file_path)
        return []

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning("Unsupported format: %s (%s)", ext, file_path)
        return []

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        docs = loader.load()
        logger.info("Loaded: %s (%d pages/chunks)", os.path.basename(file_path), len(docs))
        return docs
    except Exception as e:
        logger.error("Loading error %s: %s", file_path, e)
        return []


def load_documents(data_dir: str = "data") -> List[Document]:
    """
    Loads ALL supported files from the specified directory.

    Args:
        data_dir: Path to the directory containing files (default: 'data').

    Returns:
        List of all loaded Document objects.
    """
    if not os.path.exists(data_dir):
        logger.warning("Data directory not found: %s — creating it", data_dir)
        os.makedirs(data_dir, exist_ok=True)
        return []

    documents: List[Document] = []
    loaded_count = 0

    for root, _dirs, files in os.walk(data_dir):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            file_path = os.path.join(root, filename)
            docs = load_single_document(file_path)
            documents.extend(docs)
            if docs:
                loaded_count += 1

    logger.info("Loaded %d documents from %d files total", len(documents), loaded_count)
    return documents
