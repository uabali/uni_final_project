"""
Text Splitting Module

Splits loaded documents into smaller, coherent chunks for embedding and retrieval.
Chunk quality is the single biggest lever for RAG accuracy.
"""

import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False

PDF_SEPARATORS = [
    "\n\n\n",
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
]


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(\n\s*){3,}", "\n\n", text)
    return text.strip()


def split_documents(
    documents,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    method: str = "recursive",
    embeddings=None,
):
    if not documents:
        return []

    for doc in documents:
        doc.page_content = _clean_text(doc.page_content)

    if method == "semantic" and SEMANTIC_CHUNKER_AVAILABLE:
        if embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},
            )
        try:
            text_splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.92,
            )
        except TypeError:
            text_splitter = SemanticChunker(embeddings=embeddings)
        print("Splitter: semantic")
    else:
        if method == "semantic" and not SEMANTIC_CHUNKER_AVAILABLE:
            print("Warning: SemanticChunker unavailable, falling back to recursive.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=PDF_SEPARATORS,
            strip_whitespace=True,
            add_start_index=True,
            length_function=len,
        )
        print(f"Splitter: recursive (size={chunk_size}, overlap={chunk_overlap})")

    docs = text_splitter.split_documents(documents)

    for idx, doc in enumerate(docs):
        doc.metadata["chunk_index"] = idx

    print(f"Chunks created: {len(docs)}")
    return docs
