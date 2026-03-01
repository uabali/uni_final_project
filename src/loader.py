"""
Doküman Yükleme Modülü (Loader Module)

Bu modül, PDF ve TXT dosyalarını okuyarak LangChain Document objelerine çevirir.
Ana işlevi, verileri diskten okuyup işleme (splitting/embedding) hazır hale getirmektir.
"""

import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# PDF okuyucunun gereksiz uyarılarını gizle
logging.getLogger("pypdf").setLevel(logging.ERROR)

logger = logging.getLogger("rag.loader")

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_single_document(file_path: str) -> List[Document]:
    """
    Tek bir dosyayı yükler (PDF veya TXT).

    Args:
        file_path: Yüklenecek dosyanın tam yolu.

    Returns:
        Yüklenen Document objeleri listesi.
    """
    if not os.path.exists(file_path):
        logger.warning("Dosya bulunamadı: %s", file_path)
        return []

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning("Desteklenmeyen format: %s (%s)", ext, file_path)
        return []

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        docs = loader.load()
        logger.info("Yüklendi: %s (%d sayfa/parça)", os.path.basename(file_path), len(docs))
        return docs
    except Exception as e:
        logger.error("Yükleme hatası %s: %s", file_path, e)
        return []


def load_documents(data_dir: str = "data") -> List[Document]:
    """
    Belirtilen klasördeki TÜM desteklenen dosyaları yükler.

    Args:
        data_dir: Dosyaların bulunduğu klasör yolu (varsayılan: 'data').

    Returns:
        Tüm yüklenen Document objeleri listesi.
    """
    if not os.path.exists(data_dir):
        logger.warning("Veri klasörü bulunamadı: %s — oluşturuluyor", data_dir)
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

    logger.info("Toplam %d dosyadan %d doküman yüklendi", loaded_count, len(documents))
    return documents
