"""
Doküman Yükleme Modülü (Loader Module)

Bu modül, PDF ve TXT dosyalarını okuyarak LangChain Document objelerine çevirir.
Ana işlevi, verileri diskten okuyup işleme (splitting/embedding) hazır hale getirmektir.
"""

import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# PDF okuyucunun gereksiz uyarılarını gizle
logging.getLogger("pypdf").setLevel(logging.ERROR)

def load_single_document(file_path):
    """
    Tek bir dosyayı yükler (PDF veya TXT).
    
    Args:
        file_path (str): Yüklenecek dosyanın tam yolu.
        
    Returns:
        list: Yüklenen Document objeleri listesi.
        
    Kullanım Yeri:
        - Streamlit arayüzünde tek tek dosya yüklerken (Incremental Indexing).
        - load_documents fonksiyonu tarafından çağrılır.
    """
    if not os.path.exists(file_path):
        print(f"Hata: Dosya bulunamadi {file_path}")
        return []
    
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            print(f"Desteklenmeyen format: {ext}")
            return []
            
        docs = loader.load()
        print(f"Yuklendi: {os.path.basename(file_path)}")
        return docs
    except Exception as e:
        print(f"Yukleme hatasi {file_path}: {e}")
        return []

def load_documents(data_dir="data"):
    """
    Belirtilen klasördeki TÜM desteklenen dosyaları yükler.
    
    Args:
        data_dir (str): Dosyaların bulunduğu klasör yolu (varsayılan: 'data').
        
    Returns:
        list: Tüm yüklenen Document objeleri listesi.
        
    Kullanım Yeri:
        - Sistemin ilk başlatılmasında tüm veriyi topluca yüklemek için kullanılır.
    """
    if not os.path.exists(data_dir):
        print(f"Uyari: Veri klasoru bulunamadi {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        return []
    
    supported_extensions = [".pdf", ".txt"]
    documents = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext not in supported_extensions:
            continue
            
        docs = load_single_document(file_path)
        documents.extend(docs)
            
    return documents
