"""
Metin Parçalama Modülü (Splitter Module)

Bu modül, yüklenen uzun dokümanları daha küçük, anlamlı parçalara (chunk) böler.
Retrieval başarısı için kritik öneme sahiptir.

Desteklenen Yöntemler:
- recursive: Karakter sayısına göre bölme (hızlı, varsayılan)
- semantic: Anlamsal sınırlarda bölme (daha iyi retrieval, yavaş)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional

# SemanticChunker import (langchain_experimental gerekli)
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False
    print("Uyarı: SemanticChunker bulunamadı. 'langchain-experimental' paketini yükleyin.")

def split_documents(
    documents,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    method: str = "recursive",
    embeddings=None,
):
    """
    Dokümanları küçük parçalara böler.
    
    Args:
        documents (list): Bölünecek Document objeleri listesi.
        chunk_size (int): Her parçanın maksimum karakter sayısı (recursive için, varsayılan: 800).
        chunk_overlap (int): Parçalar arası örtüşme miktarı (recursive için, varsayılan: 120).
        method (str): Bölme yöntemi - "recursive" (hızlı) veya "semantic" (daha iyi).
        embeddings: Embedding modeli (semantic için gerekli, None ise otomatik oluşturulur).
                             
    Returns:
        list: Bölünmüş (chunked) Document objeleri.
        
    Bağlantılı Olduğu Yerler:
        - vectorstore.py: Bölünen parçalar Embedding işlemine girer.
    """
    if not documents:
        return []
    
    # Semantic Splitter kullanılıyorsa
    if method == "semantic":
        if not SEMANTIC_CHUNKER_AVAILABLE:
            print("Hata: SemanticChunker mevcut değil. Recursive splitter'a geri dönülüyor...")
            method = "recursive"
        else:
            # Embeddings yoksa oluştur
            if embeddings is None:
                print("Semantic Splitter için embedding modeli oluşturuluyor...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3",
                    model_kwargs={"device": "cuda"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            
            # SemanticChunker oluştur
            # breakpoint_threshold_type: "percentile" (benzerlik yüzdesi) veya "standard_deviation"
            # breakpoint_threshold_amount: Eşik değeri (0.95 = %95 benzerlik altında böl)
            try:
                text_splitter = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=0.95  # %95 benzerlik eşiği
                )
                print("Semantic Splitter kullanılıyor (anlamsal bölme)...")
            except TypeError as e:
                # Eğer parametreler uyumsuzsa, basit versiyonu dene
                print(f"SemanticChunker parametre hatası: {e}. Basit konfigürasyon deneniyor...")
                text_splitter = SemanticChunker(embeddings=embeddings)
                print("Semantic Splitter kullanılıyor (varsayılan ayarlarla)...")
    
    # Recursive Character Splitter (varsayılan veya fallback)
    if method == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        print(f"Recursive Splitter kullanılıyor (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    
    docs = text_splitter.split_documents(documents)
    print(f"Toplam {len(docs)} chunk oluşturuldu.")
    return docs
