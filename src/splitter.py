"""
Metin Parçalama Modülü (Splitter Module)

Bu modül, yüklenen uzun dokümanları daha küçük, anlamlı parçalara (chunk) böler.
Retrieval başarısı için kritik öneme sahiptir.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=800, chunk_overlap=120):
    """
    Dokümanları küçük parçalara böler.
    
    Args:
        documents (list): Bölünecek Document objeleri listesi.
        chunk_size (int): Her parçanın maksimum karakter sayısı (varsayılan: 800).
        chunk_overlap (int): Parçalar arası örtüşme miktarı (varsayılan: 120).
                             Bu, bağlamın kopmamasını sağlar.
                             
    Returns:
        list: Bölünmüş (chunked) Document objeleri.
        
    Bağlantılı Olduğu Yerler:
        - vectorstore.py: Bölünen parçalar Embedding işlemine girer.
    """
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs
