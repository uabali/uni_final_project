"""
Qdrant Collection'ı Sıfırlama Scripti
"""
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Mevcut collection'ları listele
collections = client.get_collections()
print(f"Mevcut collection'lar: {[c.name for c in collections.collections]}")

# rag_collection varsa sil
collection_name = "rag_collection"
try:
    client.delete_collection(collection_name)
    print(f"✅ '{collection_name}' collection'ı silindi.")
except Exception as e:
    print(f"⚠️ Collection silinirken hata: {e}")

# Tekrar kontrol et
collections = client.get_collections()
print(f"Kalan collection'lar: {[c.name for c in collections.collections]}")
