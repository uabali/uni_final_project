"""
Qdrant Collection'ı Sıfırlama Scripti
"""
import os

from qdrant_client import QdrantClient

qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
collection_name = os.getenv("QDRANT_COLLECTION", "rag_collection").strip()

client = QdrantClient(url=qdrant_url)

# Mevcut collection'ları listele
collections = client.get_collections()
print(f"Mevcut collection'lar: {[c.name for c in collections.collections]}")
print(f"Hedef: {collection_name} | URL: {qdrant_url}")

# rag_collection varsa sil
try:
    client.delete_collection(collection_name)
    print(f"✅ '{collection_name}' collection'ı silindi.")
except Exception as e:
    print(f"⚠️ Collection silinirken hata: {e}")

# Tekrar kontrol et
collections = client.get_collections()
print(f"Kalan collection'lar: {[c.name for c in collections.collections]}")
