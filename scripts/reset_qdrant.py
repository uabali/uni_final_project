"""
Qdrant Collection Reset Script
"""
import os

from qdrant_client import QdrantClient

qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
collection_name = os.getenv("QDRANT_COLLECTION", "rag_collection").strip()

client = QdrantClient(url=qdrant_url)

# List existing collections
collections = client.get_collections()
print(f"Existing collections: {[c.name for c in collections.collections]}")
print(f"Target: {collection_name} | URL: {qdrant_url}")

# Delete collection if it exists
try:
    client.delete_collection(collection_name)
    print(f"✅ Collection '{collection_name}' deleted.")
except Exception as e:
    print(f"⚠️ Error while deleting collection: {e}")

# Verify
collections = client.get_collections()
print(f"Remaining collections: {[c.name for c in collections.collections]}")
