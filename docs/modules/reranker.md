# Reranker

Cross-encoder ile retrieval sonuclarini yeniden siralayan modul. %15-25 accuracy artisi saglar. TTL cache ve adaptive skip ile optimize edilmistir.

## Kullanim

```python
from src.reranker import create_reranker, rerank_documents

# Reranker olustur
reranker = create_reranker(device="cuda")          # varsayilan model
reranker = create_reranker("fast", device="cuda")  # hizli model

# Dokumanlari yeniden sirala
reranked = rerank_documents(
    query="Sprint nedir?",
    documents=docs,
    reranker=reranker,
    top_k=6,
)
```

## Model Secenekleri

| Alias | Model | Boyut | Hiz | Accuracy |
|-------|-------|-------|-----|----------|
| `default` | BAAI/bge-reranker-base | ~400 MB | Normal | Yuksek |
| `fast` | cross-encoder/ms-marco-MiniLM-L-6-v2 | ~80 MB | Hizli | Orta |

## API Referansi

::: src.reranker.create_reranker

::: src.reranker.rerank_documents

::: src.reranker.create_rerank_retriever
