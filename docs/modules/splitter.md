# Splitter

Metin parcalama modulu. Uzun dokumanlari retrieval icin optimize edilmis kucuk chunk'lara boler.

## Kullanim

```python
from src.splitter import split_documents

# Recursive (hizli, varsayilan)
chunks = split_documents(docs, method="recursive", chunk_size=600, chunk_overlap=100)

# Semantic (daha iyi retrieval, yavas)
chunks = split_documents(docs, method="semantic", embeddings=embeddings)
```

## Yontem Karsilastirmasi

| Ozellik | Recursive | Semantic |
|---------|-----------|----------|
| Hiz | Hizli | Yavas |
| Bolme mantigi | Karakter sayisi | Anlam siniri |
| Ek bagimlilik | Yok | `langchain-experimental` |
| Embedding gerekli mi | Hayir | Evet |
| Chunk boyutu | Sabit (600 char) | Degisken |

## API Referansi

::: src.splitter
