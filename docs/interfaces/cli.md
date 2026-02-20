# CLI (main.py)

Interaktif terminal arayuzu. Gelistirme ve debug icin birincil arayuz.

## Calistirma

```bash
python main.py
```

## Akis

1. `.env` yukler (`load_dotenv`)
2. Embedding modeli olusturur
3. `data/` klasorundeki dokumanlari yukler ve boler
4. Qdrant vectorstore olusturur/yukler
5. LLM backend secer (`LLM_BACKEND` env var)
6. BM25 index kurar
7. Reranker yukler (opsiyonel)
8. Interaktif loop baslatir

## Ornek Oturum

```
--- RAG Pipeline Initializing ---
Recursive Splitter kullanılıyor (hızlı bölme, chunk_size=600, overlap=100)...
Documents split into 142 chunks.
Mevcut Qdrant veritabani yukleniyor: rag_collection
LLM backend: OpenAI (cloud)
BM25 retriever built for hybrid search.

--- RAG Ready (Auto Strategy) ---
Multi-query: PASIF (hızlı)
Re-ranking: PASIF (hızlı)
(Type 'exit' to quit)

Kullanici: Daily Scrum kac dakika surer?
Cevap: Daily Scrum 15 dakika surer.
- Her gun yapilir ve takim uyumunu saglamak icin kullanilir.

Kullanici: exit
```

## Ayarlar

`main.py` icindeki degiskenleri degistirerek:

```python
use_multi_query = False  # True → Multi-query aktif
use_rerank = False       # True → Reranking aktif
split_method = "recursive"  # "semantic" → Semantic splitter
```
