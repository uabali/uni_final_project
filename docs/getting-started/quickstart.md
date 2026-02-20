# Ilk Calistirma

Kurulum tamamlandiktan sonra sistemi hizlica test etmek icin:

## 1. Qdrant'in Calistigini Dogrula

```bash
curl http://localhost:6333/collections
```

Beklenen cikti: `{"result":{"collections":[]},"status":"ok",...}`

## 2. Dokuman Ekle

`data/` klasorune en az 1 PDF veya TXT dosyasi koyun.

## 3. CLI ile Baslat

```bash
python main.py
```

Baslangic ciktisi:

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
```

## 4. Soru Sor

```
Kullanici: Daily Scrum kac dakika surer?
Cevap: Daily Scrum 15 dakika surer.
- Her gun yapilir ve takim uyumunu saglamak icin kullanilir.
```

## 5. Streamlit ile Baslat (Opsiyonel)

```bash
streamlit run streamlit/app.py --server.headless true
```

Tarayicida `http://localhost:8501` adresini acin.

!!! info "GPU Olmadan"
    GPU yoksa `LLM_BACKEND=openai` kullanin. Embedding modeli CPU'da calisir (daha yavas ama fonksiyonel).

## Qdrant'i Sifirla (Gerektiginde)

Tum veritabanini silip sifirdan baslamak icin:

```bash
python reset_qdrant.py
```
