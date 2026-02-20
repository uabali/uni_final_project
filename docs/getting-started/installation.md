# Kurulum

## Gereksinimler

| Gereksinim | Minimum | Not |
|------------|---------|-----|
| Python | 3.11+ | `uv` ile yonetilir |
| Qdrant | Docker | `localhost:6333` |
| GPU | Opsiyonel | vLLM/Trendyol backend icin CUDA gerekli |
| OpenAI API Key | Opsiyonel | Cloud LLM kullanacaksan |

## 1. Repo'yu Klonla

```bash
git clone https://github.com/uabali/RAG.git
cd RAG
```

## 2. Bagimliliklari Kur

=== "uv (onerilen)"

    ```bash
    uv sync
    ```

=== "pip"

    ```bash
    pip install .
    ```

## 3. Qdrant'i Baslat

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant
```

!!! tip "Kontrol"
    Qdrant'in calistigini dogrulamak icin: `curl http://localhost:6333/collections`

## 4. Ortam Degiskenlerini Ayarla

`.env` dosyasi olustur:

```bash
cp .env.example .env
```

Detaylar icin [Konfigurasyon](configuration.md) sayfasina bak.

## 5. Dokumanlari Ekle

`data/` klasorune PDF veya TXT dosyalarini koy:

```bash
mkdir -p data
cp /path/to/your/documents/*.pdf data/
```

## 6. Calistir

```bash
python main.py
```

!!! success "Hazir"
    `--- RAG Ready (Auto Strategy) ---` mesajini gordugunuzde sistem hazir demektir.
