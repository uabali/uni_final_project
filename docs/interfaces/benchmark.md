# Benchmark

Performans olcum araci. RAG pipeline'i bir dataset uzerinde calistirip latency, throughput ve GPU istatistiklerini olcer.

## Calistirma

=== "OpenAI"

    ```bash
    python benchmark.py --dataset data/benchmark.jsonl --runs 3 --warmup 5 --backend openai
    ```

=== "vLLM"

    ```bash
    python benchmark.py --dataset data/benchmark.jsonl --runs 3 --warmup 5 --backend vllm
    ```

=== "vLLM + Streaming (TTFT)"

    ```bash
    python benchmark.py --dataset data/benchmark.jsonl --runs 3 --warmup 5 --backend vllm --stream
    ```

## Parametreler

| Parametre | Aciklama | Varsayilan |
|-----------|----------|-----------|
| `--dataset` | JSONL dataset yolu | (zorunlu) |
| `--runs` | Her sorgu icin tekrar sayisi | 1 |
| `--warmup` | Ilk N warmup sorgusu (hesaba katilmaz) | 0 |
| `--backend` | LLM backend | `.env`'den |
| `--stream` | Streaming modu (TTFT olcumu) | False |
| `--gpu-label` | GPU etiketi (karsilastirma icin) | "" |

## Ciktilar

| Dosya | Icerik |
|-------|--------|
| `logs/benchmark_results.jsonl` | Her sorgu icin detayli olcumler |
| `logs/benchmark_results_summary.json` | p50/p95 ozet istatistikler |

## Ornek Sonuclar (OpenAI, RTX 5070 Ti)

| Metrik | p50 | p95 |
|--------|-----|-----|
| **Toplam latency** | 1,373 ms | 2,815 ms |
| **Retrieval** | 107 ms | 166 ms |
| **LLM** | 1,266 ms | 2,726 ms |

!!! note "Gozlem"
    Retrieval (Qdrant + embedding) cok hizli (~100ms). Darbogazin neredeyse tamami LLM API latency'sinde.

## GPU Stats

Benchmark sirasinda `nvidia-smi` ile GPU kullanim oranlari, VRAM, power ve sicaklik olculur.

!!! warning "Onemli"
    Benchmark sirasinda `main.py` veya `streamlit` calistirmayin — kaynak cakismasi olusur.
