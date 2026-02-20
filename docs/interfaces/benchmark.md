# Benchmark

Performans olcum araci. RAG pipeline'i bir dataset uzerinde calistirip latency, throughput ve GPU istatistiklerini olcer.

## Calistirma

=== "Agent (onerilen)"

    ```bash
    python scripts/benchmark.py --dataset data/benchmark.jsonl --mode agent --runs 2 --warmup 3
    ```

=== "Pipeline (eski davranis)"

    ```bash
    python scripts/benchmark.py --dataset data/benchmark.jsonl --mode pipeline --runs 2 --warmup 3
    ```

=== "Eszamanli kullanici yuk testi"

    ```bash
    python scripts/benchmark.py --dataset data/benchmark.jsonl --mode agent --concurrency 8 --runs 3 --warmup 3
    ```

=== "Pipeline + Streaming (TTFT)"

    ```bash
    python scripts/benchmark.py --dataset data/benchmark.jsonl --mode pipeline --stream
    ```

## Parametreler

| Parametre | Aciklama | Varsayilan |
|-----------|----------|-----------|
| `--dataset` | JSONL dataset yolu | (zorunlu) |
| `--mode` | `agent` veya `pipeline` | `agent` |
| `--concurrency` | Eszamanli virtual user sayisi | `1` |
| `--runs` | Her sorgu icin tekrar sayisi | 1 |
| `--warmup` | Ilk N warmup sorgusu (hesaba katilmaz) | 1 |
| `--backend` | LLM backend | `.env`'den |
| `--stream` | Streaming modu (TTFT olcumu) | False |
| `--gpu-label` | GPU etiketi (karsilastirma icin) | "" |

!!! info "Not"
    `--stream` su an sadece `--mode pipeline` ile desteklenir.

## Ciktilar

| Dosya | Icerik |
|-------|--------|
| `logs/benchmark_results.jsonl` | Her sorgu icin detayli olcumler |
| `logs/benchmark_results_summary.json` | p50/p95 + success rate + achieved RPS |

## Turkce Dataset Hazirlama

Turkce benchmark semasi ve sampling plani icin:

- `docs/interfaces/benchmark-tr-dataset.md`
- `benchmarks/benchmark_tr_load.jsonl`
- `benchmarks/benchmark_tr_quality_template.jsonl`

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
