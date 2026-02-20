# Turkce Benchmark Dataset Plani

Bu sayfa, RAG sistemi icin **Turkce odakli** benchmark datasini hizli ve dogru sekilde hazirlamak icin pratik bir rehberdir.

## 1) Neden Iki Ayrı Dataset?

Tek dataset ile hem yuk testi hem kalite degerlendirmesi yapmak zor olur.

- **Load dataset**: latency, throughput, success rate olcmek icin.
- **Quality dataset**: retrieval ve cevap dogrulugu olcmek icin.

Bu repoda ikisini ayri tutmak en temiz yaklasimdir.

## 2) JSONL Sema (benchmark script ile uyumlu)

`scripts/benchmark.py` su alanlardan birini zorunlu okur:

- `input` **veya**
- `question` **veya**
- `query`

Opsiyonel ama kalite icin faydali alanlar:

- `output` veya `answer` (beklenen cevap)
- `scenario` (factual, multi_hop, no_answer, live, math)
- `difficulty` (easy, medium, hard)
- `requires_web` (true/false)
- `no_answer_expected` (true/false)
- `gold_sources` (beklenen dokuman adlari veya id listesi)
- `language` (`tr`)
- `tags` (serbest etiketler)

Not: Script bu ekstra alanlari zorunlu tutmaz, ama sonradan kalite analizi icin cok faydalidir.

## 3) Turkce Sampling Plani (onerilen dagilim)

Baslangic icin toplam **300** sorgu:

- `40%` Yerel factual (dokumanda birebir bulunan bilgi)
- `20%` Multi-hop / karsilastirma (2+ parcayi birlestirme)
- `15%` No-answer (dokumanda olmayan ama makul soru)
- `15%` Live/Web (guncel bilgi gerektiren soru)
- `10%` Tool/Math (hesap gerektiren basit sorgu)

Ek kalite kurallari:

- Sorularin en az `%70`i gercek kullanici dili gibi olsun (kisa, dogal, bazen eksik imla).
- En az `%20` typo/kolokyal varyasyon ekleyin (ornek: "nasil", "niye", "kac dk").
- En az `%15` soru uzun/komut benzeri olsun (detayli, baglamli).

## 4) Hedef Metrikler

### Yuk Testi

- `p50_total_ms`, `p95_total_ms`
- `achieved_rps`
- `success_rate`
- `error_count`

### Kalite Testi

- Top-k icinde dogru kaynak var mi? (`Recall@k` benzeri kontrol)
- Cevap beklenen bilgiyle uyumlu mu?
- `no_answer` senaryosunda halusinasyon yapıyor mu?
- `requires_web=true` senaryosunda dogru tool seciliyor mu?

## 5) Bu Repo Icin Hazir Dosyalar

Asagidaki dosyalar bu plana gore eklendi:

- `benchmarks/benchmark_tr_load.jsonl`
- `benchmarks/benchmark_tr_quality_template.jsonl`

Calistirma ornekleri:

```bash
uv run python scripts/benchmark.py --dataset benchmarks/benchmark_tr_load.jsonl --mode agent --concurrency 8 --runs 3 --warmup 3
```

```bash
uv run python scripts/benchmark.py --dataset benchmarks/benchmark_tr_quality_template.jsonl --mode agent --concurrency 1 --runs 1
```

Ikinci komut kalite datasetini teknik olarak calistirir; kalite skorlamayi ekstra analizle yorumlaman gerekir.
