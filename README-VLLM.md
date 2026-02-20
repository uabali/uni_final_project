# Qwen3-8B-AWQ ile vLLM Kullanımı — Hızlı Başlangıç

Bu dokümantasyon, Qwen3-8B-AWQ modelini vLLM ile nasıl kullanacağınızı açıklar.

## 🚀 Hızlı Başlangıç

### Yöntem 1: Server Mode (Önerilen)

**Adım 1:** vLLM server'ı başlatın (ayrı terminal):

```bash
./scripts/serve_vllm.sh
```

**Adım 2:** `.env` dosyasına server URL ekleyin:

```ini
VLLM_SERVER_URL=http://localhost:6365/v1
```

**Adım 3:** Agent'ı çalıştırın:

```bash
python main.py
```

---

> **Not:** Agent (tool-calling) için embedded mode desteklenmez. `bind_tools` yalnızca chat modellerde mevcuttur; LangChain'in VLLM wrapper'ı BaseLLM'dir. Server mode zorunludur.

---

## 📋 Gereksinimler

- **CUDA GPU** (10GB+ VRAM önerilir)
- **vLLM >= 0.8.5**: `pip install vllm>=0.8.5`
- **Python >= 3.11**

---

## ⚙️ Yapılandırma

### Server Mode Parametreleri

Environment variables ile:

```bash
export VLLM_PORT=6365
export VLLM_CONTEXT_LEN=16384
export VLLM_GPU_MEMORY=0.85

./scripts/serve_vllm.sh
```

Veya script parametreleri ile:

```bash
./scripts/serve_vllm.sh --port 6367 --context 8192 --gpu-memory 0.75
```

### LLM Parametreleri

`src/llm.py` içinde varsayılan parametreler Qwen3 için optimize edilmiştir:

- `temperature=0.7` (non-thinking mode)
- `top_p=0.95`
- `presence_penalty=1.5` (quantized modeller için önerilir)
- `max_tokens=768` (default, env ile override edilebilir)

Runtime tuning için env değişkenleri:

```bash
export LLM_MAX_TOKENS=768
export LLM_ENABLE_THINKING=false
export RAG_USE_RERANK=true
export RAG_RERANK_TOP_N=10
export RAG_RETRIEVAL_STRATEGY=hybrid
export RAG_BASE_K=8
export QDRANT_AUTO_REINDEX=smart   # true | false | smart
```

`QDRANT_AUTO_REINDEX=smart` davranisi:
- ilk calistirmada fingerprint dosyasi olusur ve index yenilenebilir
- sonraki calistirmalarda dokuman degismediyse yeniden indexleme yapmaz
- dokuman degistiyse otomatik reindex yapar

---

## 🔍 Kontrol ve Test

### Server Durumunu Kontrol Etme

```bash
# Health check
curl http://localhost:6365/health

# Model listesi
curl http://localhost:6365/v1/models
```

### Test Sorgusu

```bash
curl http://localhost:6365/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [{"role": "user", "content": "Merhaba, Türkçe konuşabilir misin?"}],
    "max_tokens": 100
  }'
```

---

## 📊 VRAM Kullanımı

| Context Length | VRAM Kullanımı | Durum |
|----------------|----------------|-------|
| 8k | ~6-7 GB | ✅ Rahat |
| 16k | ~7-8 GB | ✅ Güvenli (önerilen) |
| 32k | ~10-11 GB | ⚠️ Sınırda |

---

## 🐛 Sorun Giderme

### "CUDA out of memory" Hatası

```bash
# GPU memory kullanımını düşürün
./scripts/serve_vllm.sh --gpu-memory 0.75

# Veya context length'i azaltın
./scripts/serve_vllm.sh --context 8192
```

### "Connection refused" Hatası

vLLM server'ın çalıştığından emin olun:

```bash
# Başka bir terminal'de
./scripts/serve_vllm.sh

# Kontrol edin
curl http://localhost:6365/health
```

### Qdrant UI/API gelmiyor (`localhost:6333`)

Qdrant ayri bir servis olarak calismalidir:

```bash
docker ps | rg qdrant
curl http://localhost:6333/collections
```

> Not: Yeni surumlerde root endpoint web panel yerine JSON donebilir. API kontrolu icin `/collections` daha guvenilir.

### Model İndirme Sorunu

HuggingFace token gerekebilir:

```bash
export HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## 📚 Detaylı Dokümantasyon

- [vLLM Serving Rehberi](docs/vllm-serving.md) — Detaylı serving modları ve production deployment
- [Qwen3-8B-AWQ Model Card](https://huggingface.co/Qwen/Qwen3-8B-AWQ)

---

## 🎯 Özellikler

- ✅ **Türkçe Desteği**: 100+ dil desteği, Türkçe için optimize edilmiş
- ✅ **Tool Calling**: Hermes-style function calling (agentic task'lar için)
- ✅ **Thinking Control**: Varsayılan kapalı (`LLM_ENABLE_THINKING=false`), istenirse env ile açılabilir
- ✅ **32k Context**: Native 32k context (YaRN ile 131k'ya çıkarılabilir)
- ✅ **AWQ Quantization**: 4-bit quantization ile %3-4 kalite kaybı, ~3x hız artışı

---

## 💡 İpuçları

1. **Production için Server Mode kullanın** — Process isolation ve restart kolaylığı
2. **10GB VRAM için 16k context önerilir** — 32k sınırda
3. **Presence penalty 1.5 kullanın** — Repetitive tool call'ları kırar
4. **Latency kritikse thinking'i kapali tutun** — Tool-calling daha stabil ve daha hizli olur

---

## 🔗 Kaynaklar

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3 vLLM Deployment](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
