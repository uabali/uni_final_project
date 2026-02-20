# vLLM Serving Rehberi — Qwen3-8B-AWQ

Bu dokümantasyon, Qwen3-8B-AWQ modelini vLLM ile nasıl serve edeceğinizi açıklar.

## İki Serving Modu

### 1. Server Mode (Önerilen — Production)

vLLM ayrı bir process olarak çalışır, OpenAI-compatible API endpoint oluşturur.

**Avantajlar:**
- Model bir kez yüklenir, birden fazla client bağlanabilir
- Process isolation (agent crash olsa bile vLLM çalışmaya devam eder)
- Restart kolaylığı (sadece vLLM'i restart edersiniz)
- Birden fazla uygulama aynı modeli kullanabilir

**Kullanım:**

#### Adım 1: vLLM Server'ı Başlat

```bash
# Varsayılan ayarlarla (port 8000, 16k context)
./scripts/serve_vllm.sh

# Veya özel ayarlarla
./scripts/serve_vllm.sh --port 8001 --context 8192

# Reasoning mode'u kapatmak için
./scripts/serve_vllm.sh --no-reasoning
```

**Environment Variables ile:**

```bash
export VLLM_PORT=8000
export VLLM_CONTEXT_LEN=16384
export VLLM_GPU_MEMORY=0.85
export VLLM_ENABLE_REASONING=true

./scripts/serve_vllm.sh
```

#### Adım 2: `.env` Dosyasına Server URL Ekleyin

```ini
# vLLM Server URL (server mode için)
VLLM_SERVER_URL=http://localhost:8000/v1
```

#### Adım 3: Agent'ı Çalıştırın

```bash
python main.py
```

Agent otomatik olarak `VLLM_SERVER_URL`'e bağlanacak.

---

### 2. Embedded Mode (Development/Testing)

LangChain VLLM wrapper içinde model yüklenir (her Python process başladığında).

**Avantajlar:**
- Tek komutla çalışır (`python main.py`)
- Harici server gerekmez
- Hızlı test için uygun

**Dezavantajlar:**
- Her Python process başladığında model yüklenir (yavaş)
- Process isolation yok (agent crash olursa model de kapanır)
- Birden fazla client aynı modeli kullanamaz

**Kullanım:**

`.env` dosyasında `VLLM_SERVER_URL` **tanımlamayın** veya boş bırakın:

```ini
# VLLM_SERVER_URL tanımlama veya yorum satırına al
# VLLM_SERVER_URL=
```

Sonra direkt çalıştırın:

```bash
python main.py
```

---

## vLLM Server Komutları

### Manuel Başlatma (Script Kullanmadan)

```bash
vllm serve Qwen/Qwen3-8B-AWQ \
    --port 8000 \
    --quantization awq \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --trust-remote-code
```

### Parametreler

| Parametre | Açıklama | Varsayılan | Önerilen |
|-----------|----------|------------|----------|
| `--port` | Server port | 8000 | 8000 |
| `--quantization` | Quantization format | - | `awq` |
| `--gpu-memory-utilization` | GPU memory kullanım oranı | 0.9 | `0.85` (10GB için güvenli) |
| `--max-model-len` | Maksimum context length | 4096 | `16384` (16k) |
| `--enable-reasoning` | Thinking mode aktif | False | `True` (agentic task'lar için) |
| `--reasoning-parser` | Reasoning parser tipi | - | `deepseek_r1` (Qwen3 için) |
| `--trust-remote-code` | Remote code execution | False | `True` (Qwen3 için gerekli) |

### Server'ı Durdurma

```bash
# Ctrl+C ile
# Veya process ID ile
ps aux | grep vllm
kill <PID>
```

---

## VRAM Kullanımı (10GB Limit)

| Context Length | Tahmini VRAM | Durum |
|----------------|--------------|-------|
| 8k | ~6-7 GB | Rahat |
| 16k | ~7-8 GB | Güvenli (önerilen) |
| 32k | ~10-11 GB | Sınırda (OOM riski) |

**Öneri:** 10GB VRAM için `--max-model-len 16384` kullanın.

---

## API Endpoint Kullanımı

vLLM server başladıktan sonra OpenAI-compatible API endpoint oluşur:

**Base URL:** `http://localhost:8000/v1`

**Test:**

```bash
# Health check
curl http://localhost:8000/health

# Model listesi
curl http://localhost:8000/v1/models

# Chat completion (test)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [{"role": "user", "content": "Merhaba!"}],
    "max_tokens": 100
  }'
```

---

## Troubleshooting

### 1. "CUDA out of memory" Hatası

**Çözüm:** `--gpu-memory-utilization` değerini düşürün:

```bash
./scripts/serve_vllm.sh --gpu-memory 0.75
```

Veya context length'i azaltın:

```bash
./scripts/serve_vllm.sh --context 8192
```

### 2. "Model not found" Hatası

**Çözüm:** Model ilk çalıştırmada otomatik indirilir. HuggingFace token gerekebilir:

```bash
export HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 3. "Connection refused" Hatası (Server Mode)

**Çözüm:** vLLM server'ın çalıştığından emin olun:

```bash
# Server'ı başlatın
./scripts/serve_vllm.sh

# Başka bir terminal'de kontrol edin
curl http://localhost:8000/health
```

### 4. Reasoning Mode Çalışmıyor

**Çözüm:** vLLM >= 0.8.5 gerekli:

```bash
pip install --upgrade vllm>=0.8.5
```

---

## Performans Optimizasyonu

### Qwen3 İçin Önerilen Sampling Parametreleri

```python
temperature=0.7      # Non-thinking mode için
top_p=0.95           # Nucleus sampling
top_k=20             # Top-k sampling
presence_penalty=1.5 # Quantized modeller için önerilir (repetitive tool call'ları kırar)
```

Bu parametreler `src/llm.py` içinde varsayılan olarak ayarlanmıştır.

---

## Production Deployment

### systemd Service (Linux)

`/etc/systemd/system/vllm-qwen3.service`:

```ini
[Unit]
Description=vLLM Server - Qwen3-8B-AWQ
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/RAG
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/path/to/RAG/scripts/serve_vllm.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Kullanım:**

```bash
sudo systemctl enable vllm-qwen3
sudo systemctl start vllm-qwen3
sudo systemctl status vllm-qwen3
```

### Docker (Opsiyonel)

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install vllm>=0.8.5

EXPOSE 8000
CMD ["vllm", "serve", "Qwen/Qwen3-8B-AWQ", \
     "--port", "8000", \
     "--quantization", "awq", \
     "--gpu-memory-utilization", "0.85", \
     "--max-model-len", "16384", \
     "--enable-reasoning", \
     "--reasoning-parser", "deepseek_r1"]
```

---

## Kaynaklar

- [Qwen3-8B-AWQ Model Card](https://huggingface.co/Qwen/Qwen3-8B-AWQ)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3 vLLM Deployment Guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
