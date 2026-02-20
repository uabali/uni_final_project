# LLM

Dil modeli modulu. Uc farkli backend destekler: OpenAI (cloud), vLLM (local), Trendyol (Turkce local).

## Kullanim

```python
from src.llm import create_llm, create_trendyol_llm, create_openai_llm

# OpenAI (cloud)
llm = create_openai_llm()

# vLLM (local, CUDA gerekli)
llm = create_llm()

# Trendyol (local, CUDA gerekli)
llm = create_trendyol_llm()
```

## Backend Karsilastirmasi

| Ozellik | OpenAI | vLLM | Trendyol |
|---------|--------|------|----------|
| Model | gpt-4o-mini | LLaMA-3.1-8B AWQ | Trendyol-LLM-8B-T1 |
| Calisma yeri | Cloud | Local GPU | Local GPU |
| VRAM | 0 | ~5 GB | ~9 GB |
| Turkce | Iyi | Iyi | En iyi (fine-tuned) |
| Latency | ~1.3s (API) | Dusuk (local) | Dusuk (local) |
| Maliyet | Ucretli (token) | Ucretsiz | Ucretsiz |
| Context | 128k | 4096 | 8192 (native 32k) |

## API Referansi

::: src.llm.create_llm

::: src.llm.create_trendyol_llm

::: src.llm.create_openai_llm
