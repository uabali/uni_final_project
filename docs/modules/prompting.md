# Prompting

Prompt ve context formatlama modulu. Merkezi prompt sablonu ve dokuman birlestirme fonksiyonu saglar.

## Kullanim

```python
from src.prompting import build_prompt, format_docs

# Prompt template
prompt = build_prompt()

# Dokumanlari birlestir
context_str = format_docs(documents)
```

## Prompt Kurallari

| Kural | Detay |
|-------|-------|
| Kaynak | Sadece verilen context |
| Hallucination | "Baglamda cevap bulunamadi." fallback |
| Dil | Turkce (ASCII), teknik terimler korunur |
| Stil | 1-2 cumle direkt + bullet points |
| Few-shot | 4 ornek (basit, cok parcali, cevapsiz, teknik) |

## API Referansi

::: src.prompting
