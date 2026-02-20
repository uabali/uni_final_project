# Loader

Dokuman yukleme modulu. PDF ve TXT dosyalarini LangChain `Document` objelerine cevirir.

## Kullanim

```python
from src.loader import load_documents, load_single_document

# Tum dosyalari yukle (baslangic)
documents = load_documents("data/")

# Tek dosya yukle (incremental)
docs = load_single_document("data/ders_notlari.pdf")
```

## API Referansi

::: src.loader
