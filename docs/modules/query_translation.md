# Query Translation

Multi-query ve diger query transformation tekniklerini iceren modul. Kullanici sorularini farkli sekillerde ifade ederek retrieval basarisini arttirir.

## Kullanim

```python
from src.query_translation import generate_multi_queries, create_multi_query_retriever

# Alternatif soru uret
queries = generate_multi_queries("Sprint nedir?", llm, num_queries=3)
# ["Sprint nedir?", "Sprint'in tanimi nedir?", "What is a Sprint?", ...]

# Multi-query retriever olustur
retriever = create_multi_query_retriever(
    vectorstore=vectorstore,
    question="Sprint nedir?",
    llm=llm,
    num_queries=3,
)
```

## Nasil Calisir

```mermaid
flowchart TB
    Q["Orijinal Soru"] --> LLM["LLM<br/>(N varyasyon uret)"]
    LLM --> Q1["Varyasyon 1"]
    LLM --> Q2["Varyasyon 2"]
    LLM --> Q3["Varyasyon 3"]
    Q & Q1 & Q2 & Q3 --> R["Her biri icin<br/>ayri arama"]
    R --> D["Deduplicate"]
    D --> F["Final docs"]
```

## API Referansi

::: src.query_translation
