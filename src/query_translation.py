"""
Query Translation Modülü

Multi-query, Step-back, HyDE gibi query transformation tekniklerini içerir.
Bu modül, kullanıcı sorularını daha etkili retrieval için dönüştürür.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List


def generate_multi_queries(
    question: str, 
    llm, 
    num_queries: int = 3
) -> List[str]:
    """
    Bir soruyu birden fazla farklı şekilde ifade eder (Multi-query).
    
    Bu teknik, bir soruyu farklı açılardan ve kelimelerle ifade ederek
    retrieval accuracy'yi artırır. Her alternatif soru için arama yapılır
    ve sonuçlar birleştirilir.
    
    Args:
        question: Orijinal kullanıcı sorusu
        llm: LLM modeli (query generation için)
        num_queries: Üretilecek alternatif soru sayısı (varsayılan: 3)
        
    Returns:
        List[str]: Orijinal soru + alternatif sorular listesi
        
    Örnek:
        >>> queries = generate_multi_queries("Python'da liste nasıl sıralanır?", llm)
        >>> # ["Python'da liste nasıl sıralanır?", "Python list sorting methods", 
        >>> #  "How to sort a list in Python", "Python liste sıralama yöntemleri"]
    """
    prompt_template = """Rephrase the user's question in {num_queries} different ways.
Each rephrased question must seek the same information but use different wording, \
different angles, or a different language (e.g. Turkish ↔ English).
Keep each question short and clear.

Original question: {question}

Generate exactly {num_queries} alternative questions, one per line.
Do NOT add numbering, bullets, or any explanation. Output only the questions.

Alternative questions:"""
    
    prompt = PromptTemplate(
        input_variables=["question", "num_queries"],
        template=prompt_template
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"question": question, "num_queries": num_queries})
        
        # Satırlara böl ve temizle
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        
        # Gereksiz numaraları ve işaretleri temizle (örn: "1. ", "- ", vb.)
        cleaned_queries = []
        for q in queries:
            # Başındaki numara ve işaretleri kaldır
            q = q.lstrip("0123456789.-) ").strip()
            if q and len(q) > 5:  # Çok kısa olanları filtrele
                cleaned_queries.append(q)
        
        # Sadece istenen kadar al
        queries = cleaned_queries[:num_queries]
        
        # Orijinal soruyu başa ekle (önemli: ilk sırada olmalı)
        return [question] + queries
        
    except Exception as e:
        print(f"Multi-query generation error: {e}. Falling back to original question.")
        return [question]


def create_multi_query_retriever(
    vectorstore,
    question: str,
    llm,
    num_queries: int = 3,
    bm25_retriever=None,
    strategy="auto",
    base_k=6,
    **retriever_kwargs
):
    """
    Multi-query tekniği ile retriever oluşturur.
    
    Bu fonksiyon:
    1. Orijinal soruyu birden fazla alternatif soruya çevirir
    2. Her soru için arama yapar
    3. Sonuçları birleştirip tekrarları kaldırır
    
    Args:
        vectorstore: Qdrant vectorstore
        question: Orijinal kullanıcı sorusu
        llm: LLM modeli (query generation için)
        num_queries: Üretilecek alternatif soru sayısı
        bm25_retriever: BM25 retriever (hybrid search için)
        strategy: Arama stratejisi ("auto", "mmr", "similarity", "hybrid")
        base_k: Her query için getirilecek chunk sayısı
        **retriever_kwargs: Diğer retriever parametreleri
        
    Returns:
        Callable: Multi-query retriever fonksiyonu
    """
    from src.retriever import create_retriever
    from src.retriever import run_retriever
    
    # 1. Multi-query generation
    queries = generate_multi_queries(question, llm, num_queries=num_queries)
    
    print(f"Multi-query: searching with {len(queries)} query variants...")
    if len(queries) > 1:
        print(f"  Original: {queries[0]}")
        for i, q in enumerate(queries[1:], 1):
            print(f"  Variant {i}: {q}")
    
    # 2. Her query için retriever oluştur ve arama yap
    all_docs = []
    seen_ids = set()  # Tekrarları önlemek için
    
    for query in queries:
        retriever = create_retriever(
            vectorstore=vectorstore,
            question=query,
            bm25_retriever=bm25_retriever,
            strategy=strategy,
            base_k=base_k,
            **retriever_kwargs
        )
        
        # Arama yap
        docs = run_retriever(retriever, query)
        
        # Tekrarları filtrele (aynı içeriği farklı query'lerden gelmiş olabilir)
        for doc in docs:
            # Doc'un unique ID'si (içerik + metadata kombinasyonu)
            doc_id = hash((doc.page_content[:100], doc.metadata.get("source", "")))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
    
    # 3. Sonuçları skorlarına göre sırala (ilk gelenler daha yüksek skorlu)
    # Not: LangChain retriever'lar zaten skorlarına göre sıralı döner
    
    # 4. Top k kadar al (base_k * num_queries kadar olabilir, ama base_k ile sınırla)
    final_docs = all_docs[:base_k * 2]  # Biraz fazla al, sonra kısaltılabilir
    
    print(f"Found {len(all_docs)} unique documents, using {len(final_docs)}.")
    
    # 5. Retriever-like fonksiyon döndür
    def multi_query_retriever(query: str):
        """Multi-query retriever - query parametresi göz ardı edilir (zaten işlendi)"""
        return final_docs
    
    return multi_query_retriever
