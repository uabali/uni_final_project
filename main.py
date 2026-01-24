from src.loader import load_documents
import src.splitter as splitter
from src.vectorstore import create_embeddings, create_vectorstore
from src.llm import create_llm, create_trendyol_llm, create_openai_llm

from src.retriever import create_retriever, build_bm25_retriever
from src.reranker import create_reranker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
import sys
from dotenv import load_dotenv
from src.prompting import build_prompt, format_docs

def main():
    load_dotenv()
    print("--- RAG Pipeline Initializing ---")
    
    # Create embeddings first (needed for semantic splitter)
    embeddings = create_embeddings()
    
    # Load and process documents
    documents = load_documents()
    if not documents:
        print("Warning: No documents found. Checking for existing vectorstore...")
        docs = []
    else:
        # Semantic splitter kullanmak için: method="semantic" ve embeddings parametresini ekle
        # Varsayılan: method="recursive" (hızlı)
        # Daha iyi sonuç için: method="semantic" (yavaş ama daha iyi retrieval)
        split_method = "recursive"  # "recursive" veya "semantic" olarak değiştirilebilir
        
        if split_method == "semantic":
            print("Semantic Splitter kullanılıyor (anlamsal bölme)...")
            docs = splitter.split_documents(documents, method="semantic", embeddings=embeddings)
        else:
            # Ders notlari gibi uzun PDF'ler icin daha kucuk recursive chunk
            # (daha az karisik icerik, daha iyi retrieval)
            print("Recursive Splitter kullanılıyor (hızlı bölme, chunk_size=600, overlap=100)...")
            docs = splitter.split_documents(
                documents,
                method="recursive",
                chunk_size=600,
                chunk_overlap=100,
            )
        
        print(f"Documents split into {len(docs)} chunks.")

    # Create vectorstore
    try:
        vectorstore = create_vectorstore(docs, embeddings)
    except Exception as e:
        print(f"Error during vectorstore initialization: {e}")
        sys.exit(1)

    # Create LLM
    llm_backend = os.getenv("LLM_BACKEND", "trendyol").strip().lower()
    if llm_backend == "openai":
        llm = create_openai_llm()
        print("LLM backend: OpenAI (cloud)")
    elif llm_backend == "vllm":
        llm = create_llm()
        print("LLM backend: vLLM (local)")
    else:
        llm = create_trendyol_llm()
        print("LLM backend: Trendyol (local)")

    # Build BM25 ONCE at startup (for hybrid search)
    bm25_retriever = None
    if docs:
        try:
            bm25_retriever = build_bm25_retriever(docs)
            print("BM25 retriever built for hybrid search.")
        except Exception as e:
            print(f"BM25 build failed (hybrid disabled): {e}")

    # Create Reranker (optional, for better retrieval accuracy)
    reranker = None
    try:
        reranker = create_reranker(device="cuda")
        print("Reranker model loaded successfully.")
    except Exception as e:
        print(f"Reranker load failed (reranking disabled): {e}")
        print("Note: Reranking requires 'sentence-transformers' package.")

    # Define prompt
    prompt = build_prompt()

    # Multi-query ayarı (True yaparak aktif edebilirsiniz)
    use_multi_query = False  # True yaparak Multi-query'yi aktif edin
    
    # Re-rank ayarı (True yaparak aktif edebilirsiniz)
    use_rerank = False  # True yaparak Re-ranking'i aktif edin (reranker gerekli)
    
    print("\n--- RAG Ready (Auto Strategy) ---")
    if use_multi_query:
        print("Multi-query: AKTIF (daha iyi retrieval, daha yavaş)")
    else:
        print("Multi-query: PASIF (hızlı)")
    
    if use_rerank and reranker:
        print("Re-ranking: AKTIF (%15-25 daha iyi accuracy)")
    else:
        print("Re-ranking: PASIF (hızlı)")
    
    print("(Type 'exit' to quit)")
    
    while True:
        try:
            query = input("\nKullanici: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "cikis"]:
                break

            # Create retriever per query (auto strategy selection + optional multi-query + optional rerank)
            retriever = create_retriever(
                vectorstore=vectorstore,
                question=query,
                bm25_retriever=bm25_retriever,
                use_multi_query=use_multi_query,
                llm=llm if use_multi_query else None,
                num_queries=3,  # Multi-query için alternatif soru sayısı
                use_rerank=use_rerank and reranker is not None,
                reranker=reranker,
                rerank_top_n=20  # Reranking için alınacak doküman sayısı
            )

            # Build LCEL chain
            rag_chain = (
                {
                    "question": RunnablePassthrough(),
                    "context": retriever | RunnableLambda(format_docs)
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            print("Cevap: ", end="", flush=True)
            for chunk in rag_chain.stream(query):
                print(chunk, end="", flush=True)
            print()  # Yeni satir
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Query error: {e}")

if __name__ == "__main__":
    main()

