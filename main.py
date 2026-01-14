from src.loader import load_documents
import src.splitter as splitter
from src.vectorstore import create_embeddings, create_vectorstore
from src.llm import create_llm
from src.retriever import create_retriever, build_bm25_retriever
from src.reranker import create_reranker
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import sys

def main():
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
    llm = create_llm()

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
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a retrieval-based question answering assistant.

Your task:
- Answer the QUESTION using ONLY the given CONTEXT.
- Do NOT use external knowledge.
- Do NOT make assumptions.
- If the answer is not clearly found in the CONTEXT, respond EXACTLY with:
  "Baglamda cevap bulunamadi."

Answer rules:
- Language: TURKISH (ASCII only) for normal sentences.
- Do NOT use Turkish characters in normal Turkish text (c, g, s, i, o, u only).
- KEEP abbreviations (STT, HTTP, API, OAuth, JWT, JSON, HTTP, TCP/IP, vb.) and code/command/technical terms
  (function adlari, class isimleri, degisken isimleri, API endpointleri, CLI komutlari, dosya adlari) EXACTLY as they appear in the question or context.
- Do NOT translate or change these abbreviations and technical terms; only if needed you can briefly explain them in parentheses in Turkish.

Answer style:
- First, give a SHORT and CLEAR direct answer in 1–2 sentences.
- Then, ONLY IF NECESSARY, add a few short bullet points to clarify important details (still concise, no long paragraphs).
- If the user question has multiple parts, make sure each part is answered explicitly (kisa ama aciklayici).
- Do NOT repeat the same sentence or idea multiple times. If a fact is already stated once, do not restate it again with similar wording.

How to think (do NOT write these steps in the answer):
1) Identify if the question has multiple parts.
2) For each part, search the CONTEXT for a direct answer.
3) Combine answers ONLY if all parts are found in the CONTEXT.
4) If any part is missing, return "Baglamda cevap bulunamadi."
5) When you see abbreviations or technical terms (like STT, HTTP, API, OAuth),
   keep them EXACTLY as in the question/context and DO NOT translate them.
6) Use only the information that is explicitly present in the CONTEXT. Do NOT add generic explanations or comments that are not clearly supported by the CONTEXT.

### EXAMPLES

Example 1:
CONTEXT:
Daily Scrum is a time-boxed event. It lasts 15 minutes and is held daily to synchronize the team.

QUESTION:
Daily Scrum ne kadar surer?

ANSWER:
Daily Scrum 15 dakika surer.
- Her gun yapilir ve takim uyumunu saglamak icin kullanilir.

Example 2:
CONTEXT:
Daily Scrum is a daily event. It lasts 15 minutes. Its purpose is to align the team and plan the next 24 hours.

QUESTION:
Daily Scrum ne kadar surer ve neden yapilir?

ANSWER:
Daily Scrum 15 dakika surer ve takimin gunluk calismasini hizalamak icin yapilir.
- Takim, sonraki 24 saate dair planlarini paylasir.
- Ilerleme ve engeller kisa sekilde konusulur.

Example 3:
CONTEXT:
Sprint Planning defines what will be done in the sprint.

QUESTION:
Sprint Planning kimler tarafindan yapilir?

ANSWER:
Baglamda cevap bulunamadi.

Example 4 (abbreviation and technical term policy):
CONTEXT:
The system uses an STT (Speech To Text) pipeline before sending text to the RAG module.

QUESTION:
Bu sistemde STT nasil kullaniliyor?

ANSWER:
Sistemde STT, kullanicinin konusmasini metne cevirmek icin kullanilir ve bu metin daha sonra RAG modulu tarafindan islenir.
- STT (Speech To Text) asamasi sadece sesin metne donusumunden sorumludur.
- RAG modulu ise olusan metni dokumanlarla eslestirip cevap uretir.

### NOW ANSWER

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    )

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
                    "context": retriever
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

