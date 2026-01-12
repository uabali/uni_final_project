from src.loader import load_documents
import src.splitter as splitter
from src.vectorstore import create_embeddings, create_vectorstore
from src.llm import create_llm
from src.retriever import create_retriever, build_bm25_retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import sys

def main():
    print("--- RAG Pipeline Initializing ---")
    
    # Load and process documents
    documents = load_documents()
    if not documents:
        print("Warning: No documents found. Checking for existing vectorstore...")
        docs = []
    else:
        docs = splitter.split_documents(documents)
        print(f"Documents split into {len(docs)} chunks.")

    # Create embeddings and vectorstore
    try:
        embeddings = create_embeddings()
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
- Language: TURKISH (ASCII only)
- Do NOT use Turkish characters (c, g, s, i, o, u only)
- Answer must be SHORT, CLEAR, and DIRECT.

How to think (do NOT write these steps in the answer):
1) Identify if the question has multiple parts.
2) For each part, search the CONTEXT for a direct answer.
3) Combine answers ONLY if all parts are found in the CONTEXT.
4) If any part is missing, return "Baglamda cevap bulunamadi."

### EXAMPLES

Example 1:
CONTEXT:
Daily Scrum is a time-boxed event. It lasts 15 minutes and is held daily to synchronize the team.

QUESTION:
Daily Scrum ne kadar surer?

ANSWER:
15 dakika surer.

Example 2:
CONTEXT:
Daily Scrum is a daily event. It lasts 15 minutes. Its purpose is to align the team and plan the next 24 hours.

QUESTION:
Daily Scrum ne kadar surer ve neden yapilir?

ANSWER:
15 dakika surer ve takimin gunluk calismasini hizalamak icin yapilir.

Example 3:
CONTEXT:
Sprint Planning defines what will be done in the sprint.

QUESTION:
Sprint Planning kimler tarafindan yapilir?

ANSWER:
Baglamda cevap bulunamadi.

### NOW ANSWER

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)

    print("\n--- RAG Ready (Auto Strategy) ---")
    print("(Type 'exit' to quit)")
    
    while True:
        try:
            query = input("\nKullanici: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "cikis"]:
                break

            # Create retriever per query (auto strategy selection)
            retriever = create_retriever(
                vectorstore=vectorstore,
                question=query,
                bm25_retriever=bm25_retriever
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

