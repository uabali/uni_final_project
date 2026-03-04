"""
Query Translation Module

Contains query transformation techniques such as Multi-query, Step-back, HyDE.
This module transforms user queries for more effective retrieval.
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
    Rephrases a question in multiple different ways (Multi-query).
    
    This technique rephrases a question from different angles and with different
    wording to improve retrieval accuracy. A search is performed for each
    alternative question and results are merged.
    
    Args:
        question: Original user question
        llm: LLM model (for query generation)
        num_queries: Number of alternative questions to generate (default: 3)
        
    Returns:
        List[str]: List of original question + alternative questions
        
    Example:
        >>> queries = generate_multi_queries("How to sort a list in Python?", llm)
        >>> # ["How to sort a list in Python?", "Python list sorting methods", 
        >>> #  "Sort list Python tutorial", "Python list ordering techniques"]
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
        
        # Split into lines and clean
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        
        # Remove unnecessary numbers and markers (e.g. "1. ", "- ", etc.)
        cleaned_queries = []
        for q in queries:
            # Remove leading numbers and markers
            q = q.lstrip("0123456789.-) ").strip()
            if q and len(q) > 5:  # Filter out very short ones
                cleaned_queries.append(q)
        
        # Take only the requested amount
        queries = cleaned_queries[:num_queries]
        
        # Add original question at the beginning (important: must be first)
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
    Creates a retriever using multi-query technique.
    
    This function:
    1. Converts the original question into multiple alternative questions
    2. Performs a search for each question
    3. Merges results and removes duplicates
    
    Args:
        vectorstore: Qdrant vectorstore
        question: Original user question
        llm: LLM model (for query generation)
        num_queries: Number of alternative questions to generate
        bm25_retriever: BM25 retriever (for hybrid search)
        strategy: Search strategy ("auto", "mmr", "similarity", "hybrid")
        base_k: Number of chunks to retrieve per query
        **retriever_kwargs: Other retriever parameters
        
    Returns:
        Callable: Multi-query retriever function
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
    
    # 2. Create retriever and search for each query
    all_docs = []
    seen_ids = set()  # To prevent duplicates
    
    for query in queries:
        retriever = create_retriever(
            vectorstore=vectorstore,
            question=query,
            bm25_retriever=bm25_retriever,
            strategy=strategy,
            base_k=base_k,
            **retriever_kwargs
        )
        
        # Perform search
        docs = run_retriever(retriever, query)
        
        # Filter duplicates (same content may come from different queries)
        for doc in docs:
            # Unique ID for doc (content + metadata combination)
            doc_id = hash((doc.page_content[:100], doc.metadata.get("source", "")))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
    
    # 3. Sort results by score (first arrivals have higher scores)
    # Note: LangChain retrievers already return sorted by score
    
    # 4. Take up to base_k (could be base_k * num_queries, but limit with base_k)
    final_docs = all_docs[:base_k * 2]  # Take a bit more, can be trimmed later
    
    print(f"Found {len(all_docs)} unique documents, using {len(final_docs)}.")
    
    # 5. Return retriever-like function
    def multi_query_retriever(query: str):
        """Multi-query retriever - query parameter is ignored (already processed)"""
        return final_docs
    
    return multi_query_retriever
