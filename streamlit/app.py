"""
Streamlit RAG Chatbot Application

Features:
- Chat interface (LLM backend + Qdrant)
- Sidebar for document management (Incremental Upload/Delete)
- Dynamic vectorstore updates
- System status indicators
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.loader import load_documents, load_single_document
from src.splitter import split_documents
from src.vectorstore import (
    create_embeddings,
    create_vectorstore,
    add_documents_to_collection,
    delete_from_collection,
)
from src.llm import create_llm, create_openai_llm, create_trendyol_llm
from src.prompting import build_prompt, format_docs
from src.retriever import create_retriever, build_bm25_retriever
from src.reranker import create_reranker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Set up directories and config
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

st.set_page_config(page_title="RAG Asistanı", page_icon="🤖", layout="wide")

# --- INITIALIZATION ---
def detect_device():
    forced = os.getenv("RAG_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda"}:
        return forced
    if forced and forced != "auto":
        print(f"Uyari: Gecersiz RAG_DEVICE='{forced}'. 'auto' kullaniliyor.")
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def select_llm_backend(device: str):
    llm_backend = os.getenv("LLM_BACKEND", "trendyol").strip().lower()
    if llm_backend == "openai":
        return create_openai_llm(), "OpenAI (cloud)"
    if llm_backend in {"vllm", "trendyol"} and device != "cuda":
        if os.getenv("OPENAI_API_KEY"):
            return create_openai_llm(), "OpenAI (cloud, no GPU fallback)"
        raise RuntimeError(
            "NVIDIA GPU bulunamadi. vLLM/Trendyol backend icin CUDA gerekir. "
            "LLM_BACKEND=openai yapin veya GPU/driver kurun."
        )
    if llm_backend == "vllm":
        return create_llm(), "vLLM (local)"
    return create_trendyol_llm(), "Trendyol (local)"


def build_bm25_retriever_safe(docs):
    if not docs:
        return None
    try:
        return build_bm25_retriever(docs)
    except Exception as e:
        st.warning(f"BM25 retriever olusturulamadi: {e}")
        return None


@st.cache_resource
def initialize_rag():
    device = detect_device()
    device_label = "CUDA" if device == "cuda" else "CPU"

    embeddings = create_embeddings(device=device)

    documents = load_documents(str(DATA_DIR))
    if documents:
        docs = split_documents(
            documents,
            method="recursive",
            chunk_size=600,
            chunk_overlap=100,
        )
    else:
        docs = []

    try:
        QdrantClient(url=QDRANT_URL).get_collections()
    except Exception as e:
        raise RuntimeError(f"Qdrant baglantisi basarisiz: {e}")

    vectorstore = create_vectorstore(docs, embeddings, url=QDRANT_URL)
    llm, llm_label = select_llm_backend(device)

    # Initialize reranker (optional, can fail gracefully)
    reranker = None
    if device == "cuda":
        try:
            reranker = create_reranker(device="cuda")
        except Exception as e:
            print(f"Reranker yüklenemedi: {e}")

    return vectorstore, llm, embeddings, reranker, docs, llm_label, device_label

try:
    (
        vectorstore,
        llm,
        embeddings,
        reranker,
        initial_docs,
        llm_label,
        device_label,
    ) = initialize_rag()
    st.session_state["rag_ready"] = True
    st.session_state["llm_label"] = llm_label
    st.session_state["device_label"] = device_label

    if "bm25_docs" not in st.session_state:
        st.session_state["bm25_docs"] = list(initial_docs)
    if "bm25_retriever" not in st.session_state:
        st.session_state["bm25_retriever"] = build_bm25_retriever_safe(
            st.session_state["bm25_docs"]
        )
except Exception as e:
    st.error(f"Sistem başlatılamadı: {e}")
    st.stop()

rag_prompt = build_prompt()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Dosya ve Sistem Yönetimi")
    
    # SYSTEM STATUS
    st.subheader("📊 Sistem Durumu")
    if st.session_state.get("rag_ready"):
        st.success("🟢 Vektör DB: Hazır (Qdrant)")
        llm_label = st.session_state.get("llm_label", "Bilinmiyor")
        device_label = st.session_state.get("device_label", "Bilinmiyor")
        st.info(f"🟢 Model: {llm_label} (Aktif)")
        st.info(f"🟢 Cihaz: {device_label}")
    else:
        st.error("🔴 Sistem: Bağlı Değil")

    # RESET CHAT
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # MULTI-QUERY AYARI
    st.subheader("🔍 Arama Ayarları")
    use_multi_query = st.checkbox(
        "Multi-query (Daha İyi Arama)",
        value=st.session_state.get("use_multi_query", False),
        help="Bir soruyu birden fazla şekilde ifade ederek arama yapar. Daha iyi sonuç verir ama daha yavaştır.",
        key="multi_query_checkbox"
    )
    st.session_state["use_multi_query"] = use_multi_query
    
    if use_multi_query:
        num_queries = st.slider(
            "Alternatif Soru Sayısı",
            min_value=2,
            max_value=5,
            value=st.session_state.get("num_queries", 3),
            help="Her soru için kaç alternatif soru üretilecek",
            key="num_queries_slider"
        )
        st.session_state["num_queries"] = num_queries
    else:
        st.session_state["num_queries"] = 3  # Varsayılan (kullanılmayacak)
    
    # RE-RANK AYARI
    use_rerank = st.checkbox(
        "Re-ranking (Daha İyi Sıralama)",
        value=st.session_state.get("use_rerank", False),
        help="Sonuçları cross-encoder ile yeniden sıralar. %15-25 daha iyi accuracy sağlar.",
        key="rerank_checkbox",
        disabled=reranker is None
    )
    st.session_state["use_rerank"] = use_rerank and reranker is not None
    
    if reranker is None:
        st.caption("⚠️ Reranker mevcut değil. 'sentence-transformers' paketini yükleyin.")
    
    if use_rerank and reranker:
        rerank_top_n = st.slider(
            "Rerank için Doküman Sayısı",
            min_value=10,
            max_value=50,
            value=st.session_state.get("rerank_top_n", 20),
            help="Reranking için kaç doküman alınacak (daha fazla = daha iyi ama daha yavaş)",
            key="rerank_top_n_slider"
        )
        st.session_state["rerank_top_n"] = rerank_top_n
    else:
        st.session_state["rerank_top_n"] = 20  # Varsayılan

    st.divider()

    # FILE UPLOAD
    st.subheader("📄 Yeni Doküman Ekle")
    
    # Split method seçimi
    split_method = st.radio(
        "Bölme Yöntemi:",
        ["Recursive (Hızlı)", "Semantic (Daha İyi)"],
        index=0,
        help="Recursive: Hızlı, karakter bazlı bölme\nSemantic: Yavaş ama anlamsal sınırlarda bölme (daha iyi retrieval)"
    )
    use_semantic = split_method == "Semantic (Daha İyi)"
    
    uploaded_files = st.file_uploader(
        "PDF veya TXT yükleyin", 
        accept_multiple_files=True,
        type=['pdf', 'txt']
    )
    
    if uploaded_files:
        if st.button("Yükle ve İşle", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            new_chunks = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"İşleniyor: {uploaded_file.name}...")
                
                # 1. Save File
                file_path = DATA_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Load & Split
                docs = load_single_document(str(file_path))
                
                # Semantic veya Recursive splitter kullan
                if use_semantic:
                    # Semantic splitter kendi icinde dinamik boyut kullanir
                    chunks = split_documents(docs, method="semantic", embeddings=embeddings)
                else:
                    # Ders notlari icin daha kucuk recursive chunk'lar
                    chunks = split_documents(
                        docs,
                        method="recursive",
                        chunk_size=600,
                        chunk_overlap=100,
                    )
                
                # 3. Add to Qdrant (Incremental)
                add_documents_to_collection(vectorstore, chunks)
                new_chunks.extend(chunks)
                
                progress_bar.progress((i + 1) / len(uploaded_files))

            if new_chunks:
                bm25_docs = st.session_state.get("bm25_docs", [])
                bm25_docs.extend(new_chunks)
                st.session_state["bm25_docs"] = bm25_docs
                st.session_state["bm25_retriever"] = build_bm25_retriever_safe(bm25_docs)
            
            status_text.text("✅ Tüm dosyalar eklendi!")
            st.toast(f"{len(uploaded_files)} dosya vektör veritabanına eklendi.")
            st.rerun()

    st.divider()
    
    # FILE LIST & DELETE
    st.subheader("📚 Kayıtlı Dokümanlar")
    files = sorted([p for p in DATA_DIR.iterdir() if p.is_file()])
    if files:
        for file_path in files:
            file_name = file_path.name
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.text(f"📄 {file_name}")
            with col2:
                if st.button("❌", key=f"del_{file_name}"):
                    # 1. Delete from Qdrant
                    delete_from_collection(vectorstore, str(file_path))
                    
                    # 2. Delete file
                    file_path.unlink()

                    bm25_docs = st.session_state.get("bm25_docs", [])
                    if bm25_docs:
                        bm25_docs = [
                            doc
                            for doc in bm25_docs
                            if doc.metadata.get("source") != str(file_path)
                        ]
                        st.session_state["bm25_docs"] = bm25_docs
                        st.session_state["bm25_retriever"] = build_bm25_retriever_safe(
                            bm25_docs
                        )

                    st.toast(f"{file_name} silindi.")
                    st.rerun()
    else:
        st.info("Henüz doküman yok.")

# --- MAIN CHAT ---
st.title("🤖 RAG Asistanı")
st.caption("Qdrant ile güçlendirilmiş doküman asistanı")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Retrieval Chain
def get_response(
    question,
    use_multi_query=False,
    num_queries=3,
    use_rerank=False,
    reranker=None,
    rerank_top_n=20,
    bm25_retriever=None,
):
    retriever = create_retriever(
        vectorstore,
        question=question,
        bm25_retriever=bm25_retriever,
        strategy="auto",
        use_multi_query=use_multi_query,
        llm=llm if use_multi_query else None,
        num_queries=num_queries,
        use_rerank=use_rerank and reranker is not None,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
    )

    chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(format_docs),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return chain.stream(question)

# Input
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Arama ayarlarını session state'den al (sidebar'da ayarlanıyor)
            use_mq = st.session_state.get("use_multi_query", False)
            num_q = st.session_state.get("num_queries", 3)
            use_rr = st.session_state.get("use_rerank", False)
            rerank_top = st.session_state.get("rerank_top_n", 20)
            
            stream = get_response(
                prompt, 
                use_multi_query=use_mq, 
                num_queries=num_q,
                use_rerank=use_rr,
                reranker=reranker,
                rerank_top_n=rerank_top,
                bm25_retriever=st.session_state.get("bm25_retriever"),
            )
            for chunk in stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Hata: {e}")
            full_response = "Üzgünüm, bir hata oluştu."
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
