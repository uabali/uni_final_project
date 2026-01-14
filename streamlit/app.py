"""
Streamlit RAG Chatbot Application

Features:
- Chat interface (vLLM + Qdrant)
- Sidebar for document management (Incremental Upload/Delete)
- Dynamic vectorstore updates
- System status indicators
"""

import streamlit as st
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(".."))

from src.loader import load_single_document
from src.splitter import split_documents
from src.vectorstore import create_embeddings, create_vectorstore, add_documents_to_collection, delete_from_collection
from src.llm import create_llm
from src.retriever import create_retriever
from src.reranker import create_reranker
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set up directories
DATA_DIR = os.path.abspath("../data")
QDRANT_PATH = os.path.abspath("../qdrant_db")
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="RAG Asistanı", page_icon="🤖", layout="wide")

# --- INITIALIZATION ---
@st.cache_resource
def initialize_rag():
    embeddings = create_embeddings()
    # Initialize vectorstore without docs first
    vectorstore = create_vectorstore(docs=[], embeddings=embeddings, path=QDRANT_PATH)
    llm = create_llm()
    
    # Initialize reranker (optional, can fail gracefully)
    reranker = None
    try:
        reranker = create_reranker(device="cuda")
    except Exception as e:
        print(f"Reranker yüklenemedi: {e}")
    
    return vectorstore, llm, embeddings, reranker

try:
    vectorstore, llm, embeddings, reranker = initialize_rag()
    st.session_state["rag_ready"] = True
except Exception as e:
    st.error(f"Sistem başlatılamadı: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Dosya ve Sistem Yönetimi")
    
    # SYSTEM STATUS
    st.subheader("📊 Sistem Durumu")
    if st.session_state.get("rag_ready"):
        st.success("🟢 Vektör DB: Hazır (Qdrant)")
        st.info("🟢 Model: LLaMA vLLM (Aktif)")
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
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"İşleniyor: {uploaded_file.name}...")
                
                # 1. Save File
                file_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Load & Split
                docs = load_single_document(file_path)
                
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
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Tüm dosyalar eklendi!")
            st.toast(f"{len(uploaded_files)} dosya vektör veritabanına eklendi.")
            st.rerun()

    st.divider()
    
    # FILE LIST & DELETE
    st.subheader("📚 Kayıtlı Dokümanlar")
    files = os.listdir(DATA_DIR)
    if files:
        for file in files:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.text(f"📄 {file}")
            with col2:
                if st.button("❌", key=f"del_{file}"):
                    # 1. Delete from Qdrant
                    file_path = os.path.join(DATA_DIR, file)
                    delete_from_collection(vectorstore, file_path)
                    
                    # 2. Delete file
                    os.remove(file_path)
                    st.toast(f"{file} silindi.")
                    st.rerun()
    else:
        st.info("Henüz doküman yok.")

# --- MAIN CHAT ---
st.title("🤖 RAG Asistanı")
st.caption("LLaMA 3.1 & Qdrant ile güçlendirilmiş doküman asistanı")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Retrieval Chain
def get_response(question, use_multi_query=False, num_queries=3, use_rerank=False, reranker=None, rerank_top_n=20):
    retriever = create_retriever(
        vectorstore, 
        strategy="auto", 
        question=question,
        use_multi_query=use_multi_query,
        llm=llm if use_multi_query else None,
        num_queries=num_queries,
        use_rerank=use_rerank and reranker is not None,
        reranker=reranker,
        rerank_top_n=rerank_top_n
    )
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Sen retrieval tabanli bir asistansin.

Görevin:
- Soruyu SADECE verilen bağlama dayanarak cevapla.
- Harici bilgi kullanma, varsayım yapma.
- Eğer cevap bağlamda net olarak yoksa, tam olarak şunu yaz:
  "Baglamda cevap bulunamadi."

Cevap dili ve terim politikası:
- Normal cümleleri TÜRKÇE olarak yaz.
- Kısaltmaları (STT, HTTP, API, OAuth, JWT, JSON, TCP/IP vb.) ve kod/komut/teknik terimleri
  (function adları, class isimleri, değişken isimleri, API endpointleri, CLI komutları, dosya adları)
  soruda veya bağlamda geçtiği ŞEKLİYLE KORU.
- Bu kısaltmaları ve teknik terimleri TÜRKÇE'ye çevirmeye veya değiştirmeye çalışma;
  sadece gerekirse yanına parantez içinde kısaca ne olduğunu açıklayabilirsin.

Cevap stili:
- Önce 1–2 cümlede KISA ve NET bir direkt cevap ver.
- Daha sonra GEREKLIYSE önemli noktaları birkaç maddede kısaca açıkla.
- Soru çok parçalıysa, her parçaya açıkça cevap ver (kısa ama açıklayıcı).
- Aynı cümleyi veya fikri birden fazla kez tekrar etme; bir bilgiyi bir kez net ve kısa şekilde söylemen yeterli.

Soru: {question}
Bağlam: {context}

Cevap:
"""
    )
    
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever
        }
        | prompt
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
                rerank_top_n=rerank_top
            )
            for chunk in stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Hata: {e}")
            full_response = "Üzgünüm, bir hata oluştu."
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
