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
    return vectorstore, llm, embeddings

try:
    vectorstore, llm, embeddings = initialize_rag()
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

    # FILE UPLOAD
    st.subheader("📄 Yeni Doküman Ekle")
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
                chunks = split_documents(docs)
                
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
def get_response(question):
    retriever = create_retriever(vectorstore, strategy="auto", question=question)
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Sen yardımcı bir asistansın.
Soru: {question}
Bağlam: {context}

Cevap (Türkçe, kısa ve öz):
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
            stream = get_response(prompt)
            for chunk in stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Hata: {e}")
            full_response = "Üzgünüm, bir hata oluştu."
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
