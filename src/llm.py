"""
LLM Modülü - vLLM Backend (LLM Module)

Bu modül, büyük dil modelini (LLM) başlatır ve yönetir.
Production seviyesinde performans için 'vLLM' kütüphanesini kullanır.
"""

from langchain_community.llms import VLLM

def create_llm(
    model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    temperature=0.2
):
    """
    vLLM kullanarak dil modelini yükler.
    
    Özellikler:
    - Model: LLaMA-3.1-8B-Instruct (AWQ Quantized) -> 12GB VRAM için optimize.
    - Hallucination Önlemi: Düşük temperature (0.2) ve top_p ayarları.
    
    Args:
        model (str): HuggingFace model yolu.
        temperature (float): Yaratıcılık seviyesi (Düşük = Daha tutarlı).
        
    Returns:
        VLLM: LangChain uyumlu LLM objesi.
        
    Not: İlk çalıştırmada model ağırlıklarını (~5GB) indirir.
    """

    llm = VLLM(
        model=model,
        trust_remote_code=True,
        max_new_tokens=512,          # Cevap uzunluk limiti (RAG için yeterli)
        temperature=temperature,
        top_p=0.9,
        top_k=5,                     # Sadece en olası 5 kelimeyi değerlendir (Kesinlik artar)
        vllm_kwargs={
            "quantization": "awq",           # AWQ formatı (VRAM tasarrufu)
            "gpu_memory_utilization": 0.55,  # %55 GPU kullanımı (Embeddings'e yer bırakır)
            "max_model_len": 8192,           # Modelin context penceresi
            "enforce_eager": False,
        },
    )

    return llm
