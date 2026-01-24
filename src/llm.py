"""
LLM Modülü - vLLM Backend (LLM Module)

Bu modül, büyük dil modelini (LLM) başlatır ve yönetir.
Production seviyesinde performans için 'vLLM' kütüphanesini kullanır.
"""

from langchain_community.llms import VLLM

def create_llm(
    model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    temperature: float = 0.3,
    max_new_tokens: int = 256,
    top_p: float = 0.9,
    top_k: int = 5,
    frequency_penalty: float = 0.8,
    presence_penalty: float = 0.1,
):
    """
    vLLM kullanarak dil modelini yükler.
    
    Özellikler:
    - Model: LLaMA-3.1-8B-Instruct (AWQ INT4 Quantized) -> 12GB VRAM için optimize.
    - VRAM Dağılımı (12GB toplam):
      * LLM model ağırlıkları: ~4.5-5GB
      * KV Cache (max_model_len=2048): ~0.4GB
      * Embedding model (BAAI/bge-m3): ~1.5GB
      * PyTorch/CUDA overhead: ~1GB
      * Buffer: ~1.8GB (güvenlik için)
    - Hallucination Önlemi: Düşük temperature (0.3) ve frequency_penalty (0.8).
    
    Args:
        model (str): HuggingFace model yolu.
        temperature (float): Yaratıcılık seviyesi (Düşük = Daha tutarlı).
        max_new_tokens (int): Maksimum üretilecek token sayısı.
        top_p (float): Nucleus sampling threshold.
        top_k (int): Top-k sampling değeri.
        frequency_penalty (float): Tekrar eden tokenları cezalandırma (0.8 = güçlü).
        presence_penalty (float): Yeni tokenları ödüllendirme.
        
    Returns:
        VLLM: LangChain uyumlu LLM objesi.
        
    Not: İlk çalıştırmada model ağırlıklarını (~4.5GB) indirir.
    """

    llm = VLLM(
        model=model,
        trust_remote_code=True,
        max_new_tokens=max_new_tokens,          # Cevap uzunluk limiti
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,                            # Sadece en olası kelimeleri değerlendir
        frequency_penalty=frequency_penalty,    # Tekrar eden ifadeleri azalt
        presence_penalty=presence_penalty,
        vllm_kwargs={
            "quantization": "awq",              # AWQ formatı (VRAM tasarrufu)
            # 12GB VRAM için optimal: %85 GPU kullanımı (10.2GB), embedding modeli için ~1.8GB buffer
            "gpu_memory_utilization": 0.85,
            # RAG için 2048 context yeterli (soru + cevap + retrieved chunks)
            # 4096'e çıkarırsan KV cache daha fazla yer kaplar, 12GB için riskli olabilir
            "max_model_len": 4096,              # Modelin context penceresi (12GB VRAM için optimize)
            "enforce_eager": False,
        },
    )

    return llm


def create_trendyol_llm(
    model: str = "Trendyol/Trendyol-LLM-8B-T1",
    temperature: float = 0.3,
    max_new_tokens: int = 512,
    top_p: float = 0.9,
    top_k: int = 5,
    frequency_penalty: float = 0.8,
    presence_penalty: float = 0.1,
    use_think_mode: bool = True,
):
    """
    vLLM kullanarak Trendyol LLM-8B-T1 modelini yükler.
    
    Özellikler:
    - Model: Trendyol LLM-8B-T1 (Qwen3-8B tabanlı, Türkçe fine-tuned)
    - VRAM Dağılımı (12GB toplam):
      * LLM model ağırlıkları (FP16): ~8-9GB
      * KV Cache (max_model_len=8192): ~2-3GB
      * Embedding model (BAAI/bge-m3): ~1.5GB
      * PyTorch/CUDA overhead: ~1GB
      * Buffer: ~0.5GB
    - Context Length: 32k token (native), 8k kullanıyoruz (VRAM limiti)
    - Özel Modlar: /think (explicit reasoning) ve /no_think (concise)
    
    Args:
        model (str): HuggingFace model yolu (default: Trendyol/Trendyol-LLM-8B-T1).
        temperature (float): Yaratıcılık seviyesi (Düşük = Daha tutarlı).
        max_new_tokens (int): Maksimum üretilecek token sayısı (default: 512, /think modu için daha uzun).
        top_p (float): Nucleus sampling threshold.
        top_k (int): Top-k sampling değeri.
        frequency_penalty (float): Tekrar eden tokenları cezalandırma (0.8 = güçlü).
        presence_penalty (float): Yeni tokenları ödüllendirme.
        use_think_mode (bool): True ise /think modu (explicit reasoning), False ise /no_think (concise).
        
    Returns:
        VLLM: LangChain uyumlu LLM objesi.
        
    Not: 
    - İlk çalıştırmada model ağırlıklarını (~8-9GB) indirir.
    - Model AWQ formatında değil, quantization parametresi kullanılmaz.
    - 12GB VRAM için optimize edilmiştir (10GB model için).
    """
    
    llm = VLLM(
        model=model,
        trust_remote_code=True,  # Qwen3 tabanlı olduğu için gerekli
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        vllm_kwargs={
            # AWQ quantization YOK - model safetensors/pytorch formatında
            "gpu_memory_utilization": 0.83,  # 10GB model için (12GB'ın %83'ü)
            "max_model_len": 8192,  # 32k destekliyor ama 8k güvenli (KV cache)
            "dtype": "float16",  # Açıkça belirt
            "enforce_eager": False,
        },
    )
    
    # use_think_mode parametresi LLM objesine metadata olarak eklenir
    # Prompt template'lerde kullanılabilir
    llm.use_think_mode = use_think_mode
    
    return llm
