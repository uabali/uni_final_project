"""
LLM Backend Modülü — Qwen3-8B-AWQ with vLLM

Agent (tool-calling) için server mode ZORUNLUDUR:
- vLLM ayrı bir process olarak serve edilmeli (scripts/serve_vllm.sh)
- ChatOpenAI ile OpenAI-compatible API endpoint'e bağlanır (bind_tools destekler)

Embedded mode (VLLM wrapper) desteklenmez — BaseLLM bind_tools sağlamaz.
"""

import os

from langchain_openai import ChatOpenAI


def create_llm(
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    top_p: float = 0.95,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.85,
):
    """
    Qwen3-8B-AWQ modelini vLLM server üzerinden ChatOpenAI ile oluşturur.

    VLLM_SERVER_URL env variable ZORUNLUDUR (örn: http://localhost:6365/v1).
    Önce ./scripts/serve_vllm.sh ile vLLM server'ı başlatın.

    Args:
        temperature: Sampling temperature (Qwen3 non-thinking için 0.7 önerilir)
        max_new_tokens: Maksimum output token (varsayilan 256; hiz/kalite dengesi)
        top_p: Nucleus sampling (Qwen3 için 0.95 önerilir)
        frequency_penalty: Token frequency penalty
        presence_penalty: Tekrari azaltir (0.85)

    Returns:
        ChatOpenAI: bind_tools destekleyen chat model instance

    Raises:
        ValueError: VLLM_SERVER_URL tanımlı değilse
    """
    vllm_server_url = os.getenv("VLLM_SERVER_URL", "").strip()

    if not vllm_server_url:
        raise ValueError(
            "Agent (tool-calling) icin vLLM server mode zorunludur.\n"
            "1. ./scripts/serve_vllm.sh ile vLLM server'i baslatin\n"
            "2. .env dosyasina VLLM_SERVER_URL=http://localhost:6365/v1 ekleyin"
        )

    # Env override'lar: üretim ortamında hızlı ayar değişimi için.
    resolved_temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
    resolved_max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(max_new_tokens)))
    resolved_top_p = float(os.getenv("LLM_TOP_P", str(top_p)))
    enable_thinking = os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true"

    llm = ChatOpenAI(
        model="Qwen/Qwen3-8B-AWQ",
        base_url=vllm_server_url,
        api_key="dummy",
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        top_p=resolved_top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )

    return llm
