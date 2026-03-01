"""
LLM Backend Modülü — vLLM uzerinden Qwen (veya baska modeller)

Agent (tool-calling) için server mode ZORUNLUDUR:
- vLLM ayrı bir process olarak serve edilmeli (scripts/serve_vllm.sh veya docker-compose)
- ChatOpenAI ile OpenAI-compatible API endpoint'e bağlanır (bind_tools destekler)

Embedded mode (VLLM wrapper) desteklenmez — BaseLLM bind_tools sağlamaz.
"""

from langchain_openai import ChatOpenAI

from src.config import load_model_config


def create_llm():
    """
    Ortak model config'ini kullanarak ChatOpenAI instance'i olusturur.

    - Model adi VLLM_MODEL env degiskeninden gelir (varsayilan: Qwen/Qwen3-8B-AWQ).
    - vLLM server URL'si VLLM_SERVER_URL ile verilir (ornegin: http://localhost:6365/v1).

    Returns:
        ChatOpenAI: bind_tools destekleyen chat model instance

    Raises:
        ValueError: VLLM_SERVER_URL tanımlı değilse
    """
    cfg = load_model_config()

    if not cfg.server_url:
        raise ValueError(
            "Agent (tool-calling) icin vLLM server mode zorunludur.\n"
            "1. ./scripts/serve_vllm.sh veya docker-compose ile vLLM server'i baslatin\n"
            "2. .env dosyasina VLLM_SERVER_URL=http://localhost:6365/v1 ekleyin"
        )

    llm = ChatOpenAI(
        model=cfg.name,
        base_url=cfg.server_url,
        api_key="dummy",
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
        top_p=cfg.top_p,
        frequency_penalty=cfg.frequency_penalty,
        presence_penalty=cfg.presence_penalty,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
        },
    )

    return llm

