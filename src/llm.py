"""
LLM Backend Module — Qwen (or other models) via vLLM

Server mode is REQUIRED for agent (tool-calling):
- vLLM must be served as a separate process (scripts/serve_vllm.sh or docker-compose)
- Connects to the OpenAI-compatible API endpoint via ChatOpenAI (supports bind_tools)

Embedded mode (VLLM wrapper) is not supported — BaseLLM does not provide bind_tools.
"""

from langchain_openai import ChatOpenAI

from src.config import load_model_config


def create_llm():
    """
    Creates a ChatOpenAI instance using the shared model config.

    - Model name comes from the VLLM_MODEL env variable (default: Qwen/Qwen3-8B-AWQ).
    - vLLM server URL is provided via VLLM_SERVER_URL (e.g.: http://localhost:6365/v1).

    Returns:
        ChatOpenAI: Chat model instance that supports bind_tools

    Raises:
        ValueError: If VLLM_SERVER_URL is not defined
    """
    cfg = load_model_config()

    if not cfg.server_url:
        raise ValueError(
            "vLLM server mode is required for agent (tool-calling).\n"
            "1. Start the vLLM server with ./scripts/serve_vllm.sh or docker-compose\n"
            "2. Add VLLM_SERVER_URL=http://localhost:6365/v1 to your .env file"
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
