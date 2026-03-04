from __future__ import annotations

"""
Central application/config layer.

Purpose:
- Read model and sampling settings from a single place
- Keep vLLM server URL and model name consistent between the Python side
  and deployment (docker-compose, local script) side
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Base settings for LLM/vLLM connection."""

    name: str
    server_url: str
    temperature: float
    max_new_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    enable_thinking: bool


def _get_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_model_config() -> ModelConfig:
    """
    Loads the shared model config from ENV.

    Important ENV variables:
    - VLLM_MODEL:    Model name served by vLLM
    - VLLM_SERVER_URL: OpenAI-compatible endpoint (e.g. http://localhost:6365/v1)
    - LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P
    - LLM_ENABLE_THINKING: "true"/"false"
    """

    # Model name shared by both the local script and docker-compose.
    model_name = os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ").strip()

    server_url = os.getenv("VLLM_SERVER_URL", "").strip()

    temperature = _get_env_float("LLM_TEMPERATURE", 0.7)
    max_new_tokens = _get_env_int("LLM_MAX_TOKENS", 512)
    top_p = _get_env_float("LLM_TOP_P", 0.95)

    # Frequency/presence penalties are currently fixed but open to ENV in the future.
    frequency_penalty = _get_env_float("LLM_FREQUENCY_PENALTY", 0.0)
    presence_penalty = _get_env_float("LLM_PRESENCE_PENALTY", 0.85)

    enable_thinking = os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true"

    return ModelConfig(
        name=model_name,
        server_url=server_url,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        enable_thinking=enable_thinking,
    )
