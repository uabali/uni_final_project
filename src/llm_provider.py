from __future__ import annotations

"""
LLM Provider abstraction — Multi-Provider Router.

This layer represents the LLM Provider / Router layer from architecture.pdf.
Supported backends:
  1. VllmProvider    — local vLLM server (Qwen3-8B-AWQ)
  2. OpenAIProvider  — OpenAI direct API
  3. LiteLLMProvider — LiteLLM proxy (100+ provider support)

Fallback chain: Automatically falls back to local vLLM if external API fails.
Runtime provider switching: create_provider_by_name() — no restart required.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .llm import create_llm

logger = logging.getLogger("rag.llm_provider")


# ── Protocol ──────────────────────────────────────────────────

class LLMProvider(Protocol):
    """Minimal LLM interface; kept close to LangChain ChatModel semantics."""

    @property
    def model(self) -> str:  # pragma: no cover - interface
        ...

    @property
    def client(self) -> BaseChatModel:  # pragma: no cover - interface
        ...

    def invoke(self, messages: Iterable[BaseMessage], **kwargs: Any) -> Any:  # pragma: no cover - interface
        ...


# ── 1. VllmProvider (Local) ───────────────────────────────────

@dataclass
class VllmProvider:
    """Wrapper for the existing vLLM-based ChatOpenAI client."""

    _client: BaseChatModel

    @property
    def model(self) -> str:
        return getattr(self._client, "model", "Qwen/Qwen3-8B-AWQ")

    @property
    def client(self) -> BaseChatModel:
        return self._client

    def invoke(self, messages: Iterable[BaseMessage], **kwargs: Any) -> Any:
        return self._client.invoke(list(messages), **kwargs)


# ── 2. OpenAIProvider (Direct) ────────────────────────────────

@dataclass
class OpenAIDirectProvider:
    """
    Provider that connects directly to the OpenAI API.
    Uses ChatOpenAI so bind_tools() support is available.
    """

    _client: BaseChatModel
    _model_name: str = "gpt-4o-mini"

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def client(self) -> BaseChatModel:
        return self._client

    def invoke(self, messages: Iterable[BaseMessage], **kwargs: Any) -> Any:
        return self._client.invoke(list(messages), **kwargs)


def _create_openai_provider(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> OpenAIDirectProvider:
    """Creates an OpenAI direct provider."""
    from langchain_openai import ChatOpenAI

    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OpenAI API key required (OPENAI_API_KEY env or api_key parameter)")

    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    client = ChatOpenAI(
        model=model_name,
        api_key=key,
        base_url=base_url,
        temperature=0.7,
        max_tokens=1024,
    )

    return OpenAIDirectProvider(_client=client, _model_name=model_name)


# ── 3. LiteLLMProvider ────────────────────────────────────────

@dataclass
class LiteLLMProvider:
    """
    Connects to 100+ LLM providers via LiteLLM proxy.
    LiteLLM provides a ChatOpenAI-compatible endpoint.

    Usage:
      - If litellm proxy is running: base_url=http://localhost:4000/v1
      - Direct litellm library: fallback via litellm.completion()
    """

    _client: BaseChatModel
    _model_name: str = "gpt-4o-mini"

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def client(self) -> BaseChatModel:
        return self._client

    def invoke(self, messages: Iterable[BaseMessage], **kwargs: Any) -> Any:
        return self._client.invoke(list(messages), **kwargs)


def _create_litellm_provider(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> LiteLLMProvider:
    """
    Creates a LiteLLM provider.
    Works via LiteLLM proxy or directly through the litellm library.
    """
    from langchain_openai import ChatOpenAI

    # LiteLLM proxy endpoint (defined in docker-compose)
    litellm_url = base_url or os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    key = api_key or os.getenv("LITELLM_API_KEY", os.getenv("OPENAI_API_KEY", "sk-1234"))
    model_name = model or os.getenv("LITELLM_MODEL", "gpt-4o-mini")

    client = ChatOpenAI(
        model=model_name,
        api_key=key,
        base_url=litellm_url,
        temperature=0.7,
        max_tokens=1024,
    )

    return LiteLLMProvider(_client=client, _model_name=model_name)


# ── 4. FallbackProvider ──────────────────────────────────────

@dataclass
class FallbackProvider:
    """
    Fallback chain: falls back to fallback provider if primary fails.

    Typical usage:
      primary  = OpenAI or LiteLLM (external)
      fallback = VllmProvider (local)
    """

    primary: LLMProvider
    fallback: LLMProvider
    _active: str = "primary"

    @property
    def model(self) -> str:
        if self._active == "primary":
            return f"{self.primary.model} (with fallback)"
        return f"{self.fallback.model} (fallback active)"

    @property
    def client(self) -> BaseChatModel:
        """Fallback provider returns the primary's client (for bind_tools compatibility)."""
        if self._active == "fallback":
            return self.fallback.client
        return self.primary.client

    def invoke(self, messages: Iterable[BaseMessage], **kwargs: Any) -> Any:
        try:
            result = self.primary.invoke(messages, **kwargs)
            self._active = "primary"
            return result
        except Exception as exc:
            logger.warning(
                f"Primary provider ({self.primary.model}) failed: {exc}. "
                f"Falling back to ({self.fallback.model})..."
            )
            self._active = "fallback"
            return self.fallback.invoke(messages, **kwargs)


# ── Factory Functions ─────────────────────────────────────────

def create_default_provider() -> VllmProvider:
    """Current default backend: local vLLM server."""
    llm = create_llm()
    return VllmProvider(_client=llm)


def create_provider_by_name(
    name: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    with_fallback: bool = True,
) -> LLMProvider:
    """
    Creates a provider by name.

    Args:
        name: Provider name ("vllm", "openai", "litellm")
        api_key: External provider API key
        model: Model name to use
        base_url: Custom endpoint URL
        with_fallback: If True, creates an External → vLLM fallback chain

    Returns:
        LLMProvider instance

    Raises:
        ValueError: Unknown provider name
    """
    name_lower = name.lower().strip()

    if name_lower == "vllm":
        return create_default_provider()

    if name_lower == "openai":
        primary = _create_openai_provider(api_key=api_key, model=model, base_url=base_url)
        if with_fallback:
            try:
                fallback = create_default_provider()
                return FallbackProvider(primary=primary, fallback=fallback)
            except Exception:
                logger.warning("Could not create local vLLM fallback, using OpenAI only")
                return primary
        return primary

    if name_lower == "litellm":
        primary = _create_litellm_provider(api_key=api_key, model=model, base_url=base_url)
        if with_fallback:
            try:
                fallback = create_default_provider()
                return FallbackProvider(primary=primary, fallback=fallback)
            except Exception:
                logger.warning("Could not create local vLLM fallback, using LiteLLM only")
                return primary
        return primary

    raise ValueError(
        f"Unknown provider: '{name}'. "
        f"Supported providers: vllm, openai, litellm"
    )


def list_available_providers() -> list[dict[str, str]]:
    """Returns a list of available providers."""
    providers = [
        {
            "name": "vllm",
            "description": "Local vLLM server (Qwen3-8B-AWQ)",
            "requires_key": False,
        },
        {
            "name": "openai",
            "description": "OpenAI API (GPT-4o, GPT-4o-mini)",
            "requires_key": True,
        },
        {
            "name": "litellm",
            "description": "LiteLLM proxy (100+ providers: Anthropic, Google, Mistral, etc.)",
            "requires_key": True,
        },
    ]
    return providers
