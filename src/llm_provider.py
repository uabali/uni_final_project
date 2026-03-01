from __future__ import annotations

"""
LLM Provider abstraction — Multi-Provider Router.

Bu katman, architecture.pdf'teki LLM Provider / Router layer'i temsil eder.
Desteklenen backend'ler:
  1. VllmProvider    — local vLLM server (Qwen3-8B-AWQ)
  2. OpenAIProvider  — OpenAI doğrudan API
  3. LiteLLMProvider — LiteLLM proxy (100+ provider desteği)

Fallback zinciri: External API başarısız olursa otomatik olarak local vLLM'e düşer.
Runtime'da provider değiştirme: create_provider_by_name() ile yeniden başlatma gerektirmez.
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
    """Minimal LLM arayuzu; LangChain ChatModel semantigine yakin tutulur."""

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
    """Mevcut vLLM tabanli ChatOpenAI client icin sarici."""

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
    OpenAI API'sine doğrudan bağlanan provider.
    ChatOpenAI kullandığı için bind_tools() desteği var.
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
    """OpenAI doğrudan provider oluşturur."""
    from langchain_openai import ChatOpenAI

    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OpenAI API key gerekli (OPENAI_API_KEY env veya api_key parametresi)")

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
    LiteLLM proxy üzerinden 100+ LLM provider'a bağlanır.
    LiteLLM ChatOpenAI uyumlu bir endpoint sunar.

    Kullanım:
      - litellm proxy çalışıyorsa: base_url=http://localhost:4000/v1
      - Doğrudan litellm library: litellm.completion() ile fallback
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
    LiteLLM provider oluşturur.
    LiteLLM proxy veya doğrudan litellm library üzerinden çalışır.
    """
    from langchain_openai import ChatOpenAI

    # LiteLLM proxy endpoint'i (docker-compose'da tanımlı)
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
    Fallback zinciri: primary başarısız olursa fallback provider'a düşer.

    Tipik kullanım:
      primary  = OpenAI veya LiteLLM (external)
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
        """Fallback provider, primary'nin client'ını döner (bind_tools uyumluluğu için)."""
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
                f"Primary provider ({self.primary.model}) başarısız: {exc}. "
                f"Fallback'e ({self.fallback.model}) geçiliyor..."
            )
            self._active = "fallback"
            return self.fallback.invoke(messages, **kwargs)


# ── Factory Functions ─────────────────────────────────────────

def create_default_provider() -> VllmProvider:
    """Bugunku varsayılan backend: local vLLM server."""
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
    İsme göre provider oluşturur.

    Args:
        name: Provider adı ("vllm", "openai", "litellm")
        api_key: External provider API key
        model: Kullanılacak model adı
        base_url: Custom endpoint URL
        with_fallback: True ise External → vLLM fallback zinciri oluşturur

    Returns:
        LLMProvider instance

    Raises:
        ValueError: Bilinmeyen provider adı
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
                logger.warning("Local vLLM fallback oluşturulamadı, sadece OpenAI kullanılacak")
                return primary
        return primary

    if name_lower == "litellm":
        primary = _create_litellm_provider(api_key=api_key, model=model, base_url=base_url)
        if with_fallback:
            try:
                fallback = create_default_provider()
                return FallbackProvider(primary=primary, fallback=fallback)
            except Exception:
                logger.warning("Local vLLM fallback oluşturulamadı, sadece LiteLLM kullanılacak")
                return primary
        return primary

    raise ValueError(
        f"Bilinmeyen provider: '{name}'. "
        f"Desteklenen provider'lar: vllm, openai, litellm"
    )


def list_available_providers() -> list[dict[str, str]]:
    """Kullanılabilir provider'ların listesini döner."""
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
            "description": "LiteLLM proxy (100+ provider: Anthropic, Google, Mistral, vb.)",
            "requires_key": True,
        },
    ]
    return providers
