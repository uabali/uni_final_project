from __future__ import annotations

"""
Tool invocation abstraction — MCP + Local Hybrid.

Bu katman, architecture.pdf'teki MCP/Tool Execution layer'i temsil eder:

1. LocalToolInvoker  — Mevcut LangChain tool'ları (search_documents, web_search, calculator)
2. McpToolInvoker    — FastMCP server'a HTTP üzerinden tool çağrısı
3. HybridToolInvoker — Local + MCP birleşimi; tool'un kaydına göre uygun invoker'ı seçer

Güvenlik:
  - MCP çağrılarında timeout kontrolü
  - İzin verilen tool listesi (whitelist)
  - Hata izolasyonu (bir tool'un hatası diğerlerini etkilemez)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set

from langchain_core.tools import BaseTool

from .tools import ALL_TOOLS

logger = logging.getLogger("rag.tooling")


# ── Protocol ──────────────────────────────────────────────────

class ToolInvoker(Protocol):
    """Tool çağırımı için soyut arayüz."""

    def invoke(self, name: str, args: Dict[str, Any]) -> Any:  # pragma: no cover - interface
        ...

    def list_tools(self) -> List[str]:  # pragma: no cover - interface
        ...


# ── 1. LocalToolInvoker ──────────────────────────────────────

@dataclass
class LocalToolInvoker:
    """
    Var olan LangChain tool fonksiyonlarını (search_documents, web_search, calculator)
    dogrudan process içinde çalıştıran invoker.
    """

    tools: Dict[str, BaseTool]

    @classmethod
    def from_default(cls) -> "LocalToolInvoker":
        return cls(tools={tool.name: tool for tool in ALL_TOOLS})

    def invoke(self, name: str, args: Dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown local tool: {name}")
        return tool.invoke(args)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())


# ── 2. McpToolInvoker ────────────────────────────────────────

@dataclass
class McpToolInvoker:
    """
    FastMCP server'a HTTP üzerinden tool çağrısı yapan invoker.

    MCP (Model Context Protocol) standardına uygun olarak:
    - SSE transport ile tool discovery ve invocation
    - Timeout kontrolü
    - Whitelist bazlı yetkilendirme

    Kullanım:
        invoker = McpToolInvoker(
            server_url="http://localhost:8001",
            allowed_tools={"weather", "stock_price"},
        )
        result = invoker.invoke("weather", {"city": "Istanbul"})
    """

    server_url: str = field(
        default_factory=lambda: os.getenv("MCP_SERVER_URL", "http://localhost:8001")
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("MCP_TIMEOUT", "30"))
    )
    allowed_tools: Set[str] = field(default_factory=set)
    _discovered_tools: Dict[str, Any] = field(default_factory=dict, repr=False)

    def discover_tools(self) -> Dict[str, Any]:
        """MCP server'dan kullanılabilir tool'ları keşfeder."""
        import urllib.request
        import json

        try:
            url = f"{self.server_url}/tools"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            tools = {}
            for tool_info in data.get("tools", []):
                name = tool_info.get("name", "")
                if self.allowed_tools and name not in self.allowed_tools:
                    continue
                tools[name] = tool_info

            self._discovered_tools = tools
            logger.info(f"MCP tool'ları keşfedildi: {list(tools.keys())}")
            return tools

        except Exception as exc:
            logger.warning(f"MCP tool discovery başarısız: {exc}")
            return {}

    def invoke(self, name: str, args: Dict[str, Any]) -> Any:
        """MCP server üzerinden tool çağrısı yapar."""
        import urllib.request
        import json

        # Yetkilendirme kontrolü
        if self.allowed_tools and name not in self.allowed_tools:
            raise PermissionError(f"Tool '{name}' MCP whitelist'te değil")

        try:
            url = f"{self.server_url}/tools/{name}/invoke"
            payload = json.dumps({"arguments": args}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            return result.get("result", result)

        except PermissionError:
            raise
        except TimeoutError:
            raise TimeoutError(f"MCP tool '{name}' timeout ({self.timeout_seconds}s)")
        except Exception as exc:
            logger.error(f"MCP tool invocation hatası ({name}): {exc}")
            raise RuntimeError(f"MCP tool '{name}' çağrısı başarısız: {exc}") from exc

    def list_tools(self) -> List[str]:
        if not self._discovered_tools:
            self.discover_tools()
        return list(self._discovered_tools.keys())


# ── 3. HybridToolInvoker ─────────────────────────────────────

@dataclass
class HybridToolInvoker:
    """
    Local + MCP birleşimi.

    - Bazı tool'lar local çalışır (hesaplama, RAG search)
    - Bazıları MCP üzerinden dış servislere yönlendirilir (hava durumu, API çağrıları)
    - Bir tool bulunamazsa sırasıyla local → MCP aranır

    Konfigürasyon:
        mcp_only_tools: Set[str] — Sadece MCP üzerinden çalışacak tool'lar
        local_only_tools: Set[str] — Sadece local çalışacak tool'lar
    """

    local: LocalToolInvoker
    mcp: Optional[McpToolInvoker] = None
    mcp_only_tools: Set[str] = field(default_factory=set)
    local_only_tools: Set[str] = field(default_factory=lambda: {"search_documents", "calculator"})

    @classmethod
    def from_env(cls) -> "HybridToolInvoker":
        """Environment değişkenlerinden hybrid invoker oluşturur."""
        local = LocalToolInvoker.from_default()

        mcp: Optional[McpToolInvoker] = None
        mcp_url = os.getenv("MCP_SERVER_URL", "").strip()
        if mcp_url:
            allowed = set(
                filter(None, os.getenv("MCP_ALLOWED_TOOLS", "").split(","))
            )
            mcp = McpToolInvoker(
                server_url=mcp_url,
                allowed_tools=allowed,
            )
            try:
                mcp.discover_tools()
            except Exception as exc:
                logger.warning(f"MCP bağlantısı kurulamadı: {exc}")

        mcp_only = set(
            filter(None, os.getenv("MCP_ONLY_TOOLS", "").split(","))
        )

        return cls(local=local, mcp=mcp, mcp_only_tools=mcp_only)

    def invoke(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Tool'u uygun invoker üzerinden çağırır.

        Karar sırası:
          1. mcp_only_tools'da ise → MCP
          2. local_only_tools'da ise → Local
          3. İkisinde de yoksa → Local dene, yoksa MCP dene
        """
        # MCP-only tool'lar
        if name in self.mcp_only_tools:
            if self.mcp is None:
                raise RuntimeError(f"Tool '{name}' MCP gerektiriyor ama MCP bağlantısı yok")
            return self.mcp.invoke(name, args)

        # Local-only tool'lar
        if name in self.local_only_tools:
            return self.local.invoke(name, args)

        # Hibrit: önce local, sonra MCP
        if name in self.local.tools:
            return self.local.invoke(name, args)

        if self.mcp is not None:
            try:
                return self.mcp.invoke(name, args)
            except Exception as exc:
                logger.warning(f"MCP fallback da başarısız ({name}): {exc}")
                raise

        raise ValueError(f"Unknown tool: {name} (ne local ne MCP'de bulunamadı)")

    def list_tools(self) -> List[str]:
        """Tüm kullanılabilir tool'ların birleşik listesi."""
        tools = set(self.local.list_tools())
        if self.mcp is not None:
            tools.update(self.mcp.list_tools())
        return sorted(tools)


# ── Default Instance ──────────────────────────────────────────

DEFAULT_TOOL_INVOKER = LocalToolInvoker.from_default()
