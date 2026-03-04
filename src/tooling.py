from __future__ import annotations

"""
Tool invocation abstraction — MCP + Local Hybrid.

This layer represents the MCP/Tool Execution layer from architecture.pdf:

1. LocalToolInvoker  — Existing LangChain tools (search_documents, web_search, calculator)
2. McpToolInvoker    — Tool invocation over HTTP to FastMCP server
3. HybridToolInvoker — Combines Local + MCP; selects appropriate invoker based on tool registry

Security:
  - Timeout control on MCP calls
  - Allowed tool list (whitelist)
  - Error isolation (one tool's error does not affect others)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set

from langchain_core.tools import BaseTool

from .tools import ALL_TOOLS
from .context import get_request_context
from .policy import get_policy_engine
from .audit import get_audit_logger

logger = logging.getLogger("rag.tooling")


# ── Protocol ──────────────────────────────────────────────────

class ToolInvoker(Protocol):
    """Abstract interface for tool invocation."""

    def invoke(self, name: str, args: Dict[str, Any]) -> Any:  # pragma: no cover - interface
        ...

    def list_tools(self) -> List[str]:  # pragma: no cover - interface
        ...


# ── 1. LocalToolInvoker ──────────────────────────────────────

@dataclass
class LocalToolInvoker:
    """
    Invoker that runs existing LangChain tool functions (search_documents, web_search, calculator)
    directly in-process.
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
    Invoker that makes tool calls to a FastMCP server over HTTP.

    Following the MCP (Model Context Protocol) standard:
    - Tool discovery and invocation via SSE transport
    - Timeout control
    - Whitelist-based authorization

    Usage:
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
        """Discovers available tools from the MCP server."""
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
            logger.info(f"MCP tools discovered: {list(tools.keys())}")
            return tools

        except Exception as exc:
            logger.warning(f"MCP tool discovery failed: {exc}")
            return {}

    def invoke(self, name: str, args: Dict[str, Any]) -> Any:
        """Invokes a tool via the MCP server."""
        import urllib.request
        import json

        # Static whitelist authorization check
        if self.allowed_tools and name not in self.allowed_tools:
            raise PermissionError(f"Tool '{name}' is not in the MCP whitelist")

        # Department × MCP matrix check (PolicyEngine)
        ctx = get_request_context()
        policy = get_policy_engine()
        if ctx is not None:
            if not policy.can_call_tool(ctx.department_id, ctx.role, name):
                raise PermissionError(
                    f"Department ({ctx.department_id}) is not authorized for tool '{name}'"
                )

        audit = get_audit_logger()

        try:
            url = f"{self.server_url}/tools/{name}/invoke"
            payload = json.dumps({"arguments": args}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            if audit.is_available:
                audit.log_tool_call(
                    context=ctx,
                    tool_name=name,
                    success=True,
                    extra={"mcp_server_url": self.server_url},
                )

            return result.get("result", result)

        except PermissionError:
            # Policy / whitelist errors are raised to the caller and also audit-logged.
            if audit.is_available:
                audit.log_tool_call(
                    context=ctx,
                    tool_name=name,
                    success=False,
                    error="permission_denied",
                    extra={"mcp_server_url": self.server_url},
                )
            raise
        except TimeoutError:
            if audit.is_available:
                audit.log_tool_call(
                    context=ctx,
                    tool_name=name,
                    success=False,
                    error=f"timeout_{self.timeout_seconds}s",
                    extra={"mcp_server_url": self.server_url},
                )
            raise TimeoutError(f"MCP tool '{name}' timeout ({self.timeout_seconds}s)")
        except Exception as exc:
            logger.error(f"MCP tool invocation error ({name}): {exc}")
            if audit.is_available:
                audit.log_tool_call(
                    context=ctx,
                    tool_name=name,
                    success=False,
                    error=str(exc),
                    extra={"mcp_server_url": self.server_url},
                )
            raise RuntimeError(f"MCP tool '{name}' invocation failed: {exc}") from exc

    def list_tools(self) -> List[str]:
        if not self._discovered_tools:
            self.discover_tools()
        return list(self._discovered_tools.keys())


# ── 3. HybridToolInvoker ─────────────────────────────────────

@dataclass
class HybridToolInvoker:
    """
    Combines Local + MCP.

    - Some tools run locally (calculation, RAG search)
    - Some are routed to external services via MCP (weather, API calls)
    - If a tool is not found, it searches local → MCP in order

    Configuration:
        mcp_only_tools: Set[str] — Tools that only run via MCP
        local_only_tools: Set[str] — Tools that only run locally
    """

    local: LocalToolInvoker
    mcp: Optional[McpToolInvoker] = None
    mcp_only_tools: Set[str] = field(default_factory=set)
    local_only_tools: Set[str] = field(default_factory=lambda: {"search_documents", "calculator"})

    @classmethod
    def from_env(cls) -> "HybridToolInvoker":
        """Creates a hybrid invoker from environment variables."""
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
                logger.warning(f"MCP connection could not be established: {exc}")

        mcp_only = set(
            filter(None, os.getenv("MCP_ONLY_TOOLS", "").split(","))
        )

        return cls(local=local, mcp=mcp, mcp_only_tools=mcp_only)

    def invoke(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Invokes the tool via the appropriate invoker.

        Decision order:
          1. If in mcp_only_tools → MCP
          2. If in local_only_tools → Local
          3. If in neither → Try Local first, then MCP
        """
        # MCP-only tools
        if name in self.mcp_only_tools:
            if self.mcp is None:
                raise RuntimeError(f"Tool '{name}' requires MCP but no MCP connection exists")
            return self.mcp.invoke(name, args)

        # Local-only tools
        if name in self.local_only_tools:
            return self.local.invoke(name, args)

        # Hybrid: try local first, then MCP
        if name in self.local.tools:
            return self.local.invoke(name, args)

        if self.mcp is not None:
            try:
                return self.mcp.invoke(name, args)
            except Exception as exc:
                logger.warning(f"MCP fallback also failed ({name}): {exc}")
                raise

        raise ValueError(f"Unknown tool: {name} (not found in local or MCP)")

    def list_tools(self) -> List[str]:
        """Combined list of all available tools."""
        tools = set(self.local.list_tools())
        if self.mcp is not None:
            tools.update(self.mcp.list_tools())
        return sorted(tools)


# ── Default Instance ──────────────────────────────────────────

DEFAULT_TOOL_INVOKER = LocalToolInvoker.from_default()
