from __future__ import annotations

"""
Policy engine — department × MCP permission matrix & tool ACL.

Bu modül, architecture dokümanındaki "DEPARTMAN × MCP YETKİ MATRİSİ"ni kodlar.
Amaç:
- Her departman için hangi MCP domain'lerine erişilebileceğini tanımlamak
- Tool ismine bakarak (github_*, jira_*, ldap_*, finance_*) ilgili domain'i
  çıkarmak ve departman yetkisini kontrol etmek

Notlar:
- Gerçek MCP tool isimleri FastMCP server tarafında tanımlanır; burada isim
  üzerinden sezgisel (prefix / substring) eşleştirme yapılır.
- Varsayılan matrix, plan dokümanındaki tabloya göre:

  Engineering:
    - GitHub MCP      → RW
    - Jira/Confluence → RO
    - HR / LDAP       → ⛔
    - Finance DB      → ⛔

  Project Mgmt:
    - GitHub MCP      → RO
    - Jira/Confluence → RW
    - HR / LDAP       → RO
    - Finance DB      → ⛔

  HR:
    - GitHub MCP      → ⛔
    - Jira/Confluence → RO
    - HR / LDAP       → RW
    - Finance DB      → ⛔

  Finance:
    - GitHub MCP      → ⛔
    - Jira/Confluence → ⛔
    - HR / LDAP       → ⛔
    - Finance DB      → RW
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .context import get_default_department_id


_DEFAULT_MATRIX: Dict[str, Dict[str, Optional[str]]] = {
    # dept_key: {domain: permission or None}
    "engineering": {
        "github": "rw",
        "jira": "ro",
        "hr": None,
        "finance": None,
    },
    "project_mgmt": {
        "github": "ro",
        "jira": "rw",
        "hr": "ro",
        "finance": None,
    },
    "hr": {
        "github": None,
        "jira": "ro",
        "hr": "rw",
        "finance": None,
    },
    "finance": {
        "github": None,
        "jira": None,
        "hr": None,
        "finance": "rw",
    },
}


def _normalize_dept(department_id: str | None) -> str:
    if not department_id:
        return get_default_department_id()
    name = department_id.strip().lower()
    # Basit eşleştirme: "engineering", "Engineering", "ENG" gibi varyantlar
    if name in {"eng", "engineering"}:
        return "engineering"
    if name in {"pm", "project", "project_mgmt", "project-management"}:
        return "project_mgmt"
    if name in {"hr", "human_resources", "human-resources"}:
        return "hr"
    if name in {"fin", "finance", "finans"}:
        return "finance"
    # Bilinmeyen departmanlar için default namespace
    return name


def _infer_mcp_domain(tool_name: str) -> Optional[str]:
    """
    Tool ismine bakarak hangi MCP domain'ine ait olabileceğini tahmin eder.

    Örnek:
    - github_list_repos → "github"
    - jira_create_issue → "jira"
    - confluence_search → "jira"
    - ldap_get_employee → "hr"
    - hr_get_employee   → "hr"
    - finance_query_transactions → "finance"
    """
    name = tool_name.lower()

    if name.startswith("github_") or "github" in name:
        return "github"
    if name.startswith("jira_") or "jira" in name:
        return "jira"
    if name.startswith("confluence_") or "confluence" in name:
        return "jira"
    if name.startswith("ldap_") or "ldap" in name or name.startswith("hr_"):
        return "hr"
    if "finance" in name or "financedb" in name or name.startswith("finance_"):
        return "finance"

    return None


@dataclass
class PolicyEngine:
    """
    Basit department × MCP domain policy engine.

    Şu an için sadece:
      - domain düzeyinde izin (var / yok)
      - permission string'i ("ro" / "rw") metadata amaçlı; yazma/okuma ayrımı
        MCP tarafındaki tool tasarımına bırakılmıştır.
    """

    matrix: Dict[str, Dict[str, Optional[str]]]

    def can_use_domain(self, department_id: str, domain: str) -> bool:
        dept_key = _normalize_dept(department_id)
        dept_row = self.matrix.get(dept_key)
        if not dept_row:
            # Bilinmeyen departmanlar için engellemek yerine izin vermek daha az sürprizli;
            # gerçek kurumsal senaryoda burası sıkılaştırılabilir.
            return True
        perm = dept_row.get(domain)
        return bool(perm)

    def can_call_tool(self, department_id: str, role: str, tool_name: str) -> bool:
        domain = _infer_mcp_domain(tool_name)
        if domain is None:
            # Domain ile eşleştirilemeyen MCP tool'ları policy açısından nötr kabul edilir.
            return True
        return self.can_use_domain(department_id, domain)


_POLICY_ENGINE: Optional[PolicyEngine] = None


def get_policy_engine() -> PolicyEngine:
    """Global PolicyEngine instance'ını döner (şimdilik yalnızca default matrix ile)."""
    global _POLICY_ENGINE
    if _POLICY_ENGINE is None:
        _POLICY_ENGINE = PolicyEngine(matrix=_DEFAULT_MATRIX)
    return _POLICY_ENGINE

