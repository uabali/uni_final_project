from __future__ import annotations

"""
Policy engine — department × MCP permission matrix & tool ACL.

This module encodes the "DEPARTMENT × MCP PERMISSION MATRIX" from the
architecture document.

Purpose:
- Define which MCP domains each department can access
- Check tool names (github_*, jira_*, ldap_*, finance_*) to infer the
  corresponding domain and verify department authorization

Notes:
- Actual MCP tool names are defined on the FastMCP server side; here we use
  heuristic matching (prefix / substring) against tool names.
- Default matrix is based on the plan document table:

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
    # Simple matching: handles variants like "engineering", "Engineering", "ENG"
    if name in {"eng", "engineering"}:
        return "engineering"
    if name in {"pm", "project", "project_mgmt", "project-management"}:
        return "project_mgmt"
    if name in {"hr", "human_resources", "human-resources"}:
        return "hr"
    if name in {"fin", "finance", "finans"}:
        return "finance"
    # Default namespace for unknown departments
    return name


def _infer_mcp_domain(tool_name: str) -> Optional[str]:
    """
    Infers which MCP domain a tool might belong to based on its name.

    Examples:
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
    Simple department × MCP domain policy engine.

    Currently only supports:
      - Domain-level permission (exists / does not exist)
      - Permission string ("ro" / "rw") for metadata purposes; read/write
        distinction is left to MCP-side tool design.
    """

    matrix: Dict[str, Dict[str, Optional[str]]]

    def can_use_domain(self, department_id: str, domain: str) -> bool:
        dept_key = _normalize_dept(department_id)
        dept_row = self.matrix.get(dept_key)
        if not dept_row:
            # For unknown departments, allowing is less surprising than blocking;
            # in a real enterprise scenario, this should be tightened.
            return True
        perm = dept_row.get(domain)
        return bool(perm)

    def can_call_tool(self, department_id: str, role: str, tool_name: str) -> bool:
        domain = _infer_mcp_domain(tool_name)
        if domain is None:
            # MCP tools that cannot be mapped to a domain are considered policy-neutral.
            return True
        return self.can_use_domain(department_id, domain)


_POLICY_ENGINE: Optional[PolicyEngine] = None


def get_policy_engine() -> PolicyEngine:
    """Returns the global PolicyEngine instance (currently with default matrix only)."""
    global _POLICY_ENGINE
    if _POLICY_ENGINE is None:
        _POLICY_ENGINE = PolicyEngine(matrix=_DEFAULT_MATRIX)
    return _POLICY_ENGINE
