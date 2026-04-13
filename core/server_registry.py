"""
Server Registry and Tool Loader — implements ISL and ITL from ATLAS paper.

Iterative Server Loading (ISL) — Section 2.1:
  Agent starts with a compact index of available MCP servers.
  Selects one server at a time, loads only that server's tools.

Iterative Tool Loading (ITL) — Section 2.2:
  Upon loading a server, agent sees only tool names (lightweight).
  Selectively materializes full schemas only for tools needed at each step.
"""

from servers import ALL_SERVERS
from core.mcp_server import MCPServer


# Compact server index — this is what the agent sees first under ISL
# (Section 2.1: "a compact index of available MCP servers")
SERVER_INDEX: dict[str, str] = {
    name: cls.description for name, cls in ALL_SERVERS.items()
}


def get_all_tool_schemas_text() -> str:
    """
    BASELINE (Naive agent): Load ALL tool schemas from ALL servers upfront.
    This is what ATLAS seeks to avoid — it saturates context.
    """
    parts = []
    for server_name, server_cls in ALL_SERVERS.items():
        server = server_cls()
        parts.append(f"=== Server: {server_name} ===")
        parts.append(f"Description: {server.description}")
        mcp = MCPServer(server_name)
        all_tools = mcp.list_tool_names()
        parts.append(mcp.get_tools_info(all_tools))
        parts.append("")
    return "\n".join(parts)


def get_server_index_text() -> str:
    """ISL step 1: compact server overview for the agent."""
    lines = ["Available MCP Servers:"]
    for name, desc in SERVER_INDEX.items():
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


def get_tool_names_for_server(server_name: str) -> list[str]:
    """ISL step 2 / ITL step 1: return tool names only (no schemas)."""
    mcp = MCPServer(server_name)
    return mcp.list_tool_names()


def get_tool_names_text(server_name: str) -> str:
    """Format tool names for prompt injection."""
    names = get_tool_names_for_server(server_name)
    return f"Tools in '{server_name}': {', '.join(names)}"


def get_tool_schemas_text(server_name: str, tool_names: list[str]) -> str:
    """ITL step 2: materialize full schemas only for selected tools."""
    mcp = MCPServer(server_name)
    return mcp.get_tools_info(tool_names)
