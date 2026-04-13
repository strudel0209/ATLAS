"""
MCPServer Python abstraction — the scaffold described in ATLAS paper Appendix A.

Key design from the paper:
  - Normalizes heterogeneous MCP server JSON schemas into Python-native functions
  - Tool outputs converted to Python-native types via ast.literal_eval (Section A.3)
  - get_tools_info() for Iterative Tool Loading (Section A.5)
  - Informative error messages with closest-match suggestions (Section A.6)

This class bridges abstract agent reasoning and executable tool behavior,
matching the MCPServer class described in Section A.2.
"""

import ast
import difflib
from typing import Any

from servers import ALL_SERVERS


class MCPServer:
    """
    Python-native wrapper around MCP servers.
    Dynamically binds server tools as callable Python methods.

    Usage (matches paper Appendix A.7 PTC examples):
        math_mcp = MCPServer('Math MCP')
        result = math_mcp.sum(numbers=[1, 2, 3])
    """

    def __init__(self, server_name: str):
        server_cls = ALL_SERVERS.get(server_name)
        if server_cls is None:
            available = list(ALL_SERVERS.keys())
            close = difflib.get_close_matches(server_name, available, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise ValueError(
                f"MCP Server '{server_name}' not found. Available: {available}.{hint}"
            )
        self._server = server_cls()
        self._server_name = server_name
        self._tool_names = list(self._server.tools.keys())

        # Dynamically bind each tool as a method (Section A.2)
        for tool_name in self._tool_names:
            setattr(self, tool_name, self._make_caller(tool_name))

    def _make_caller(self, tool_name: str):
        """Create a Python-callable wrapper for a single MCP tool."""
        def caller(**kwargs) -> Any:
            raw = self._server.execute(tool_name, kwargs)
            return self._convert_output(raw)
        # Attach metadata for introspection
        caller.__name__ = tool_name
        caller.__doc__ = self._server.tools[tool_name]["description"]
        return caller

    @staticmethod
    def _convert_output(raw: Any) -> Any:
        """Convert server output to Python-native types (Section A.3)."""
        if isinstance(raw, str):
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return raw
        return raw

    def get_tools_info(self, tool_names: list[str]) -> str:
        """
        Iterative Tool Loading (Section A.5):
        Materialize full Python function signatures + examples only for
        requested tools. This is substantially more token-efficient than
        raw JSON schemas.
        """
        infos = []
        for name in tool_names:
            if name not in self._server.tools:
                close = difflib.get_close_matches(name, self._tool_names, n=1)
                hint = f" Did you mean '{close[0]}'?" if close else ""
                infos.append(f"# ERROR: Tool '{name}' not found.{hint}")
                continue
            tool = self._server.tools[name]
            sig = self._format_signature(name, tool)
            infos.append(sig)
        return "\n\n".join(infos)

    def list_tool_names(self) -> list[str]:
        """Return compact list of tool names (ITL step 1)."""
        return self._tool_names.copy()

    def _format_signature(self, name: str, tool: dict) -> str:
        """Format a tool as a compact Python-native signature with example."""
        params_parts = []
        for pname, pinfo in tool["params"].items():
            req = "(required)" if pinfo.get("required") else f"(optional, default=None)"
            params_parts.append(f"  {pname}: {pinfo['type']} {req} — {pinfo['description']}")
        params_str = "\n".join(params_parts)

        lines = [
            f"{name}(",
            params_str,
            f") -> {tool.get('returns', 'Any')}",
            f"  Description: {tool['description']}",
        ]
        if "example_call" in tool:
            lines.append(f"  Example: {tool['example_call']}")
        if "example_output" in tool:
            lines.append(f"  Output:  {tool['example_output']}")
        return "\n".join(lines)

    def __getattr__(self, name: str):
        """Enhanced error for incorrect function names (Section A.6)."""
        if name.startswith("_"):
            raise AttributeError(name)
        close = difflib.get_close_matches(name, self._tool_names, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise AttributeError(
            f"MCP Server '{self._server_name}' doesn't have the tool '{name}'. "
            f"Available tools: {self._tool_names}.{hint}"
        )

    def __repr__(self) -> str:
        return f"MCPServer('{self._server_name}', tools={self._tool_names})"
