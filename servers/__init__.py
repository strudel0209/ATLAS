"""
Mock MCP Servers simulating real MCP server behavior.
Based on servers used in ATLAS paper (arXiv:2603.06713):
  - Math MCP (from MCPBench - 28 servers used in training)
  - Time MCP (from MCPBench)
  - Weather MCP (simulated for multi-server demonstration)

Each server exposes:
  - name: server identifier
  - tools: dict of tool_name -> {description, params, returns}
  - execute(tool_name, args) -> result
"""

from servers.math_server import MathMCPServer
from servers.time_server import TimeMCPServer
from servers.weather_server import WeatherMCPServer

ALL_SERVERS = {
    "Math MCP": MathMCPServer,
    "Time MCP": TimeMCPServer,
    "Weather MCP": WeatherMCPServer,
}
