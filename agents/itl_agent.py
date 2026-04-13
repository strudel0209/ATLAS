"""
ITL Agent — Iterative Tool Loading (ATLAS Section 2.2, Table 1 Row 14).

From the paper:
  "ATLAS addresses this through Iterative Tool Loading (ITL), which separates
   high-level planning from detailed tool grounding. Upon loading a server,
   the agent initially observes only a compact list of tool names, enabling
   lightweight reasoning about capabilities and plan structure without committing
   context to full schemas. As execution proceeds, the agent selectively
   materializes detailed schemas only for the tools required."

This agent adds ITL on top of ISL:
  1. Starts with compact server index
  2. Loads a server → sees ONLY tool names (not schemas)
  3. Selectively requests full schemas for specific tools
  4. Makes JSON tool calls
"""

import json
import re
from openai import OpenAI

from agents.base_agent import BaseAgent
from core.server_registry import (
    get_server_index_text,
    get_tool_names_text,
    get_tool_schemas_text,
)
from core.mcp_server import MCPServer

ITL_SYSTEM = """You are a strategic agent that solves tasks using LOCAL tool servers.
This system has built-in MCP tool servers.  When you write action blocks below,
the system executes them locally and returns results.
You MUST use these tool protocols — they are fully functional.

{server_index}

Available actions:
1. Load a server's tool NAMES (lightweight):
```fetch_tools
{{"server": "Server Name"}}
```

2. Get DETAILED schemas for specific tools:
```get_tool_info
{{"server": "Server Name", "tools": ["tool1", "tool2"]}}
```

3. Call a tool:
```tool_call
{{"server": "Server Name", "tool": "tool_name", "args": {{...}}}}
```

4. Final answer: <answer>...</answer>

Strategy:
1. Pick relevant server(s) from the index
2. Fetch tool names to understand capabilities
3. Request detailed info ONLY for the tools you actually need
4. Execute tool calls with correct parameters
5. Synthesize and answer

Example:
User: What is the average of 10, 20, 30?
Assistant: I need the Math MCP server. Let me see its tools.
```fetch_tools
{{"server": "Math MCP"}}
```
System: Tools in 'Math MCP': add, subtract, multiply, division, sum, mean, median, mode, min, max, floor, ceiling, round
Assistant: I need the mean tool. Let me get its schema.
```get_tool_info
{{"server": "Math MCP", "tools": ["mean"]}}
```
System: Tool details:
mean(numbers: list (required) — Calculate the mean) -> float
Assistant:
```tool_call
{{"server": "Math MCP", "tool": "mean", "args": {{"numbers": [10, 20, 30]}}}}
```
System: Tool result (Math MCP:mean): 20.0
Assistant: <answer>The average is 20.0.</answer>"""

# Few-shot priming: a concrete fetch_tools → get_tool_info → tool_call exchange.
_ITL_FEWSHOT = [
    {"role": "user", "content": "What is the average of 10 and 20?"},
    {"role": "assistant", "content": 'I need math tools.\n```fetch_tools\n{"server": "Math MCP"}\n```'},
    {"role": "user", "content": "Tools in 'Math MCP': add, subtract, multiply, division, sum, mean, median, mode, min, max, floor, ceiling, round"},
    {"role": "assistant", "content": 'I need the mean tool.\n```get_tool_info\n{"server": "Math MCP", "tools": ["mean"]}\n```'},
    {"role": "user", "content": "Tool details:\nmean(numbers: list (required) — Calculate the mean) -> float"},
    {"role": "assistant", "content": '```tool_call\n{"server": "Math MCP", "tool": "mean", "args": {"numbers": [10, 20]}}\n```'},
    {"role": "user", "content": "Tool result (Math MCP:mean): 15.0"},
    {"role": "assistant", "content": "<answer>The average of 10 and 20 is 15.0.</answer>"},
]


class ITLAgent(BaseAgent):
    """
    Iterative Tool Loading agent.
    Matches ITL rows in ATLAS Table 1 (e.g., Row 14).
    """

    name = "ITL (Iterative Tool Loading)"
    description = "Loads tool names first, then schemas on-demand, JSON calls"

    def solve(self, task: str) -> list[dict]:
        self.reset_metrics()

        server_index = get_server_index_text()
        system_msg = ITL_SYSTEM.format(server_index=server_index)

        messages = [
            {"role": "system", "content": system_msg},
            *_ITL_FEWSHOT,
            {"role": "user", "content": task},
        ]
        self._add_to_trajectory("system", f"[Server index: {len(server_index)} chars]")
        self._add_to_trajectory("user", task)

        loaded_servers: dict[str, MCPServer] = {}
        refusal_count = 0

        for _ in range(self.max_turns):
            reply = self._call_llm(messages)
            self._add_to_trajectory("assistant", reply)

            # Degeneration guard
            if self._is_degenerate(reply):
                messages.append({"role": "assistant", "content": reply[:200]})
                messages.append({"role": "user", "content": "Your output was malformed. Provide your final answer in <answer>...</answer> tags."})
                continue

            # Refusal guard
            if self._is_refusal(reply):
                refusal_count += 1
                messages.append({"role": "assistant", "content": reply})
                if refusal_count >= 2:
                    messages.append({"role": "user", "content": "The tools ARE available. But if you prefer, compute the answer directly and give it in <answer>...</answer> tags."})
                else:
                    messages.append({"role": "user", "content": "The tool protocols ARE functional — this system executes them locally. Please try using ```fetch_tools``` first."})
                continue

            if re.search(r'<answer>(.*?)</answer>', reply, re.DOTALL):
                break

            # ISL: Fetch tool names only
            fetch_match = re.search(r'```fetch_tools\s*\n?(.*?)\n?```', reply, re.DOTALL)
            if fetch_match:
                no_action_count = 0
                try:
                    req = json.loads(fetch_match.group(1))
                    server_name = req["server"]
                    names_text = get_tool_names_text(server_name)
                    loaded_servers[server_name] = MCPServer(server_name)

                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": names_text})
                    self._add_to_trajectory("tools_listed", names_text)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    error_msg = f"Error: {e}"
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": error_msg})
                    self._add_to_trajectory("error", error_msg)
                continue

            # ITL: Get detailed schemas for selected tools
            info_match = re.search(r'```get_tool_info\s*\n?(.*?)\n?```', reply, re.DOTALL)
            if info_match:
                no_action_count = 0
                try:
                    req = json.loads(info_match.group(1))
                    server_name = req["server"]
                    tool_names = req["tools"]
                    schemas = get_tool_schemas_text(server_name, tool_names)

                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": f"Tool details:\n{schemas}"})
                    self._add_to_trajectory("tool_schemas", f"[Loaded {len(tool_names)} schemas]")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    error_msg = f"Error: {e}"
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": error_msg})
                    self._add_to_trajectory("error", error_msg)
                continue

            # Tool call
            tool_match = re.search(r'```tool_call\s*\n?(.*?)\n?```', reply, re.DOTALL)
            if tool_match:
                no_action_count = 0
                try:
                    call = json.loads(tool_match.group(1))
                    server_name = call["server"]
                    tool_name = call["tool"]
                    args = call.get("args", {})

                    mcp = loaded_servers.get(server_name)
                    if mcp is None:
                        result = f"Error: server '{server_name}' not loaded."
                    else:
                        tool_fn = getattr(mcp, tool_name, None)
                        if tool_fn is None:
                            result = f"Error: tool '{tool_name}' not found"
                        else:
                            result = tool_fn(**args)

                    result_str = f"Tool result ({server_name}:{tool_name}): {result}"
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": result_str})
                    self._add_to_trajectory("tool_result", result_str)
                except (json.JSONDecodeError, KeyError) as e:
                    error_msg = f"Error: {e}"
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": error_msg})
                    self._add_to_trajectory("error", error_msg)
            else:
                no_action_count += 1
                messages.append({"role": "assistant", "content": reply})
                if no_action_count >= 3:
                    messages.append({"role": "user", "content": "Please provide your final answer now in <answer>...</answer> tags."})
                else:
                    messages.append({"role": "user", "content": "Use ```fetch_tools``` to load a server, ```get_tool_info``` for schemas, ```tool_call``` to call a tool, or provide <answer>...</answer>."})

        return self.trajectory
