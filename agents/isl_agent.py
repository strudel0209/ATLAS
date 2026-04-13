"""
ISL Agent — Iterative Server Loading (ATLAS Section 2.1, Table 1 Rows 4/9).

From the paper:
  "ATLAS treats server selection as an explicit decision. At the start of
   an episode, the agent is given a compact index of available MCP servers
   and a meta-operation to retrieve tool schemas for a selected server."

This agent:
  1. Starts with compact server index only
  2. Selects a server, loads ALL its tool schemas
  3. Makes JSON tool calls against that server
  4. Can load additional servers as needed
"""

import json
import re
from openai import OpenAI

from agents.base_agent import BaseAgent
from core.server_registry import get_server_index_text, get_tool_schemas_text, get_tool_names_for_server
from core.mcp_server import MCPServer

ISL_SYSTEM = """You are a strategic agent that solves tasks using LOCAL tool servers.
This system has built-in MCP tool servers.  When you write a ```fetch_tools``` or
```tool_call``` block, the system executes it locally and returns results.
You MUST use these tool protocols — they are fully functional.

{server_index}

To load a server's tools, respond with:
```fetch_tools
{{"server": "Server Name"}}
```

To call a tool after loading, respond with:
```tool_call
{{"server": "Server Name", "tool": "tool_name", "args": {{...}}}}
```

When done, provide your final answer inside <answer>...</answer> tags.

Strategy:
1. First decide which server(s) are needed based on names and descriptions
2. Load one server at a time to see its tools
3. Execute the required tool calls
4. Synthesize results and provide final answer

Example:
User: What is 5 + 10?
Assistant: I need math tools. Let me load the Math MCP server.
```fetch_tools
{{"server": "Math MCP"}}
```
System: Tools in server='Math MCP':
sum(numbers: list (required) — Sum a list of numbers) -> float
  ...
Assistant: Now I'll call the sum tool.
```tool_call
{{"server": "Math MCP", "tool": "sum", "args": {{"numbers": [5, 10]}}}}
```
System: Tool result (Math MCP:sum): 15
Assistant: <answer>The sum of 5 and 10 is 15.</answer>"""

# Few-shot priming: a concrete fetch_tools + tool_call exchange.
_ISL_FEWSHOT = [
    {"role": "user", "content": "What is 5 + 10?"},
    {"role": "assistant", "content": 'I need math tools. Let me load the Math MCP server.\n```fetch_tools\n{"server": "Math MCP"}\n```'},
    {"role": "user", "content": "Tools in server='Math MCP':\nadd(a: float (required), b: float (required)) -> float\n  Description: Add two numbers\nsum(numbers: list (required)) -> float\n  Description: Sum a list of numbers"},
    {"role": "assistant", "content": '```tool_call\n{"server": "Math MCP", "tool": "add", "args": {"a": 5, "b": 10}}\n```'},
    {"role": "user", "content": "Tool result (Math MCP:add): 15"},
    {"role": "assistant", "content": "<answer>5 + 10 = 15.</answer>"},
]


class ISLAgent(BaseAgent):
    """
    Iterative Server Loading agent.
    Matches ISL rows in ATLAS Table 1 (e.g., Rows 4, 9).
    """

    name = "ISL (Iterative Server Loading)"
    description = "Loads servers on-demand, all tools per server, JSON calls"

    def solve(self, task: str) -> list[dict]:
        self.reset_metrics()

        server_index = get_server_index_text()
        system_msg = ISL_SYSTEM.format(server_index=server_index)

        messages = [
            {"role": "system", "content": system_msg},
            *_ISL_FEWSHOT,
            {"role": "user", "content": task},
        ]
        self._add_to_trajectory("system", f"[Server index: {len(server_index)} chars]")
        self._add_to_trajectory("user", task)

        loaded_servers: dict[str, MCPServer] = {}
        refusal_count = 0
        no_action_count = 0

        for _ in range(self.max_turns):
            reply = self._call_llm(messages)
            self._add_to_trajectory("assistant", reply)

            # Degeneration guard
            if self._is_degenerate(reply):
                messages.append({"role": "assistant", "content": reply[:200]})
                messages.append({"role": "user", "content": "Your output was malformed. Provide your final answer in <answer>...</answer> tags."})
                no_action_count += 1
                if no_action_count >= 3:
                    break
                continue

            # Refusal guard
            if self._is_refusal(reply):
                refusal_count += 1
                no_action_count += 1
                messages.append({"role": "assistant", "content": reply})
                if refusal_count >= 2:
                    messages.append({"role": "user", "content": "The tools ARE available in this system. But if you prefer, compute the answer directly and give it in <answer>...</answer> tags."})
                else:
                    messages.append({"role": "user", "content": "The tool protocols ARE functional — this system executes ```fetch_tools``` and ```tool_call``` blocks locally. Please try using them."})
                if no_action_count >= 3:
                    break
                continue

            # Check for final answer
            if re.search(r'<answer>(.*?)</answer>', reply, re.DOTALL):
                break

            # Check for fetch_tools request (ISL action)
            fetch_match = re.search(r'```fetch_tools\s*\n?(.*?)\n?```', reply, re.DOTALL)
            if fetch_match:
                no_action_count = 0
                try:
                    req = json.loads(fetch_match.group(1))
                    server_name = req["server"]
                    tool_names = get_tool_names_for_server(server_name)
                    schemas = get_tool_schemas_text(server_name, tool_names)
                    loaded_servers[server_name] = MCPServer(server_name)

                    result_str = f"Tools in server='{server_name}':\n{schemas}"
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": result_str})
                    self._add_to_trajectory("server_loaded", result_str)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    error_msg = f"Error loading server: {e}"
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": error_msg})
                    self._add_to_trajectory("error", error_msg)
                continue

            # Check for tool call
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
                        result = f"Error: server '{server_name}' not loaded. Use fetch_tools first."
                    else:
                        tool_fn = getattr(mcp, tool_name, None)
                        if tool_fn is None:
                            result = f"Error: tool '{tool_name}' not found in {server_name}"
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
                    messages.append({"role": "user", "content": "Use ```fetch_tools``` to load a server, ```tool_call``` to call a tool, or provide your answer in <answer>...</answer> tags."})

        return self.trajectory
