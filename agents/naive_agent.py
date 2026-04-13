"""
Naive Agent — Baseline with all tools eagerly loaded (Row 1 pattern in Table 1).

From ATLAS paper Section 1:
  "In practice, agents are often connected to hundreds of tools across many
   MCP servers, and exposing the full tool registry upfront forces reasoning
   over large, heterogeneous schemas, causing tool definitions and outputs
   to dominate the context window."

This agent loads ALL tool schemas from ALL servers into the system prompt,
then makes JSON-style tool calls one at a time (turn-by-turn).
It represents the worst-case context usage that ATLAS aims to improve.
"""

import json
import re
from openai import OpenAI

from agents.base_agent import BaseAgent
from core.server_registry import get_all_tool_schemas_text
from core.mcp_server import MCPServer
from servers import ALL_SERVERS

# System prompt for the naive agent — all schemas loaded upfront
NAIVE_SYSTEM = """You are a helpful agent that solves tasks using the tool servers below.
This system has LOCAL tool servers built in.  When you write a ```tool_call``` block,
the system executes it and returns results.

Available MCP servers and tools:

{all_schemas}

To call a tool, respond with a JSON block:
```tool_call
{{"server": "Server Name", "tool": "tool_name", "args": {{...}}}}
```

You may call ONE tool per turn.  After getting results, decide the next step.
When you have all the data you need, provide your final answer inside <answer>...</answer> tags."""


# Few-shot priming: a concrete tool exchange injected into conversation
# history so the model sees the protocol working and follows it.
_NAIVE_FEWSHOT = [
    {"role": "user", "content": "What is the sum of 5, 10, and 15?"},
    {"role": "assistant", "content": '```tool_call\n{"server": "Math MCP", "tool": "sum", "args": {"numbers": [5, 10, 15]}}\n```'},
    {"role": "user", "content": "Tool result (Math MCP:sum): 30"},
    {"role": "assistant", "content": "<answer>The sum of 5, 10, and 15 is 30.</answer>"},
]


class NaiveAgent(BaseAgent):
    """
    Eager-load all tools agent.
    Matches the "All Tools Loaded" rows in ATLAS Table 1 (e.g., Row 1).
    """

    name = "Naive (All Tools Loaded)"
    description = "Loads all tool schemas upfront, JSON tool calls per turn"

    def solve(self, task: str) -> list[dict]:
        self.reset_metrics()

        # Load ALL schemas into prompt — the anti-pattern ATLAS addresses
        all_schemas = get_all_tool_schemas_text()
        system_msg = NAIVE_SYSTEM.format(all_schemas=all_schemas)

        messages = [
            {"role": "system", "content": system_msg},
            *_NAIVE_FEWSHOT,
            {"role": "user", "content": task},
        ]
        self._add_to_trajectory("system", f"[Loaded all schemas: {len(all_schemas)} chars]")
        self._add_to_trajectory("user", task)

        # Instantiate all servers for tool execution
        servers = {name: MCPServer(name) for name in ALL_SERVERS}
        no_action_count = 0  # consecutive turns without valid action

        for _ in range(self.max_turns):
            reply = self._call_llm(messages)
            self._add_to_trajectory("assistant", reply)

            # Degeneration guard: truncate repetitive garbage and redirect
            if self._is_degenerate(reply):
                messages.append({"role": "assistant", "content": reply[:200]})
                messages.append({"role": "user", "content": "Your output was malformed. Provide your final answer in <answer>...</answer> tags based on tool results so far."})
                no_action_count += 1
                if no_action_count >= 3:
                    break
                continue

            # Refusal guard: model claims it can't use tools
            if self._is_refusal(reply):
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "The tools ARE available — this system executes them locally. Use ```tool_call``` blocks as described. Or provide your answer in <answer>...</answer> tags."})
                no_action_count += 1
                if no_action_count >= 3:
                    break
                continue

            # Check for final answer
            answer_match = re.search(r'<answer>(.*?)</answer>', reply, re.DOTALL)
            if answer_match:
                break

            # Check for tool call
            tool_match = re.search(r'```tool_call\s*\n?(.*?)\n?```', reply, re.DOTALL)
            if tool_match:
                no_action_count = 0  # valid action resets counter
                try:
                    call = json.loads(tool_match.group(1))
                    server_name = call["server"]
                    tool_name = call["tool"]
                    args = call.get("args", {})

                    mcp = servers.get(server_name)
                    if mcp is None:
                        result = f"Error: server '{server_name}' not found"
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
                    error_msg = f"Error parsing tool call: {e}"
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": error_msg})
                    self._add_to_trajectory("error", error_msg)
            else:
                # No tool call and no answer
                no_action_count += 1
                messages.append({"role": "assistant", "content": reply})
                if no_action_count >= 3:
                    messages.append({"role": "user", "content": "Please provide your final answer now in <answer>...</answer> tags."})
                else:
                    messages.append({"role": "user", "content": "Please use ```tool_call``` blocks or provide your final answer in <answer>...</answer> tags."})

        return self.trajectory
