"""
ATLAS Agent — Full ISL + ITL + Programmatic Tool Calling (Table 1 Row 19).

This is the complete ATLAS agent combining all three mechanisms:
  - ISL (Section 2.1): Compact server index, load servers on-demand
  - ITL (Section 2.2): See tool names first, load schemas selectively
  - PTC (Section 2.3): Write Python code to orchestrate tools

From the paper (Section 2.3):
  "ATLAS replaces [JSON tool calling] with unified programmatic execution model,
   where all tool interactions are mediated by a persistent Python interpreter.
   Tool calls are expressed as function invocations, control flow is encoded
   explicitly using programming constructs, and intermediate results are stored
   in program state rather than surfaced to the model."

The system prompt matches the PTC prompt from Appendix F.4.3.
"""

import json
import re
from openai import OpenAI

from agents.base_agent import BaseAgent
from core.server_registry import get_server_index_text, get_tool_names_text
from core.code_executor import CodeExecutor
from core.mcp_server import MCPServer

# PTC system prompt — adapted from ATLAS Appendix F.4.3
ATLAS_SYSTEM = """You are a reasoning agent that solves tasks through Python code execution with LOCAL MCP tool servers.
This system has built-in tool servers and a persistent Python interpreter.  When you write
```code``` or ```fetch_tools``` blocks, the system executes them locally and returns results.
You MUST use code execution for calculations — do NOT compute in your head.

{server_index}

MCPServer is pre-imported in the Python interpreter.

MCP Python Coding Rules:
1. Create a brief plan of steps to complete the task.
2. Initialize MCPServer objects: `math_mcp = MCPServer('Math MCP')`
3. ALWAYS first fetch tool schemas: `math_mcp.get_tools_info(['tool1', 'tool2'])`
4. After understanding schemas, write code to implement your plan.
5. If errors occur, call get_tools_info() again to check formats.

To execute Python code, respond with:
```code
<your python code here>
```

To load a server's tool names:
```fetch_tools
{{"server": "Server Name"}}
```

When done, provide your final answer inside <answer>...</answer> tags.

Example:
User: What is the sum and mean of 10, 20, 30?
Assistant: I'll load the Math MCP tools and compute.
```fetch_tools
{{"server": "Math MCP"}}
```
System: Tools in 'Math MCP': add, subtract, multiply, division, sum, mean, median, mode, min, max, floor, ceiling, round
Assistant: I see sum and mean tools. Let me get their schemas and compute.
```code
math_mcp = MCPServer('Math MCP')
print(math_mcp.get_tools_info(['sum', 'mean']))
```
System: stdout:
sum(numbers: list (required) — Sum a list of numbers) -> float
mean(numbers: list (required) — Calculate the mean) -> float
Assistant: Now I'll call both tools.
```code
total = math_mcp.sum(numbers=[10, 20, 30])
avg = math_mcp.mean(numbers=[10, 20, 30])
print(f"Sum: {{total}}, Mean: {{avg}}")
```
System: stdout: Sum: 60, Mean: 20.0
Assistant: <answer>The sum is 60 and the mean is 20.0.</answer>

Key advantage: Store intermediate results in Python variables instead of re-injecting them into the conversation. Use loops, conditionals, and data structures for control flow."""

# Few-shot priming: a concrete fetch_tools → code execution exchange.
_ATLAS_FEWSHOT = [
    {"role": "user", "content": "What is 5 + 10?"},
    {"role": "assistant", "content": '```fetch_tools\n{"server": "Math MCP"}\n```'},
    {"role": "user", "content": "Tools in 'Math MCP': add, subtract, multiply, division, sum, mean, median, mode, min, max, floor, ceiling, round"},
    {"role": "assistant", "content": '```code\nmath_mcp = MCPServer(\'Math MCP\')\nresult = math_mcp.add(a=5, b=10)\nprint(result)\n```'},
    {"role": "user", "content": "stdout:\n15"},
    {"role": "assistant", "content": "<answer>5 + 10 = 15.</answer>"},
]


class ATLASAgent(BaseAgent):
    """
    Full ATLAS agent: ISL + ITL + PTC.
    Matches the best-performing SLM variant in Table 1 (Row 19).
    """

    name = "ATLAS (ISL + ITL + PTC)"
    description = "Iterative loading + programmatic Python orchestration"

    def __init__(self, client: OpenAI, model: str, max_turns: int = 20):
        super().__init__(client, model, max_turns)
        self.executor = CodeExecutor()

    def solve(self, task: str) -> list[dict]:
        self.reset_metrics()
        self.executor.reset()

        server_index = get_server_index_text()
        system_msg = ATLAS_SYSTEM.format(server_index=server_index)

        messages = [
            {"role": "system", "content": system_msg},
            *_ATLAS_FEWSHOT,
            {"role": "user", "content": task},
        ]
        self._add_to_trajectory("system", f"[Server index: {len(server_index)} chars]")
        self._add_to_trajectory("user", task)

        no_action_count = 0

        for _ in range(self.max_turns):
            reply = self._call_llm(messages)
            self._add_to_trajectory("assistant", reply)

            # Degeneration guard: SLMs can fall into repetitive loops
            if self._is_degenerate(reply):
                messages.append({"role": "assistant", "content": reply[:200]})
                messages.append({"role": "user", "content": "Your output was malformed. Provide your final answer in <answer>...</answer> tags based on results so far."})
                no_action_count += 1
                if no_action_count >= 3:
                    break
                continue

            # Refusal guard
            if self._is_refusal(reply):
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "The tool servers and Python interpreter ARE available in this system. Use ```code``` or ```fetch_tools``` blocks. Or provide your answer in <answer>...</answer> tags."})
                no_action_count += 1
                if no_action_count >= 3:
                    break
                continue

            if re.search(r'<answer>(.*?)</answer>', reply, re.DOTALL):
                break

            # Fetch tool names (ISL + ITL step 1)
            fetch_match = re.search(r'```fetch_tools\s*\n?(.*?)\n?```', reply, re.DOTALL)
            if fetch_match:
                no_action_count = 0  # valid action resets counter
                try:
                    req = json.loads(fetch_match.group(1))
                    server_name = req["server"]
                    names_text = get_tool_names_text(server_name)

                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": names_text})
                    self._add_to_trajectory("tools_listed", names_text)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    error_msg = f"Error: {e}"
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content": error_msg})
                    self._add_to_trajectory("error", error_msg)
                continue

            # PTC: Execute Python code
            code_match = re.search(r'```(?:code|python)\s*\n(.*?)\n```', reply, re.DOTALL)
            if code_match:
                no_action_count = 0  # valid action resets counter
                code = code_match.group(1)
                result = self.executor.execute(code)

                if result["success"]:
                    output_parts = []
                    if result["stdout"]:
                        output_parts.append(f"stdout:\n{result['stdout']}")
                    if result["result"] is not None:
                        output_parts.append(f"result:\n{result['result']}")
                    if not output_parts:
                        output_parts.append("Code executed successfully (no output).")
                    output_str = "\n".join(output_parts)

                    # Truncate to prevent context explosion (per paper: max_tool_response_length=4000)
                    if len(output_str) > 4000:
                        output_str = output_str[:4000] + "\n... [truncated]"
                else:
                    output_str = f"Error:\n{result['error']}"

                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": output_str})
                self._add_to_trajectory("code_result", output_str)
            else:
                no_action_count += 1
                messages.append({"role": "assistant", "content": reply})
                if no_action_count >= 3:
                    messages.append({"role": "user", "content": "Please provide your final answer now in <answer>...</answer> tags."})
                else:
                    messages.append({"role": "user", "content": "Write Python in ```code``` blocks, use ```fetch_tools``` to load servers, or provide your answer in <answer>...</answer> tags."})

        return self.trajectory
