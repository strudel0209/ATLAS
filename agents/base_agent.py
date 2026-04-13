"""
Base agent class — shared infrastructure for all ATLAS agent variants.

All agents use the OpenAI Chat Completions API, which is compatible with:
  - OpenAI models (GPT-4o-mini, etc.)
  - Azure AI Foundry (Phi-4-mini, Phi-4) via OpenAI-compatible endpoint
  - Ollama local models (Qwen3-4B, Phi-4-mini) via OpenAI-compatible API
  - Any OpenAI-compatible inference endpoint

The paper evaluates on Qwen3-4B-Instruct and Qwen2.5-7B-Instruct (Section 4.2).
For 2026 SLMs, we support Phi-4-mini (3.8B) and Qwen3-4B via Ollama or API.
"""

import re
from openai import OpenAI
from core.token_counter import count_tokens

# Phrases that indicate the model is refusing to engage with tool protocols.
# Phi-4-mini's safety RLHF often triggers these when it interprets our custom
# tool protocols (```fetch_tools```, ```tool_call```) as "external API access."
_REFUSAL_PHRASES = [
    "i don't have the capability",
    "i do not have the capability",
    "i can't access external",
    "i cannot access external",
    "i'm unable to",
    "i don't have the ability",
    "as an ai developed by microsoft",
    "as an ai, i don",
    "outside of this text-based interface",
]


class BaseAgent:
    """Base class for all agent variants."""

    name: str = "BaseAgent"
    description: str = ""

    def __init__(self, client: OpenAI, model: str, max_turns: int = 20):
        self.client = client
        self.model = model
        self.max_turns = max_turns
        # Metrics tracked per task (matching paper Table 1 columns)
        self.total_tokens = 0
        self.turn_count = 0
        self.trajectory: list[dict] = []
        self.token_history: list[int] = []  # tokens at each turn (for Figure 1 plot)

    def solve(self, task: str) -> list[dict]:
        """Run the agent on a task. Returns the trajectory."""
        raise NotImplementedError

    def reset_metrics(self):
        """Reset metrics between tasks."""
        self.total_tokens = 0
        self.turn_count = 0
        self.trajectory = []
        self.token_history = []

    def _call_llm(self, messages: list[dict], **kwargs) -> str:
        """Make a single LLM call and track token usage."""
        msg_tokens = count_tokens(
            "\n".join(m.get("content", "") for m in messages)
        )
        self.total_tokens += msg_tokens
        self.token_history.append(self.total_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            **kwargs,
        )
        reply = response.choices[0].message.content or ""
        self.total_tokens += count_tokens(reply)
        self.turn_count += 1
        return reply

    def _add_to_trajectory(self, role: str, content: str):
        """Record a step in the trajectory (for judge evaluation)."""
        self.trajectory.append({"role": role, "content": content})

    # ── SLM failure-mode detectors ────────────────────────────────────

    @staticmethod
    def _is_degenerate(reply: str) -> bool:
        """Detect degenerate repetitive output (a common SLM failure mode).

        When the model falls into a repetition loop (e.g. "200,000,000,000,…")
        each turn generates ~2-4K chars of garbage that accumulates in context,
        causing exponential token growth.  Detecting this early lets us truncate
        and redirect instead of wasting 14 turns of garbage.
        """
        if len(reply) < 200:
            return False
        # Any 8-char substring that repeats 20+ times is degenerate
        for i in range(0, min(len(reply), 100), 8):
            pattern = reply[i:i + 8]
            if pattern.strip() and reply.count(pattern) >= 20:
                return True
        # Mostly commas / digits with no real words
        alphanum = sum(c.isalpha() for c in reply)
        if len(reply) > 500 and alphanum / len(reply) < 0.05:
            return True
        return False

    @staticmethod
    def _is_refusal(reply: str) -> bool:
        """Detect model refusing to engage with tool protocols.

        Phi-4-mini's safety RLHF interprets custom tool-call protocols as
        "external API access" and generates refusal messages like "I don't have
        the capability to directly fetch external data."  Detecting this lets
        us inject a corrective prompt or break the loop.
        """
        lower = reply.lower()
        return any(phrase in lower for phrase in _REFUSAL_PHRASES)
