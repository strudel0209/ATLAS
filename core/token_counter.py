"""
Token counting utility for measuring context usage.
Uses tiktoken (OpenAI's tokenizer) for accurate token counts,
matching the token measurement methodology in ATLAS paper Table 1.
"""

import tiktoken

# cl100k_base is used by GPT-4, GPT-4o, and is a fair baseline tokenizer
# for measuring context size across different model families.
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    return len(_ENCODER.encode(text))


def count_messages_tokens(messages: list[dict]) -> int:
    """
    Count tokens in an OpenAI-style messages list.
    Follows the token counting rules for chat messages.
    """
    total = 0
    for msg in messages:
        total += 4  # <im_start>, role, \n, <im_end>
        content = msg.get("content", "")
        if content:
            total += count_tokens(content)
        # Tool call content
        if "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                total += count_tokens(tc.get("function", {}).get("name", ""))
                total += count_tokens(tc.get("function", {}).get("arguments", ""))
    total += 2  # priming tokens
    return total
