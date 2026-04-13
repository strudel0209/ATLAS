"""
Math MCP Server - mirrors the Math MCP from MCPBench used in ATLAS paper.
Tools: add, subtract, multiply, division, sum, mean, median, mode, min, max,
       floor, ceiling, round
(See ATLAS paper Appendix A.7 and D.1 for exact tool listing)
"""

import math
import statistics
from typing import Any


class MathMCPServer:
    name = "Math MCP"
    description = "Statistical and arithmetic computations on numbers and lists."

    tools = {
        "add": {
            "description": "Adds two numbers together",
            "params": {
                "firstNumber": {"type": "number", "required": True, "description": "The first addend"},
                "secondNumber": {"type": "number", "required": True, "description": "The second addend"},
            },
            "returns": "number",
            "example_call": "add(firstNumber=3, secondNumber=5)",
            "example_output": "8",
        },
        "subtract": {
            "description": "Subtracts second number from first",
            "params": {
                "firstNumber": {"type": "number", "required": True, "description": "The minuend"},
                "secondNumber": {"type": "number", "required": True, "description": "The subtrahend"},
            },
            "returns": "number",
            "example_call": "subtract(firstNumber=10, secondNumber=3)",
            "example_output": "7",
        },
        "multiply": {
            "description": "Multiplies two numbers",
            "params": {
                "firstNumber": {"type": "number", "required": True, "description": "First factor"},
                "secondNumber": {"type": "number", "required": True, "description": "Second factor"},
            },
            "returns": "number",
            "example_call": "multiply(firstNumber=4, secondNumber=5)",
            "example_output": "20",
        },
        "division": {
            "description": "Divides first number by second",
            "params": {
                "firstNumber": {"type": "number", "required": True, "description": "The dividend"},
                "secondNumber": {"type": "number", "required": True, "description": "The divisor (non-zero)"},
            },
            "returns": "number",
            "example_call": "division(firstNumber=10, secondNumber=3)",
            "example_output": "3.3333333333333335",
        },
        "sum": {
            "description": "Computes the sum of a list of numbers",
            "params": {
                "numbers": {"type": "list[number]", "required": True, "description": "List of numbers to sum"},
            },
            "returns": "number",
            "example_call": "sum(numbers=[1, 2, 3, 4])",
            "example_output": "10",
        },
        "mean": {
            "description": "Computes the arithmetic mean of a list of numbers",
            "params": {
                "numbers": {"type": "list[number]", "required": True, "description": "List of numbers"},
            },
            "returns": "number",
            "example_call": "mean(numbers=[10, 20, 30])",
            "example_output": "20.0",
        },
        "median": {
            "description": "Computes the median of a list of numbers",
            "params": {
                "numbers": {"type": "list[number]", "required": True, "description": "List of numbers"},
            },
            "returns": "number",
            "example_call": "median(numbers=[1, 3, 5, 7])",
            "example_output": "4.0",
        },
        "mode": {
            "description": "Finds the most common value in a list of numbers",
            "params": {
                "numbers": {"type": "list[number]", "required": True, "description": "List of numbers"},
            },
            "returns": "string",
            "example_call": "mode(numbers=[1, 2, 2, 3])",
            "example_output": "'Entries (2) appeared 2 times'",
        },
        "min": {
            "description": "Returns the minimum value from a list of numbers",
            "params": {
                "numbers": {"type": "list[number]", "required": True, "description": "List of numbers"},
            },
            "returns": "number",
            "example_call": "min(numbers=[5, 3, 8, 1])",
            "example_output": "1",
        },
        "max": {
            "description": "Returns the maximum value from a list of numbers",
            "params": {
                "numbers": {"type": "list[number]", "required": True, "description": "List of numbers"},
            },
            "returns": "number",
            "example_call": "max(numbers=[5, 3, 8, 1])",
            "example_output": "8",
        },
        "floor": {
            "description": "Returns the floor of a number (rounds down to nearest integer)",
            "params": {
                "number": {"type": "number", "required": True, "description": "Number to floor"},
            },
            "returns": "number",
            "example_call": "floor(number=3.7)",
            "example_output": "3",
        },
        "ceiling": {
            "description": "Returns the ceiling of a number (rounds up to nearest integer)",
            "params": {
                "number": {"type": "number", "required": True, "description": "Number to ceil"},
            },
            "returns": "number",
            "example_call": "ceiling(number=3.2)",
            "example_output": "4",
        },
        "round": {
            "description": "Rounds a number to specified decimal places",
            "params": {
                "number": {"type": "number", "required": True, "description": "Number to round"},
                "decimals": {"type": "number", "required": False, "description": "Decimal places (default 0)"},
            },
            "returns": "number",
            "example_call": "round(number=3.14159, decimals=2)",
            "example_output": "3.14",
        },
    }

    @staticmethod
    def execute(tool_name: str, args: dict) -> Any:
        dispatch = {
            "add": lambda a: a["firstNumber"] + a["secondNumber"],
            "subtract": lambda a: a["firstNumber"] - a["secondNumber"],
            "multiply": lambda a: a["firstNumber"] * a["secondNumber"],
            "division": lambda a: a["firstNumber"] / a["secondNumber"] if a["secondNumber"] != 0 else "Error: division by zero",
            "sum": lambda a: builtins_sum(a["numbers"]),
            "mean": lambda a: statistics.mean(a["numbers"]),
            "median": lambda a: statistics.median(a["numbers"]),
            "mode": lambda a: _mode_str(a["numbers"]),
            "min": lambda a: builtins_min(a["numbers"]),
            "max": lambda a: builtins_max(a["numbers"]),
            "floor": lambda a: math.floor(a["number"]),
            "ceiling": lambda a: math.ceil(a["number"]),
            "round": lambda a: builtins_round(a["number"], a.get("decimals", 0)),
        }
        if tool_name not in dispatch:
            return f"Error: unknown tool '{tool_name}'. Available: {list(dispatch.keys())}"
        try:
            return dispatch[tool_name](args)
        except (KeyError, TypeError) as e:
            return f"Error calling {tool_name}: {e}"


# Avoid shadowing builtins
builtins_sum = sum
builtins_min = min
builtins_max = max
builtins_round = round


def _mode_str(numbers: list) -> str:
    from collections import Counter
    counts = Counter(numbers)
    most_common_val, most_common_count = counts.most_common(1)[0]
    return f"Entries ({most_common_val}) appeared {most_common_count} times"
