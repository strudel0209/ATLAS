"""
Rubric data structures for ATLAS evaluation — Section 3.1.

From the paper:
  "Each rubric criteria is scored, and then its weighted sums are normalized
   under the category it folds into. There are four categories:
   (i) Task Fulfillment (TF); (ii) Tool Appropriateness (TA);
   (iii) Tool Grounding (TG); (iv) Parameter Accuracy (PA)."

Rubric distribution per paper Appendix F.1:
  - 5 rubrics for Task Fulfillment
  - 3 rubrics for Tool Appropriateness
  - 2 rubrics for Tool Grounding
  - 2 rubrics for Parameter Accuracy
"""

from dataclasses import dataclass


@dataclass
class RubricCriterion:
    """Single rubric criterion as defined in ATLAS Section 3.1."""
    name: str
    category: str  # "TF", "TA", "TG", "PA"
    description: str
    weight: int  # 1-10 per scoring instructions in Appendix F.1


# Category weights for final score aggregation
# (derived from the paper's emphasis: TF most important, then TA, TG, PA)
CATEGORY_WEIGHTS = {
    "TF": 0.40,
    "TA": 0.30,
    "TG": 0.20,
    "PA": 0.10,
}

CATEGORY_NAMES = {
    "TF": "Task Fulfillment",
    "TA": "Tool Appropriateness",
    "TG": "Tool Grounding",
    "PA": "Parameter Accuracy",
}


def build_farm_report_rubrics() -> list[RubricCriterion]:
    """
    Pre-built rubrics for the farm harvest report task
    (mirrors the exact task from ATLAS paper Appendix A.7).
    Distribution: 5 TF + 3 TA + 2 TG + 2 PA = 12 total.
    """
    return [
        # Task Fulfillment (5 criteria)
        RubricCriterion(
            name="Statistical completeness",
            category="TF",
            description="Agent computes ALL required statistics: total output, average yield, median, mode, min, max, and spread (range).",
            weight=10,
        ),
        RubricCriterion(
            name="Revenue calculation",
            category="TF",
            description="Agent correctly computes total revenue at $30 per ton from the total output.",
            weight=9,
        ),
        RubricCriterion(
            name="Profit computation",
            category="TF",
            description="Agent correctly computes net profit (revenue minus $2000 × 10 farms) and profit margin as a rounded percentage.",
            weight=9,
        ),
        RubricCriterion(
            name="Fertilizer budget logic",
            category="TF",
            description="Agent correctly applies conditional logic: if gap > 30 tons → $10/ton of gap (ceiling), else $500 (floor).",
            weight=8,
        ),
        RubricCriterion(
            name="Numerical accuracy",
            category="TF",
            description="All computed numbers match expected values exactly (total=1555, mean=155.5, median=152.5, etc.).",
            weight=10,
        ),
        # Tool Appropriateness (3 criteria)
        RubricCriterion(
            name="Server selection",
            category="TA",
            description="Agent correctly identifies Math MCP as the relevant server for all computations.",
            weight=8,
        ),
        RubricCriterion(
            name="Tool selection efficiency",
            category="TA",
            description="Agent uses appropriate math tools (sum, mean, median, mode, min, max, ceiling/floor) rather than manual calculation.",
            weight=7,
        ),
        RubricCriterion(
            name="No redundant tool calls",
            category="TA",
            description="Agent avoids unnecessary or duplicate tool invocations.",
            weight=6,
        ),
        # Tool Grounding (2 criteria)
        RubricCriterion(
            name="Output grounding",
            category="TG",
            description="All reported numbers directly correspond to tool execution results, not hallucinated or estimated values.",
            weight=8,
        ),
        RubricCriterion(
            name="Intermediate result usage",
            category="TG",
            description="Agent correctly uses intermediate tool outputs (e.g. mean value) in subsequent calculations (e.g. gap computation).",
            weight=7,
        ),
        # Parameter Accuracy (2 criteria)
        RubricCriterion(
            name="Input data accuracy",
            category="PA",
            description="The yield list [120,150,150,200,180,170,160,140,130,155] is passed correctly to all tool calls.",
            weight=9,
        ),
        RubricCriterion(
            name="Parameter format correctness",
            category="PA",
            description="All tool parameters use correct types and names (e.g. 'numbers' for list operations, 'number' for scalar).",
            weight=6,
        ),
    ]


def build_multi_server_rubrics() -> list[RubricCriterion]:
    """
    Rubrics for the multi-server weather + time + math task.
    Tests cross-server orchestration — the harder scenario from ATLAS.
    """
    return [
        # Task Fulfillment (5 criteria)
        RubricCriterion(
            name="Weather data retrieval",
            category="TF",
            description="Agent retrieves current weather forecast for the requested city.",
            weight=10,
        ),
        RubricCriterion(
            name="Alert retrieval",
            category="TF",
            description="Agent checks for and reports any active weather alerts.",
            weight=8,
        ),
        RubricCriterion(
            name="Temperature conversion",
            category="TF",
            description="Agent computes average historical temperature and presents it correctly.",
            weight=9,
        ),
        RubricCriterion(
            name="Time context",
            category="TF",
            description="Agent retrieves current local time for the city's timezone.",
            weight=7,
        ),
        RubricCriterion(
            name="Comprehensive summary",
            category="TF",
            description="Agent provides a coherent summary combining weather, alerts, historical data, and time.",
            weight=8,
        ),
        # Tool Appropriateness (3 criteria)
        RubricCriterion(
            name="Multi-server awareness",
            category="TA",
            description="Agent uses Weather MCP for weather data, Time MCP for time, and Math MCP for calculations.",
            weight=9,
        ),
        RubricCriterion(
            name="Tool selection precision",
            category="TA",
            description="Agent selects get_forecast, get_alerts, get_historical_temp, get_current_time, and mean as needed.",
            weight=7,
        ),
        RubricCriterion(
            name="Execution efficiency",
            category="TA",
            description="Agent minimizes total tool calls and avoids redundant server interactions.",
            weight=6,
        ),
        # Tool Grounding (2 criteria)
        RubricCriterion(
            name="Data grounding",
            category="TG",
            description="All weather data, alerts, and temperatures directly come from tool outputs.",
            weight=8,
        ),
        RubricCriterion(
            name="Cross-server consistency",
            category="TG",
            description="Data from different servers is correctly combined (e.g. timezone matches city).",
            weight=7,
        ),
        # Parameter Accuracy (2 criteria)
        RubricCriterion(
            name="City parameter consistency",
            category="PA",
            description="Agent uses the same city name across all server calls.",
            weight=8,
        ),
        RubricCriterion(
            name="Timezone parameter accuracy",
            category="PA",
            description="Agent maps the city to the correct IANA timezone for the Time MCP call.",
            weight=6,
        ),
    ]
