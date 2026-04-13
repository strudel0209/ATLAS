"""
ATLAS Demo — Side-by-side comparison of all 4 agent variants.

Reproduces the key findings from the ATLAS paper (arXiv:2603.06713):
  - Table 1: Task Fulfillment, Avg Turns, Avg Tokens across variants
  - Figure 1: Context growth across agent designs
  - Section 5.4: Rubric-based vs generic judging comparison

Usage:
  # With Azure AI Foundry (recommended — no local downloads):
  python demo.py --provider azure --model Phi-4-mini-instruct
  python demo.py --provider azure --model Ministral-3B
  python demo.py --provider azure --model Llama-3.2-3B-Instruct
  python demo.py --provider azure --model Phi-4

  # With OpenAI API:
  python demo.py --provider openai --model gpt-4o-mini

  # With Ollama (local, optional):
  python demo.py --provider ollama --model qwen3:4b

Environment variables:
  AZURE_ENDPOINT     — for Azure provider. Supports two formats:
                        Project-scoped (recommended): https://<resource>.services.ai.azure.com/api/projects/<project>
                        Resource-level (legacy):      https://<resource>.services.ai.azure.com
  AZURE_API_KEY      — API key for Azure provider
  AZURE_API_VERSION  — optional API version for Azure resource-level endpoints (default: 2024-10-21)
  OPENAI_API_KEY     — for OpenAI provider

SLM models supported (2026):
  Azure AI Foundry (serverless API):
  - Phi-4-mini-instruct — 3.8B params, Microsoft (closest match to paper's Qwen3-4B)
  - Ministral-3B — 3B params, Mistral AI (edge-optimized SLM)
  - Llama-3.2-3B-Instruct — 3.2B params, Meta (agentic SLM with tool-use training)
  - Phi-4 — 14B params, Microsoft (mid-size comparison slot, arXiv:2412.08905)

  Ollama (local inference, optional):
  - Qwen3-4B (qwen3:4b) — 4B params, primary model from the ATLAS paper
  - Phi-4-mini (phi4-mini:latest) — 3.8B params, Microsoft

  OpenAI API:
  - GPT-4o-mini — comparison baseline
"""

import argparse
import os
import sys
import json
import time
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv(override=True)  # Load .env file (override empty env vars from shell)

import httpx
from openai import AzureOpenAI, OpenAI

from agents.naive_agent import NaiveAgent
from agents.isl_agent import ISLAgent
from agents.itl_agent import ITLAgent
from agents.atlas_agent import ATLASAgent
from evaluation.rubric_judge import RubricJudge
from evaluation.rubrics import (
    build_farm_report_rubrics,
    build_multi_server_rubrics,
    CATEGORY_NAMES,
)

# ──────────────────────────────────────────────────────────────────────
# Tasks — from ATLAS paper Appendix A.7 (farm report) and multi-server
# ──────────────────────────────────────────────────────────────────────

TASK_FARM_REPORT = """I'm pulling together a report on last quarter's harvest from our 10 farms, and honestly I need some hard numbers. We recorded yields of 120, 150, 150, 200, 180, 170, 160, 140, 130, and 155 tons. Here's what I'm trying to nail down:
- What's our total output, average yield per farm, the median and the most common harvest size, plus our lowest and highest yields and the overall spread?
- Then, at $30 a ton, what does that translate to in revenue?
- After covering $2,000 in fixed costs per farm (so 10 farms total), what's left as net profit and what's our profit margin when you express it as a percentage (rounded to the nearest whole number)?
- Finally, I'm curious about the gap between our top-performing farm (200 tons) and the average yield — if that difference is more than 30 tons, I want to budget extra fertilizer at $10 per ton of that gap (and round up); if it's 30 or less, I'll stick with a $500 allowance (and round down).

Could you crunch all those figures? I really need solid data — can't go to my boss with just guesses. Please ensure all findings are supported by concrete data and verifiable sources. I need specific numbers and evidence, not generalizations."""

TASK_MULTI_SERVER = """I need a comprehensive weather briefing for New York City. Please:
1. Get the current weather forecast (temperature, conditions, humidity, wind)
2. Check for any active weather alerts
3. Get the historical monthly average temperatures for the past year
4. Calculate the average of those 12 monthly temperatures using math tools
5. Get the current local time in New York
6. Summarize everything in a single report with concrete numbers.

I need actual data from the tools, not estimates. Please ensure all findings are backed by tool results."""


# Expected outputs for ground-truth validation
EXPECTED_FARM = {
    "total": 1555,
    "average": 155.5,
    "median": 152.5,
    "mode_value": 150,
    "min": 120,
    "max": 200,
    "spread": 80,
    "revenue": 46650,
    "fixed_costs": 20000,
    "net_profit": 26650,
    "profit_margin_pct": 57,
    "gap": 44.5,
    "fertilizer_budget": 450,  # gap > 30, so ceil(44.5 * 10) = 450
}


def _foundry_http_client() -> httpx.Client:
    """httpx client that strips x-stainless-* headers added by the openai SDK.

    Azure AI Foundry project-scoped endpoints reject requests when the total
    header count exceeds ~7.  The openai SDK (v2.x) adds ~8 telemetry headers
    (x-stainless-lang, x-stainless-os, …) on every request, pushing the count
    over that limit and causing an HTTP 431 from the Foundry gateway.

    See: https://github.com/openai/openai-python/issues — "431 Unknown error"
    """
    def _strip(request: httpx.Request):
        for key in [k for k in request.headers if k.startswith("x-stainless")]:
            del request.headers[key]
    return httpx.Client(event_hooks={"request": [_strip]})


def create_client(provider: str) -> OpenAI:
    """Create OpenAI-compatible client for the chosen provider."""
    if provider == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: Set OPENAI_API_KEY environment variable")
            sys.exit(1)
        return OpenAI(api_key=api_key)
    elif provider == "azure":
        endpoint = os.environ.get("AZURE_ENDPOINT")
        api_key = os.environ.get("AZURE_API_KEY")
        if not endpoint or not api_key:
            print("Error: Set AZURE_ENDPOINT and AZURE_API_KEY environment variables")
            sys.exit(1)
        # Project-scoped Foundry endpoint → OpenAI v1 API.
        # Uses the same pattern as AIProjectClient.get_openai_client() from
        # azure-ai-projects (base_url = endpoint + "/openai/v1"), but without
        # pulling in azure-ai-projects/azure-identity as hard dependencies.
        # See: https://learn.microsoft.com/azure/foundry/foundry-models/concepts/endpoints
        if "/api/projects/" in endpoint:
            return OpenAI(
                base_url=endpoint.rstrip("/") + "/openai/v1",
                api_key=api_key,
                http_client=_foundry_http_client(),
            )
        else:
            # Resource-level endpoint: use AzureOpenAI directly
            api_version = os.environ.get("AZURE_API_VERSION", "2024-10-21")
            return AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
    else:
        print(f"Unknown provider: {provider}. Use: ollama, openai, azure")
        sys.exit(1)


def run_agent(agent, task: str) -> dict:
    """Run an agent on a task and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {agent.name}")
    print(f"{'='*60}")

    start = time.time()
    try:
        trajectory = agent.solve(task)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "name": agent.name,
            "turns": 0,
            "tokens": 0,
            "time_s": 0,
            "trajectory": [],
            "token_history": [],
            "error": str(e),
        }
    elapsed = time.time() - start

    print(f"  Turns: {agent.turn_count}")
    print(f"  Tokens: {agent.total_tokens:,}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "name": agent.name,
        "turns": agent.turn_count,
        "tokens": agent.total_tokens,
        "time_s": round(elapsed, 1),
        "trajectory": trajectory,
        "token_history": agent.token_history,
        "error": None,
    }


def print_comparison_table(results: list[dict], scores: dict):
    """Print Table 1-style comparison."""
    print("\n" + "=" * 80)
    print("ATLAS DEMO RESULTS — Agent Comparison (see paper Table 1)")
    print("=" * 80)

    headers = ["Agent Variant", "TF (0-10)", "TA", "TG", "PA", "Turns", "Tokens", "Time(s)"]
    rows = []
    for r in results:
        name = r["name"]
        sc = scores.get(name, {})
        cat = sc.get("category_scores", {})
        rows.append([
            name,
            sc.get("total_score_10", "ERR"),
            f"{cat.get('Tool Appropriateness', 0):.2f}",
            f"{cat.get('Tool Grounding', 0):.2f}",
            f"{cat.get('Parameter Accuracy', 0):.2f}",
            r["turns"],
            f"{r['tokens']:,}",
            r["time_s"],
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Context efficiency comparison
    if len(results) >= 2:
        naive_tokens = results[0]["tokens"] if results[0]["tokens"] > 0 else 1
        print("\nContext Efficiency vs Naive baseline:")
        for r in results[1:]:
            reduction = (1 - r["tokens"] / naive_tokens) * 100
            print(f"  {r['name']}: {reduction:+.1f}% tokens")


def print_token_growth(results: list[dict]):
    """Print ASCII approximation of Figure 1 (context growth over turns)."""
    print("\n" + "=" * 80)
    print("CONTEXT GROWTH (Figure 1 from paper)")
    print("=" * 80)
    successful = [r["tokens"] for r in results if r["tokens"] > 0]
    if not successful:
        print("  (all agents failed — no token data to display)")
        return
    max_tokens = max(successful)
    bar_width = 50

    for r in results:
        name_short = r["name"][:30].ljust(30)
        if r["tokens"] > 0:
            bar_len = int((r["tokens"] / max_tokens) * bar_width)
            bar = "█" * bar_len
            print(f"  {name_short} |{bar}| {r['tokens']:,} tokens")
        else:
            print(f"  {name_short} | ERROR |")


def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Demo: Compare agent architectures on MCP tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --provider azure --model Phi-4-mini-instruct
  python demo.py --provider azure --model Ministral-3B
  python demo.py --provider azure --model Llama-3.2-3B-Instruct
  python demo.py --provider azure --model Phi-4
  python demo.py --provider openai --model gpt-4o-mini
  python demo.py --provider ollama --model qwen3:4b
        """,
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "azure"],
        default="azure",
        help="LLM provider (default: azure)",
    )
    parser.add_argument(
        "--model",
        default="Phi-4-mini-instruct",
        help="Model name / deployment name for the agent (default: Phi-4-mini-instruct)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for the judge (default: same as --model)",
    )
    parser.add_argument(
        "--task",
        choices=["farm", "multi", "both"],
        default="farm",
        help="Task to run (default: farm)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["naive", "isl", "itl", "atlas", "all"],
        default=["all"],
        help="Agent variants to run (default: all)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum turns per agent (default: 15)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip rubric evaluation (just measure tokens/turns)",
    )

    args = parser.parse_args()

    # Print configuration
    print("ATLAS Demo Configuration:")
    print(f"  Provider: {args.provider}")
    print(f"  Agent Model: {args.model}")
    print(f"  Judge Model: {args.judge_model or args.model}")
    print(f"  Task(s): {args.task}")
    print(f"  Max turns: {args.max_turns}")

    client = create_client(args.provider)
    judge_model = args.judge_model or args.model

    # Select agents
    agent_classes = []
    agent_selection = args.agents if "all" not in args.agents else ["naive", "isl", "itl", "atlas"]
    for a in agent_selection:
        if a == "naive":
            agent_classes.append(NaiveAgent)
        elif a == "isl":
            agent_classes.append(ISLAgent)
        elif a == "itl":
            agent_classes.append(ITLAgent)
        elif a == "atlas":
            agent_classes.append(ATLASAgent)

    # Select tasks
    tasks = []
    if args.task in ("farm", "both"):
        tasks.append(("Farm Harvest Report (single-server)", TASK_FARM_REPORT, build_farm_report_rubrics()))
    if args.task in ("multi", "both"):
        tasks.append(("Weather Briefing (multi-server)", TASK_MULTI_SERVER, build_multi_server_rubrics()))

    for task_name, task_text, rubrics in tasks:
        print(f"\n{'#'*80}")
        print(f"# TASK: {task_name}")
        print(f"{'#'*80}")

        # Run all agent variants
        results = []
        for AgentCls in agent_classes:
            agent = AgentCls(client, args.model, max_turns=args.max_turns)
            result = run_agent(agent, task_text)
            results.append(result)

        # Evaluate with rubric judge
        scores = {}
        if not args.skip_judge:
            print("\n" + "-" * 60)
            print("Evaluating with Rubric Judge...")
            print("-" * 60)

            judge = RubricJudge(client, judge_model)

            for r in results:
                if r["error"]:
                    scores[r["name"]] = {
                        "category_scores": {v: 0 for v in CATEGORY_NAMES.values()},
                        "total_score_10": 0,
                    }
                    continue

                print(f"  Judging: {r['name']}...")
                try:
                    sc = judge.score_trajectory(task_text, r["trajectory"], rubrics)
                    scores[r["name"]] = sc
                    print(f"    Score: {sc['total_score_10']}/10")
                except Exception as e:
                    print(f"    Judge error: {e}")
                    scores[r["name"]] = {
                        "category_scores": {v: 0 for v in CATEGORY_NAMES.values()},
                        "total_score_10": 0,
                    }
        else:
            for r in results:
                scores[r["name"]] = {
                    "category_scores": {v: 0 for v in CATEGORY_NAMES.values()},
                    "total_score_10": "N/A",
                }

        # Print results
        print_comparison_table(results, scores)
        print_token_growth(results)

    print("\n" + "=" * 80)
    print("Demo complete. See ATLAS paper (arXiv:2603.06713) for full analysis.")
    print("=" * 80)


if __name__ == "__main__":
    main()
