"""
Rubric-based Judge — ATLAS Section 3.1.

From the paper:
  "We replace direct LLM judge with LLM judge under the guidance of task-level
   structured rubrics. For each task, we create a set of rubrics that explicitly
   define success for each coarse-grained criteria."

  "A key advantage of rubric-based rewards is scalability... small language models
   (SLMs) can serve as effective judges when guided by task-level rubrics."

This judge implements the scoring formula from Section 3.1:
  S_R(τ) = Σ(W_i * d_i(τ)) / Σ(W_i)  for each category R ∈ {TF, TA, TG, PA}

The LLM judge (can be an SLM like Qwen3-30B or Phi-4-mini) scores each
criterion 0-1 based on the rubric evaluation prompt from Appendix F.2.
"""

import json
import re
from typing import Optional

from openai import OpenAI

from evaluation.rubrics import (
    RubricCriterion,
    CATEGORY_WEIGHTS,
    CATEGORY_NAMES,
)
from core.token_counter import count_tokens


# Rubric evaluation prompt — adapted from ATLAS Appendix F.2
RUBRIC_EVAL_PROMPT = """You are an expert evaluator for assessing the performance of AI agents on MCP-based tool calling tasks using pre-defined rubrics.

Given a USER_QUERY describing the task assigned to the agent, a RUBRICS list defining the evaluation criteria, and a TRAJECTORY representing the agent's actions and outputs during task execution, your goal is to return a score between 0 and 1 for each rubric based on the degree to which the agent met the expectations.

Scoring guide:
- 0.1-0.3: 10-30% of the criterion is satisfied
- 0.4-0.6: 40-60% of the criterion is satisfied
- 0.7-0.8: 70-80% of the criterion is satisfied
- 0.9-1.0: 90-100% of the criterion is satisfied

USER_QUERY:
{task}

RUBRICS:
{rubrics_text}

TRAJECTORY:
{trajectory_text}

Return ONLY a JSON array of {n_rubrics} numbers between 0 and 1 in the same order as the rubrics.
Example: [0.9, 0.8, 1.0, 0.7, ...]"""


# Generic (non-rubric) judge prompt — adapted from ATLAS Appendix F.3
GENERIC_JUDGE_PROMPT = """You are a STRICT evaluator. Your role is to critically assess AI agent performance with HIGH STANDARDS.

The average score should be around 4-5 out of 10, NOT 7-8. Assign scores ONLY based on evidence.

Score these dimensions (1-10 each):
1. Task Fulfillment: How completely were task requirements met?
2. Tool Appropriateness: Were the right tools selected for each subtask?
3. Tool Grounding: Are agent claims grounded in actual tool outputs?
4. Parameter Accuracy: Were tool parameters correct and complete?

TASK:
{task}

TRAJECTORY:
{trajectory_text}

Return ONLY a JSON object:
{{"task_fulfillment": X, "tool_appropriateness": X, "tool_grounding": X, "parameter_accuracy": X}}"""


class RubricJudge:
    """
    Rubric-based LLM judge per ATLAS Section 3.1.
    Can use any OpenAI-compatible endpoint (Azure AI, Ollama, etc.)
    Paper uses Qwen3-30B-Instruct as SLM judge.
    """

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.judge_tokens_used = 0

    def score_trajectory(
        self,
        task: str,
        trajectory: list[dict],
        rubrics: list[RubricCriterion],
    ) -> dict:
        """
        Score a trajectory against structured rubrics.
        Returns per-category scores and weighted total.
        """
        trajectory_text = self._format_trajectory(trajectory)
        rubrics_text = self._format_rubrics(rubrics)

        prompt = RUBRIC_EVAL_PROMPT.format(
            task=task,
            rubrics_text=rubrics_text,
            trajectory_text=trajectory_text,
            n_rubrics=len(rubrics),
        )

        self.judge_tokens_used += count_tokens(prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        scores = self._parse_scores(raw, len(rubrics))

        # Compute per-category weighted scores (Section 3.1 formula)
        category_scores = {}
        for cat in CATEGORY_WEIGHTS:
            cat_rubrics = [(r, s) for r, s in zip(rubrics, scores) if r.category == cat]
            if cat_rubrics:
                weighted_sum = sum(r.weight * s for r, s in cat_rubrics)
                weight_sum = sum(r.weight for r, _ in cat_rubrics)
                category_scores[cat] = weighted_sum / weight_sum
            else:
                category_scores[cat] = 0.0

        # Final weighted score across categories
        total = sum(
            CATEGORY_WEIGHTS[cat] * category_scores.get(cat, 0)
            for cat in CATEGORY_WEIGHTS
        )

        return {
            "per_criterion": list(zip([r.name for r in rubrics], scores)),
            "category_scores": {
                CATEGORY_NAMES[cat]: round(category_scores[cat], 3)
                for cat in category_scores
            },
            "total_score": round(total, 3),
            "total_score_10": round(total * 10, 2),  # 0-10 scale matching paper
        }

    def score_trajectory_generic(self, task: str, trajectory: list[dict]) -> dict:
        """
        Generic (non-rubric) judge — the baseline that ATLAS rubrics outperform.
        Uses the prompt from Appendix F.3.
        """
        trajectory_text = self._format_trajectory(trajectory)
        prompt = GENERIC_JUDGE_PROMPT.format(
            task=task,
            trajectory_text=trajectory_text,
        )

        self.judge_tokens_used += count_tokens(prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            match = re.search(r'\{[^}]+\}', raw)
            data = json.loads(match.group()) if match else {
                "task_fulfillment": 3,
                "tool_appropriateness": 3,
                "tool_grounding": 3,
                "parameter_accuracy": 3,
            }

        return {
            "category_scores": {
                "Task Fulfillment": data.get("task_fulfillment", 0) / 10,
                "Tool Appropriateness": data.get("tool_appropriateness", 0) / 10,
                "Tool Grounding": data.get("tool_grounding", 0) / 10,
                "Parameter Accuracy": data.get("parameter_accuracy", 0) / 10,
            },
            "total_score_10": round(
                data.get("task_fulfillment", 0) * 0.4
                + data.get("tool_appropriateness", 0) * 0.3
                + data.get("tool_grounding", 0) * 0.2
                + data.get("parameter_accuracy", 0) * 0.1,
                2,
            ),
        }

    @staticmethod
    def _format_trajectory(trajectory: list[dict]) -> str:
        parts = []
        for i, step in enumerate(trajectory):
            role = step.get("role", "unknown")
            content = step.get("content", "")
            parts.append(f"[Step {i+1} - {role}]\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_rubrics(rubrics: list[RubricCriterion]) -> str:
        lines = []
        for i, r in enumerate(rubrics):
            lines.append(
                f"{i+1}. [{r.category}] {r.name} (weight={r.weight}): {r.description}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_scores(raw: str, expected: int) -> list[float]:
        """Parse LLM response into list of float scores."""
        # Try JSON array
        match = re.search(r'\[[\d\s.,]+\]', raw)
        if match:
            try:
                scores = json.loads(match.group())
                if len(scores) == expected:
                    return [max(0.0, min(1.0, float(s))) for s in scores]
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: extract all floats
        floats = re.findall(r'(?<!\d)0?\.\d+|1\.0|0\.0', raw)
        scores = [float(f) for f in floats[:expected]]
        while len(scores) < expected:
            scores.append(0.5)  # neutral default
        return [max(0.0, min(1.0, s)) for s in scores]
