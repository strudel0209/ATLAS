"""
ATLAS Demo Project
==================
Adaptive Tool Loading and Scoped Context — arXiv:2603.06713

A Python implementation demonstrating the ATLAS framework from
Microsoft Research for efficient agentic reasoning with SLMs
in large-scale MCP tool ecosystems.

Compares 4 agent variants:
  1. Naive (All Tools Loaded) — baseline
  2. ISL (Iterative Server Loading)
  3. ITL (Iterative Tool Loading)
  4. ATLAS (ISL + ITL + Programmatic Tool Calling)

Quick start:
  pip install -r requirements.txt
  python demo.py --provider ollama --model qwen3:4b
"""
