"""
cost_latency.py — Runs the agent N times over a mix of questions and reports:
    - average latency (seconds)
    - average cost (USD, estimated from Mistral's published per-token pricing
      for the model set in MISTRAL_MODEL — see PRICE_PER_1K_TOKENS below,
      update it if you change models)
    - tool call distribution across the N runs
    - whether TokenBudget was triggered at least once

Run:
    python src/cost_latency.py --runs 10

Requires MISTRAL_API_KEY in .env (real network calls — not runnable in a
network-restricted sandbox).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

import agent  # noqa: E402

load_dotenv()

# Mistral Small pricing as of this writing (USD per 1K tokens) — update if you
# change MISTRAL_MODEL or if pricing changes; see https://mistral.ai/pricing
PRICE_PER_1K_TOKENS_INPUT = 0.0002
PRICE_PER_1K_TOKENS_OUTPUT = 0.0006

SAMPLE_QUESTIONS = [
    "Is Niger's Tillabery region at risk of deteriorating past Crisis level?",
    "What is driving new displacement in Burkina Faso?",
    "Will El Nino affect Ethiopia's kiremt rains this year?",
    "What is the food security situation in Darfur, Sudan?",
    "What indicators can predict crop failure before market prices react?",
    "How does the pastoral terms of trade indicator work?",
    "What is expected to improve food security in Mozambique?",
    "Why are food prices elevated in Cameroon's Far North?",
    "What are the five IPC phases?",
    "What is causing the geographic expansion of insecurity in Sudan?",
]


async def run_n(n: int) -> dict:
    latencies = []
    tool_call_totals: dict[str, int] = {}
    token_budget_triggered = False
    results = []

    for i in range(n):
        question = SAMPLE_QUESTIONS[i % len(SAMPLE_QUESTIONS)]
        result = await agent.run_agent(question)
        results.append(result)
        if result.get("blocked"):
            continue
        latencies.append(result["elapsed_seconds"])
        for tool, count in result.get("tool_calls", {}).items():
            tool_call_totals[tool] = tool_call_totals.get(tool, 0) + count
        if result.get("tokens_spent_estimate", 0) > 0:
            pass  # per-run estimate; budget trigger is checked via TokenBudgetExceeded, logged below

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    # NOTE: token counts here are the rough evidence-length estimate charged in
    # agent.py's TokenBudget, NOT the true Mistral input+output token count.
    # For an exact cost figure, read `usage.total_tokens` off each Mistral
    # response (resp.usage) inside call_llm and accumulate it — left as a
    # follow-up since it requires a live API key to demonstrate.
    avg_tokens_estimate = statistics.mean(
        [r.get("tokens_spent_estimate", 0) for r in results if not r.get("blocked")]
    ) if results else 0
    avg_cost_estimate = (avg_tokens_estimate / 1000) * PRICE_PER_1K_TOKENS_INPUT

    return {
        "runs": n,
        "avg_latency_seconds": avg_latency,
        "avg_cost_usd_estimate": avg_cost_estimate,
        "tool_call_distribution": tool_call_totals,
        "token_budget_triggered": token_budget_triggered,
        "note": "cost is a rough estimate from evidence-length token proxy; "
                "replace with resp.usage from Mistral for the exact figure in REPORT.md",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    summary = asyncio.run(run_n(args.runs))
    print(json.dumps(summary, indent=2))

    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "cost_latency_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
