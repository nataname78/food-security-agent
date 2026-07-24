"""
interactive.py — Interactive CLI: type questions live, get answers live.

Run:
    python src/interactive.py

Type a question and press Enter. Type 'quit' or 'exit' (or Ctrl+C) to stop.
Keeps a single MCP subprocess/session open for the whole conversation, so
after the first question (which loads the embedding/reranking models),
subsequent questions are fast.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import agent  # noqa: E402


def print_result(result: dict) -> None:
    if result.get("blocked"):
        print(f"\n🛑 BLOCKED BY GUARDRAILS")
        print(f"   Reason: {result['reason']}\n")
        return

    print(f"\n{result['answer']}")
    print(f"\n--- Metadata ---")
    print(f"Critic verdict:              {result['critic_verdict']}")
    print(f"Self-consistency agreement:  {result['self_consistency_agreement']:.2f}")
    print(f"Tool calls:                  {result['tool_calls']}")
    print(f"Elapsed time:                {result['elapsed_seconds']:.1f}s\n")


async def main():
    print("Food Security Early Warning Agent — Interactive mode")
    print("Type your question and press Enter. Type 'quit' to stop.\n")
    print("(loading models on first question — may take a moment)\n")

    async with agent.MCPToolClient() as mcp_client:
        while True:
            try:
                question = input("Your question > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Bye.")
                break

            result = await agent.run_agent(question, mcp_client=mcp_client)
            print_result(result)


if __name__ == "__main__":
    asyncio.run(main())