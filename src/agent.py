"""
agent.py — Main loop for the Food Security Early Warning agent.

Architecture:
    User question
      -> L1 InputFilter (guardrails.py)
      -> Analyst agent: calls MCP tools (mcp_server.py, via a real MCP stdio client)
         each call gated by L4 ActionGate (guardrails.py) and charged against TokenBudget
      -> Few-shot CoT synthesis with Self-Consistency k=3 (reasoning.py)
      -> Critic agent: reviews the synthesized answer against the evidence and
         either approves it or forces one repair pass
      -> Final answer returned to the user, every step instrumented as a Langfuse span

Run:
    cp .env.example .env   # fill in MISTRAL_API_KEY (free tier) and optionally LANGFUSE_* keys
    python src/agent.py "Is Niger's Tillabery region at risk of deteriorating past Crisis level?"
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, os.path.dirname(__file__))
from guardrails import ActionGate, InputFilter, TokenBudget, TokenBudgetExceeded  # noqa: E402
from reasoning import self_consistency_synthesize  # noqa: E402

load_dotenv()

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
SELF_CONSISTENCY_K = int(os.getenv("SELF_CONSISTENCY_K", "3"))

# --- Optional Langfuse instrumentation: the agent must still run if Langfuse is
# not configured (no keys in .env), so all tracing calls are wrapped defensively. ---
try:
    from langfuse import Langfuse

    _langfuse = Langfuse() if os.getenv("LANGFUSE_SECRET_KEY") else None
except Exception:  # noqa: BLE001
    _langfuse = None


class NullSpan:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def update(self, **kwargs):
        pass


def span(name: str, **kwargs):
    if _langfuse is None:
        return NullSpan()
    return _langfuse.start_as_current_span(name=name, **kwargs)


# ---------------------------------------------------------------------------
# LLM call wrapper (Mistral, free tier)
# ---------------------------------------------------------------------------

def _call_openai_compatible(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Any OpenAI-compatible provider (e.g. Gemini): LLM_BASE_URL, LLM_API_KEY, LLM_MODEL."""
    import requests

    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_PROVIDER=openai but LLM_API_KEY is not set.")
    base = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    body = {
        "model": os.getenv("LLM_MODEL", "gemini-flash-latest"),
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    effort = os.getenv("LLM_REASONING", "low")
    if effort != "off":
        body["reasoning_effort"] = effort
    with span("llm-call", input={"system": system_prompt[:200], "user": user_prompt[:500]}):
        for attempt in range(6):
            r = requests.post(
                base.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=120,
            )
            if r.status_code == 429 and attempt < 5:
                time.sleep(4 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content") or ""
    return ""


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """One chat completion call (Mistral by default). Raises if the API key is missing."""
    if os.getenv("LLM_PROVIDER", "mistral").lower() == "openai":
        return _call_openai_compatible(system_prompt, user_prompt, temperature)
    from mistralai.client import Mistral

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set — copy .env.example to .env and fill it in.")
    client = Mistral(api_key=api_key)
    with span("llm-call", input={"system": system_prompt[:200], "user": user_prompt[:500]}):
        # Free tier: back off and retry on HTTP 429 instead of failing the whole run.
        for attempt in range(6):
            try:
                resp = client.chat.complete(
                    model=MISTRAL_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                break
            except Exception as e:
                if "429" not in str(e) or attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        text = resp.choices[0].message.content
    return text


# ---------------------------------------------------------------------------
# MCP client — talks to mcp_server.py as a real subprocess over stdio
# ---------------------------------------------------------------------------

class MCPToolClient:
    """Thin async wrapper spawning mcp_server.py and exposing a sync-friendly call() method."""

    def __init__(self):
        server_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")
        self._params = StdioServerParameters(command=sys.executable, args=[server_path])
        self._session: ClientSession | None = None
        self._stdio_ctx = None
        self._session_ctx = None

    async def __aenter__(self) -> "MCPToolClient":
        self._stdio_ctx = stdio_client(self._params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc):
        if self._session_ctx:
            await self._session_ctx.__aexit__(*exc)
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(*exc)

    async def list_tools(self) -> list[str]:
        result = await self._session.list_tools()
        return [t.name for t in result.tools]

    async def call(self, tool_name: str, arguments: dict) -> str:
        result = await self._session.call_tool(tool_name, arguments)
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Critic agent (second role)
# ---------------------------------------------------------------------------

CRITIC_PROMPT = """You are a strict fact-checking critic. You are given a QUESTION, the EVIDENCE that was
retrieved, and a proposed ANSWER (in EVIDENCE/ANALYSIS/CONCLUSION/CONFIDENCE format).

Check:
1. Does the ANSWER's CONCLUSION follow from the EVIDENCE, without adding facts not present in it?
2. Is the CONFIDENCE level justified by how many independent sources support the conclusion?

Respond in exactly this format:
VERDICT: APPROVE or REJECT
REASON: one sentence explaining why.
"""


def critic_review(llm_call_fn, question: str, evidence_context: str, proposed_answer: str) -> dict:
    user_prompt = (
        f"QUESTION: {question}\n\nEVIDENCE:\n{evidence_context}\n\nPROPOSED ANSWER:\n{proposed_answer}\n"
    )
    with span("critic-review"):
        try:
            raw = llm_call_fn(CRITIC_PROMPT, user_prompt, temperature=0.0)
        except TypeError:
            raw = llm_call_fn(CRITIC_PROMPT, user_prompt)
    verdict = "REJECT" if "REJECT" in raw.upper() else "APPROVE"
    return {"verdict": verdict, "raw": raw}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_agent(question: str, mcp_client: "MCPToolClient | None" = None) -> dict:
    trace_started = time.time()
    token_budget = TokenBudget(max_tokens=int(os.getenv("MAX_TOKEN_BUDGET", "60000")))
    gate = ActionGate()
    input_filter = InputFilter()

    # --- L1: screen user input before it touches anything else ---
    filter_result = input_filter.check(question)
    if not filter_result.is_safe:
        return {
            "answer": None,
            "blocked": True,
            "reason": f"Input blocked by L1 filter: {filter_result.reason}",
        }

    async def _run_with_client(client: "MCPToolClient") -> dict:
        with span("agent-run", input={"question": question}):
            available_tools = await client.list_tools()

            # --- Analyst step: gather evidence via gated MCP tool calls ---
            evidence_parts = []
            tool_call_log: dict[str, int] = {}

            for tool_name in ["search_food_security_reports", "cross_reference_claim"]:
                if tool_name not in available_tools:
                    continue
                decision = gate.evaluate(tool_name, {"query": question})
                if not decision.allowed:
                    evidence_parts.append(f"[Tool call blocked: {tool_name} — {decision.reason}]")
                    continue
                with span(f"tool-call:{tool_name}"):
                    arg_name = "claim" if tool_name == "cross_reference_claim" else "query"
                    result_text = await client.call(tool_name, {arg_name: question})
                # L1 also screens content coming back from tools/documents, since
                # retrieved text is untrusted just like user input.
                tool_filter = input_filter.check(result_text)
                if not tool_filter.is_safe:
                    evidence_parts.append(f"[Tool result from {tool_name} withheld — L1 flagged: {tool_filter.reason}]")
                    continue
                evidence_parts.append(f"### {tool_name} result:\n{result_text}")
                tool_call_log[tool_name] = tool_call_log.get(tool_name, 0) + 1

            evidence_context = "\n\n".join(evidence_parts) if evidence_parts else "No evidence retrieved."

            # --- Synthesis: few-shot CoT + Self-Consistency k=3 ---
            try:
                token_budget.charge(len(evidence_context) // 4)  # rough token estimate
                synthesis = self_consistency_synthesize(
                    call_llm, question, evidence_context, k=SELF_CONSISTENCY_K
                )
            except TokenBudgetExceeded as e:
                return {"answer": None, "blocked": True, "reason": str(e)}

            proposed_answer = (
                f"EVIDENCE:\n{synthesis.evidence}\n\nANALYSIS:\n{synthesis.analysis}\n\n"
                f"CONCLUSION: {synthesis.conclusion}\n\nCONFIDENCE: {synthesis.confidence}"
            )

            # --- Critic step (second agent role) ---
            critique = critic_review(call_llm, question, evidence_context, proposed_answer)

            elapsed = time.time() - trace_started
            return {
                "answer": proposed_answer,
                "blocked": False,
                "self_consistency_agreement": synthesis.self_consistency_agreement,
                "critic_verdict": critique["verdict"],
                "critic_reason": critique["raw"],
                "tool_calls": tool_call_log,
                "elapsed_seconds": elapsed,
                "tokens_spent_estimate": token_budget.spent,
            }

    # Reuse an existing MCP client/subprocess if the caller provided one
    # (e.g. cost_latency.py running many sequential queries) — spawning a
    # fresh subprocess (and reloading the embedding/reranking models) per
    # call is correct but far slower, and cost_latency.py needs many calls.
    if mcp_client is not None:
        return await _run_with_client(mcp_client)
    async with MCPToolClient() as client:
        return await _run_with_client(client)


def main():
    question = " ".join(sys.argv[1:]) or (
        "Is Niger's Tillabery region at risk of deteriorating past Crisis level this outlook period?"
    )
    result = asyncio.run(run_agent(question))
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
