# Architecture — Food Security Early Warning Agent

## Overview

The agent answers analyst-style questions about acute food insecurity risk
(IPC phases, drivers, early-warning signals) for a given country/region, using
a locally-indexed corpus of food-security reports plus a live climate data
tool, with a security stack, structured reasoning, self-consistency voting,
and a critic pass before returning an answer.

## Diagram

```
                              ┌──────────────────────────┐
                              │        User question      │
                              └─────────────┬─────────────┘
                                            │
                                   L1 InputFilter (guardrails.py)
                                   (injection patterns + NFKC
                                    unicode normalization)
                                            │  blocked? -> return refusal, stop
                                            ▼
                              ┌──────────────────────────┐
                              │       Analyst step         │
                              │  (agent.py: run_agent)      │
                              └─────────────┬─────────────┘
                                            │
                     ┌──────────────────────┼───────────────────────┐
                     ▼                      ▼                       ▼
           MCP tool call:          MCP tool call:           MCP tool call:
      search_food_security_    cross_reference_claim    get_climate_indicator
      reports (hybrid RAG)      (corroboration check)     (NASA POWER, optional)
                     │                      │                       │
              each call gated by L4 ActionGate (risk matrix, call cap)
              each call's TEXT screened again by L1 (untrusted content)
                     │                      │                       │
                     └──────────────────────┼───────────────────────┘
                                            ▼
                              ┌──────────────────────────┐
                              │   Evidence context (str)   │
                              └─────────────┬─────────────┘
                                            ▼
                       Few-shot CoT synthesis, Self-Consistency k=3
                       (reasoning.py: EVIDENCE/ANALYSIS/CONCLUSION/CONFIDENCE,
                        majority vote across 3 samples at temperature>0)
                                            ▼
                              ┌──────────────────────────┐
                              │      Critic agent          │
                              │  (2nd LLM role: APPROVE /   │
                              │   REJECT against evidence)  │
                              └─────────────┬─────────────┘
                                            ▼
                              ┌──────────────────────────┐
                              │      Final answer          │
                              └──────────────────────────┘

  Every LLM call and every tool call above is wrapped in a Langfuse span
  (agent.py: span()), degrading to a no-op if Langfuse keys are absent.
```

## Retrieval pipeline detail (retrieval.py)

```
query
  ├─► BM25 sparse search (top 15)      ─┐
  └─► Dense search, cosine sim (top 15) ─┤
                                         ▼
                          Reciprocal Rank Fusion (RRF, k=60)
                                         ▼
                     Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
                                         ▼
                     Parent-document expansion (child paragraph
                     -> full source document, deduplicated)
                                         ▼
                              top_k context passed to LLM
```

## Components

| File | Role |
|---|---|
| `src/agent.py` | Main orchestration loop; MCP client; Mistral LLM call wrapper; critic step; Langfuse spans |
| `src/mcp_server.py` | Custom MCP server exposing 3 tools over stdio transport |
| `src/retrieval.py` | Hybrid BM25 + dense retrieval, RRF fusion, cross-encoder reranking, parent-child chunking |
| `src/guardrails.py` | L1 InputFilter, L4 ActionGate (risk matrix + call cap), TokenBudget |
| `src/reasoning.py` | Few-shot CoT prompt, structured-answer parsing, Self-Consistency (k=3) majority vote |
| `src/eval_ragas.py` | RAGAS baseline-vs-final evaluation script |
| `src/cost_latency.py` | Runs the agent N times, reports average cost/latency/tool-call distribution |
| `tests/test_security.py` | 5 required injection tests + 4 bonus guardrail tests |

## One non-obvious design decision, explained

**Why gate documents/tool output through L1 too, not just user input.**
Most injection-defense designs only screen the user's own message. Here, L1
is applied a second time to every string that comes back from a tool call
(`search_food_security_reports`, etc.) before it is added to the evidence
context handed to the synthesis LLM. The trade-off: this costs an extra
regex pass per tool call (cheap) but closes a real attack surface — an
indirect prompt injection embedded inside a *retrieved document* (e.g. a
poisoned report file containing "ignore previous instructions...") would
otherwise reach the LLM unfiltered, since the user never typed it. Given that
this agent's evidentiary corpus is exactly the kind of untrusted, document-like
content indirect injection research shows is a common attack vector for
RAG agents, treating tool output as no more trustworthy than user input by
default was judged worth the small added latency.

## What would break first in production

See REPORT.md §6 (Limitations & what's next).
