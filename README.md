# Food Security Early Warning Agent

A production AI agent that answers analyst questions about acute food
insecurity risk (IPC phases, drivers, early-warning signals) for a given
region, grounded in a local corpus of food-security reports plus a live
climate data tool — with a security stack, structured/self-consistent
reasoning, a critic pass, and full observability.

Built for the "Production AI Agent" homework (topic #3 — Food security /
crop failure early warning). See `docs/architecture.md` for the full
architecture diagram and `REPORT.md` for the write-up.

## Setup

```bash
git clone https://github.com/nataname78/food-security-agent.git
cd food-security-agent
cp .env.example .env      # fill in MISTRAL_API_KEY (free tier: console.mistral.ai)
                          # Langfuse keys are optional — leave blank to disable tracing
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python src/agent.py "Is Niger's Tillabery region at risk of deteriorating past Crisis level?"
```

The first run downloads two small models from Hugging Face
(`all-MiniLM-L6-v2` for dense embeddings, `ms-marco-MiniLM-L-6-v2` for
reranking) — this needs a normal internet connection once; after that they're
cached locally. If these models can't be reached, `retrieval.py` degrades
gracefully to BM25-only search rather than crashing.

## Repository structure

```
├── README.md                 # this file
├── REPORT.md                 # the written report
├── requirements.txt
├── .env.example
├── src/
│   ├── agent.py               # main agent loop (analyst + critic + guardrails + tracing)
│   ├── mcp_server.py           # custom MCP server, 3 tools
│   ├── retrieval.py            # hybrid BM25 + dense + RRF + cross-encoder reranking
│   ├── guardrails.py            # L1 input filter, L4 action gate, TokenBudget
│   ├── reasoning.py             # few-shot CoT + Self-Consistency k=3
│   ├── eval_ragas.py            # RAGAS baseline-vs-final evaluation
│   └── cost_latency.py          # cost/latency/tool-distribution measurement over N runs
├── tests/
│   └── test_security.py        # 5 injection tests (+4 bonus guardrail tests)
├── docs/
│   └── architecture.md         # architecture diagram + component descriptions
└── data/
    ├── README.md                # what's in data/, how to refresh with real reports
    ├── corpus/                  # 8 synthetic food-security report documents
    └── eval/questions.jsonl     # RAGAS evaluation question set (12 Q/A pairs)
```

## Running the tests

```bash
python -m pytest tests/test_security.py -v
```

## Running the evaluation

```bash
python src/eval_ragas.py       # writes docs/ragas_results.json
python src/cost_latency.py --runs 10   # writes docs/cost_latency_results.json
```

Both scripts need `MISTRAL_API_KEY` set (they make real LLM calls) and are
not runnable in a fully network-restricted sandbox — run them locally or in
CI with normal internet access.

## MCP server (standalone / inspector testing)

```bash
python src/mcp_server.py                              # run standalone (stdio)
npx @modelcontextprotocol/inspector python src/mcp_server.py   # interactive inspector
```

## Notes on data

The `data/corpus/` documents are **synthetic** reports written for this
project (not copy-pasted from FEWS NET/FAO), grounded in real 2026 regional
dynamics and IPC terminology — this avoids any copyright issue while keeping
retrieval and evaluation meaningful. See `data/README.md` for how to swap in
real public reports.
