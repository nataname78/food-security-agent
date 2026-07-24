"""
eval_ragas.py — RAGAS baseline (naive top-k BM25) vs final (hybrid + rerank) evaluation.

Run:
    python src/eval_ragas.py

Produces:
    docs/ragas_results.json — raw scores, used to fill the table in REPORT.md

Note: requires MISTRAL_API_KEY in .env (used both to generate answers for each
question and, internally, by ragas' default LLM-judge metrics). If `ragas` fails
to import in your environment (a known fragile dependency chain involving
langchain_community), this script falls back to a lightweight manual
implementation of the same 4 metrics so it always produces a usable table —
but prefer fixing the ragas install (a clean `pip install -r requirements.txt`
in a fresh virtualenv usually resolves it) for a rubric-accurate run.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

from retrieval import HybridRetriever
from reasoning import self_consistency_synthesize

load_dotenv()

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "questions.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "ragas_results.json")


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_answer(question: str, context: str) -> str:
    from agent import call_llm  # lazy import: requires MISTRAL_API_KEY at call time

    synthesis = self_consistency_synthesize(call_llm, question, context, k=1)
    return synthesis.conclusion


def run_pipeline_variant(retriever: HybridRetriever, questions: list[dict], variant: str) -> list[dict]:
    rows = []
    for item in questions:
        q = item["question"]
        if variant == "baseline":
            results = retriever.search_baseline(q, top_k=3)
        else:
            results = retriever.search(q, top_k=3)
        contexts = [r.chunk.text for r in results]
        context_str = "\n\n".join(contexts)
        answer = generate_answer(q, context_str)
        rows.append(
            {
                "question": q,
                "ground_truth": item["ground_truth"],
                "contexts": contexts,
                "answer": answer,
            }
        )
    return rows


def score_with_ragas(rows: list[dict]) -> dict:
    from ragas import evaluate, SingleTurnSample, EvaluationDataset
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["ground_truth"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
    )
    df = result.to_pandas()
    return {
        "context_recall": float(df["context_recall"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
    }


def score_fallback(rows: list[dict]) -> dict:
    """Manual proxy metrics, used only if the `ragas` package fails to import.
    Not a substitute for real RAGAS scores in the final report — fix the
    dependency and re-run for the actual submission if at all possible."""
    import re

    def word_set(text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    recalls, precisions, faiths, relevs = [], [], [], []
    for r in rows:
        gt_words = word_set(r["ground_truth"])
        ctx_words = word_set(" ".join(r["contexts"]))
        ans_words = word_set(r["answer"])
        q_words = word_set(r["question"])

        recalls.append(len(gt_words & ctx_words) / max(1, len(gt_words)))
        precisions.append(len(gt_words & ctx_words) / max(1, len(ctx_words)) if ctx_words else 0.0)
        faiths.append(len(ans_words & ctx_words) / max(1, len(ans_words)) if ans_words else 0.0)
        relevs.append(len(ans_words & q_words) / max(1, len(q_words)) if q_words else 0.0)

    n = max(1, len(rows))
    return {
        "context_recall": sum(recalls) / n,
        "context_precision": sum(precisions) / n,
        "faithfulness": sum(faiths) / n,
        "answer_relevancy": sum(relevs) / n,
    }


def main():
    questions = load_questions()
    retriever = HybridRetriever(CORPUS_DIR, use_dense=True, use_reranker=True)

    print(f"Running baseline (naive BM25 top-k) over {len(questions)} questions...")
    baseline_rows = run_pipeline_variant(retriever, questions, "baseline")

    print(f"Running final (hybrid + rerank) over {len(questions)} questions...")
    final_rows = run_pipeline_variant(retriever, questions, "final")

    try:
        baseline_scores = score_with_ragas(baseline_rows)
        final_scores = score_with_ragas(final_rows)
        method = "ragas"
    except Exception as e:  # noqa: BLE001
        print(f"[eval_ragas] ragas unavailable ({e}); using fallback manual metrics.")
        baseline_scores = score_fallback(baseline_rows)
        final_scores = score_fallback(final_rows)
        method = "fallback"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"method": method, "baseline": baseline_scores, "final": final_scores},
            f,
            indent=2,
        )

    print("\n=== RAGAS RESULTS ===")
    print(f"Method used: {method}")
    print(f"{'Metric':<20} {'Baseline':>10} {'Final':>10}")
    for metric in ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]:
        print(f"{metric:<20} {baseline_scores[metric]:>10.3f} {final_scores[metric]:>10.3f}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
