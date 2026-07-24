"""
reasoning.py — Few-shot CoT synthesis + Self-Consistency (k=3) for the
Food Security Early Warning agent.

The synthesis step is the final step of the pipeline: given the user question
and the retrieved/tool evidence, produce a structured answer in the
EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE format, sampled k times at
non-zero temperature, then reconciled by majority vote on the CONCLUSION field.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

FEW_SHOT_SYNTHESIS_PROMPT = """You are a food-security analysis assistant. You answer strictly from the
EVIDENCE provided to you — you never invent facts not present in the evidence.
Always structure your answer in exactly these four sections:

EVIDENCE: bullet points quoting/paraphrasing only the facts you were given that are relevant to the question.
ANALYSIS: your reasoning connecting the evidence to the question, step by step.
CONCLUSION: one or two sentences giving the direct answer.
CONFIDENCE: one of [low, medium, high], based on how many independent sources corroborate the conclusion.

---
Example 1

Question: Is Niger's Tillabéry region at risk of deteriorating past Crisis level this outlook period?

EVIDENCE:
- Tillabéry households are depleting stocks earlier than seasonal average due to conflict-restricted field access.
- Crisis-level (Phase 3) outcomes are already reported in conflict-affected departments.
- The outlook states a risk of localized deterioration to Emergency (Phase 4) if displacement accelerates.

ANALYSIS:
The current classification is already Crisis (Phase 3), and the source explicitly names conflict-driven
displacement as the variable that would push the situation to the next phase. Since displacement is described
as an active, ongoing driver rather than a resolved one, the conditions for further deterioration are present
in the source itself, not inferred.

CONCLUSION: Yes — Tillabéry is at risk of deteriorating to Emergency (Phase 4) if conflict-driven displacement
continues to accelerate, per the outlook's own stated risk scenario.

CONFIDENCE: high

---
Example 2

Question: Will Ethiopia's harvest improve this year?

EVIDENCE:
- Elevated probability (50-60%) of El Niño developing during the kiremt rains.
- El Niño historically correlates with reduced rainfall in southern pastoral zones and above-average
  rainfall in some western cropping zones.

ANALYSIS:
The evidence gives a probabilistic climate driver with opposing regional effects (drier south, wetter west),
not a single national harvest outcome. A single yes/no answer at national level would overstate the certainty
and direction implied by the source.

CONCLUSION: The picture is mixed and regionally dependent — western cropping areas could see gains while
southern pastoral zones face elevated risk; there isn't enough evidence to give one national-level answer.

CONFIDENCE: medium

---

Now answer the following question using ONLY the evidence given to you.
"""


@dataclass
class SynthesisResult:
    raw_samples: list[str]
    evidence: str
    analysis: str
    conclusion: str
    confidence: str
    self_consistency_agreement: float  # fraction of k samples whose conclusion matched the majority


def _extract_section(text: str, section: str, next_sections: list[str]) -> str:
    pattern = rf"{section}:\s*(.*?)(?=\n(?:{'|'.join(next_sections)}):|\Z)"
    m = re.search(pattern, text, re.S | re.I)
    return m.group(1).strip() if m else ""


def parse_structured_answer(text: str) -> dict:
    return {
        "evidence": _extract_section(text, "EVIDENCE", ["ANALYSIS", "CONCLUSION", "CONFIDENCE"]),
        "analysis": _extract_section(text, "ANALYSIS", ["CONCLUSION", "CONFIDENCE"]),
        "conclusion": _extract_section(text, "CONCLUSION", ["CONFIDENCE"]),
        "confidence": _extract_section(text, "CONFIDENCE", []).lower().strip(),
    }


def _normalize_conclusion_for_voting(conclusion: str) -> str:
    """Very coarse normalization so near-identical conclusions across samples can be
    grouped for majority voting (full semantic equivalence checking is out of scope;
    a stricter implementation would embed conclusions and cluster them)."""
    return re.sub(r"\W+", " ", conclusion.lower()).strip()[:120]


def self_consistency_synthesize(llm_call_fn, question: str, evidence_context: str, k: int = 3) -> SynthesisResult:
    """Run the synthesis prompt k times (non-zero temperature expected from llm_call_fn)
    and reconcile via majority vote over conclusions.

    llm_call_fn: callable(system_prompt: str, user_prompt: str) -> str
                 (kept generic so it can be swapped for a mock in tests, or the real
                 Mistral client in agent.py)
    """
    user_prompt = f"Question: {question}\n\nEvidence:\n{evidence_context}\n"
    samples = [llm_call_fn(FEW_SHOT_SYNTHESIS_PROMPT, user_prompt) for _ in range(k)]
    parsed = [parse_structured_answer(s) for s in samples]

    normalized = [_normalize_conclusion_for_voting(p["conclusion"]) for p in parsed]
    vote_counts = Counter(normalized)
    majority_key, majority_count = vote_counts.most_common(1)[0]

    # Pick the first sample whose normalized conclusion matches the majority as the
    # representative answer (keeps its EVIDENCE/ANALYSIS/CONFIDENCE intact).
    winner_idx = normalized.index(majority_key)
    winner = parsed[winner_idx]
    agreement = majority_count / max(1, len(parsed))

    return SynthesisResult(
        raw_samples=samples,
        evidence=winner["evidence"],
        analysis=winner["analysis"],
        conclusion=winner["conclusion"],
        confidence=winner["confidence"] or "low",
        self_consistency_agreement=agreement,
    )


if __name__ == "__main__":
    # Smoke test with a mocked LLM (no network / no API key required).
    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        return (
            "EVIDENCE:\n- Mock evidence line.\n\n"
            "ANALYSIS:\nMock reasoning.\n\n"
            "CONCLUSION: Mock conclusion, always the same for this test.\n\n"
            "CONFIDENCE: medium"
        )

    result = self_consistency_synthesize(mock_llm, "Test question?", "Mock evidence context", k=3)
    print(result)
    assert result.self_consistency_agreement == 1.0
    print("OK — self-consistency smoke test passed")
