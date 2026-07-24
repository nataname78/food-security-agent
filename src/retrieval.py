"""
retrieval.py — Hybrid retrieval pipeline for the Food Security Early Warning agent.

Pipeline:
    1. Load + parent-child chunk the corpus (parent = whole document, child = paragraph)
    2. BM25 sparse search over child chunks
    3. Dense search over child chunks (sentence-transformers embeddings)
    4. Reciprocal Rank Fusion (RRF) to combine BM25 + dense rankings
    5. Cross-encoder reranking of the fused candidates
    6. Parent expansion: return the parent document for each surviving child chunk,
       so the LLM gets full context rather than an isolated paragraph.

This module has no LLM dependency and no network dependency except the one-time
download of the embedding / cross-encoder models from Hugging Face on first run.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9àâäéèêëîïôöùûüç]+", text.lower())


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    text: str          # child-level text (paragraph)
    parent_text: str   # full parent document text
    source: str        # filename


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    stage: str = field(default="")  # "bm25", "dense", "rrf", "rerank" — for debugging/observability


class HybridRetriever:
    def __init__(self, corpus_dir: str, use_dense: bool = True, use_reranker: bool = True):
        self.corpus_dir = corpus_dir
        self.use_dense = use_dense
        self.use_reranker = use_reranker
        self.chunks: list[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_tokenized: list[list[str]] = []
        self._embedder = None
        self._cross_encoder = None
        self._dense_index = None  # (n_chunks, dim) numpy array
        self._load_corpus()
        self._build_bm25()
        if self.use_dense:
            self._build_dense_index()
        if self.use_reranker:
            self._load_cross_encoder()

    # ---------- Indexing ----------

    def _load_corpus(self) -> None:
        paths = sorted(glob.glob(os.path.join(self.corpus_dir, "*.md")))
        if not paths:
            raise FileNotFoundError(f"No .md files found in {self.corpus_dir}")
        for path in paths:
            source = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                parent_text = f.read()
            # Parent-child chunking: split parent doc into paragraph-level children.
            paragraphs = [p.strip() for p in parent_text.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                if len(para) < 30:  # skip trivial headers/blank fragments
                    continue
                self.chunks.append(
                    Chunk(
                        chunk_id=f"{source}::{i}",
                        parent_id=source,
                        text=para,
                        parent_text=parent_text,
                        source=source,
                    )
                )

    def _build_bm25(self) -> None:
        self._bm25_tokenized = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._bm25_tokenized)

    def _build_dense_index(self) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(EMBED_MODEL_NAME)
            texts = [c.text for c in self.chunks]
            self._dense_index = self._embedder.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
        except Exception as e:  # noqa: BLE001 — degrade gracefully, BM25-only fallback
            print(f"[retrieval] Dense index disabled (embedding model unavailable): {e}")
            self.use_dense = False
            self._embedder = None
            self._dense_index = None

    def _load_cross_encoder(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] Cross-encoder reranking disabled (model unavailable): {e}")
            self.use_reranker = False
            self._cross_encoder = None

    # ---------- Search ----------

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [RetrievalResult(chunk=self.chunks[i], score=float(scores[i]), stage="bm25") for i in ranked]

    def _dense_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self.use_dense or self._embedder is None:
            return []
        import numpy as np

        q_emb = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self._dense_index @ q_emb
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [RetrievalResult(chunk=self.chunks[i], score=float(sims[i]), stage="dense") for i in ranked]

    @staticmethod
    def _rrf_fuse(
        result_lists: list[list[RetrievalResult]], k: int = 60, top_k: int = 10
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank_in_list)."""
        fused_scores: dict[str, float] = {}
        chunk_lookup: dict[str, Chunk] = {}
        for results in result_lists:
            for rank, r in enumerate(results):
                fused_scores[r.chunk.chunk_id] = fused_scores.get(r.chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
                chunk_lookup[r.chunk.chunk_id] = r.chunk
        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=chunk_lookup[cid], score=fused_scores[cid], stage="rrf") for cid in ranked_ids
        ]

    def _rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not self.use_reranker or self._cross_encoder is None or not candidates:
            return candidates[:top_k]
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._cross_encoder.predict(pairs)
        rescored = [
            RetrievalResult(chunk=c.chunk, score=float(s), stage="rerank")
            for c, s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]

    def search(self, query: str, top_k: int = 5, fusion_pool: int = 15) -> list[RetrievalResult]:
        """Full hybrid pipeline: BM25 + dense -> RRF fuse -> cross-encoder rerank -> top_k."""
        bm25_results = self._bm25_search(query, top_k=fusion_pool)
        dense_results = self._dense_search(query, top_k=fusion_pool) if self.use_dense else []
        candidate_lists = [bm25_results] + ([dense_results] if dense_results else [])
        fused = self._rrf_fuse(candidate_lists, top_k=fusion_pool)
        reranked = self._rerank(query, fused, top_k=top_k)
        return reranked

    def search_baseline(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Naive baseline for the RAGAS before/after comparison: BM25 top-k only, no rerank."""
        return self._bm25_search(query, top_k=top_k)

    def get_parent_context(self, results: list[RetrievalResult]) -> str:
        """Expand surviving child chunks to their parent documents (deduplicated) for LLM context."""
        seen_parents: set[str] = set()
        parts = []
        for r in results:
            if r.chunk.parent_id in seen_parents:
                continue
            seen_parents.add(r.chunk.parent_id)
            parts.append(f"[Source: {r.chunk.source}]\n{r.chunk.parent_text}")
        return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    corpus_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
    retriever = HybridRetriever(corpus_dir, use_dense=True, use_reranker=True)
    query = "What remote sensing indicators predict crop failure before market prices react?"
    for r in retriever.search(query, top_k=3):
        print(f"{r.score:.3f} [{r.stage}] {r.chunk.chunk_id}: {r.chunk.text[:100]}...")
