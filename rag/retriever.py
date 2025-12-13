from __future__ import annotations

import os
import re
from typing import List, Dict, Any


_WORD_RE = re.compile(r"[0-9A-Za-z\u10A0-\u10FF]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


def _is_recommendation_query(query: str) -> bool:
    q = (query or "").lower()
    # Very small heuristic: if user explicitly asks for recommendations/offers, don't force single-offer answers.
    keywords = [
        "მირჩიე",
        "რეკომენდ",
        "შემომთავ",
        "შეთავაზ",
        "suggest",
        "recommend",
        "best",
        "offers",
    ]
    return any(k in q for k in keywords)


def _is_factual_question(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if "?" in q:
        return True

    starters = (
        "ვინ",
        "როდის",
        "სად",
        "რა",
        "როგორ",
        "რამდენ",
        "who",
        "when",
        "where",
        "what",
        "how",
        "which",
    )
    return q.startswith(starters)


class OfferRetriever:

    def __init__(self, embedding_generator, vector_store, config: Dict[str, Any]):

        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.config = config
        self.top_k = int(config.get('top_k_results', 5))

        # Optional tuning knobs
        self.min_similarity_score = float(config.get("min_similarity_score", 0.0))
        self.lexical_boost = float(config.get("lexical_boost", 0.15))
        self.max_results_for_prompt = int(config.get("max_results_for_prompt", self.top_k))
        self.max_factual_results = int(config.get("max_factual_results", 1))
        self.min_factual_overlap = float(config.get("min_factual_overlap", 0.10))
        
    def retrieve(self, query: str) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            return []

        query_vec = self.embedding_generator.generate_embeddings([query.strip()])[0]
        results = self.vector_store.similarity_search(query_vec, top_k=self.top_k)

        offers: List[Dict[str, Any]] = []
        for payload, score in results:
            # Keep original payload; extract text separately for prompt formatting.
            offers.append(
                {
                    "score": score,
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                }
            )

        return self.rerank_results(query=query, results=offers)
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        if not results:
            return []

        query_str = (query or "").strip()
        query_tokens = set(_tokenize(query_str))

        is_rec = _is_recommendation_query(query_str)
        is_fact = _is_factual_question(query_str) and not is_rec

        reranked: List[Dict[str, Any]] = []
        for r in results:
            score = float(r.get("score") or 0.0)
            if score < self.min_similarity_score:
                continue

            meta = r.get("metadata", {}) or {}
            title = str(meta.get("title") or "")
            brand = str(meta.get("brand_name") or "")
            desc = str(meta.get("description") or "")
            text = str(r.get("text") or "")

            doc_tokens = set(_tokenize(" ".join([title, brand, desc, text])))
            overlap = 0.0
            if query_tokens:
                overlap = len(query_tokens.intersection(doc_tokens)) / max(1, len(query_tokens))

            if is_fact and overlap < self.min_factual_overlap:
                # For factual questions about a specific thing (“who will sing on X”),
                # require at least some lexical overlap so we don't surface random nearby vectors.
                continue

            combined = score + (self.lexical_boost * overlap)
            out = dict(r)
            out["_overlap"] = overlap
            out["_combined_score"] = combined
            reranked.append(out)

        if not reranked:
            return []

        reranked.sort(key=lambda x: (float(x.get("_combined_score") or 0.0), float(x.get("score") or 0.0)), reverse=True)

        limit = self.max_results_for_prompt
        if is_fact:
            limit = max(1, self.max_factual_results)

        return reranked[: max(1, int(limit))]
