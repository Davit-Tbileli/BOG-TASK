from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llm.prompts import PromptTemplates
from rag.taxonomy import TaxonomyEngine

logger = logging.getLogger(__name__)


_FOLLOW_UP_CUES = (
    # Georgian follow-up / deixis
    "ეს",
    "ეგ",
    "იმ",
    "ამ",
    "ამაზე",
    "იმაზე",
    "ამის",
    "იმის",
    "ამის შესახებ",
    "იმის შესახებ",
    "წინაზე",
    "წინა",
    "ბოლოს",
    "ბოლო",
    "კიდევ",
    "და კიდევ",
    "დამატებით",
    "უფრო დეტალურად",
    "დეტალები",
    "მეტად",
    "მითხარი მეტი",
    # English follow-up cues (just in case)
    "that",
    "this",
    "those",
    "it",
    "what about",
    "tell me more",
    "more details",
)


def _truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 1)].rstrip() + "…"


def _looks_like_follow_up(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False

    # Short / ambiguous questions often refer to the previous turn
    if len(q) <= 25:
        return True

    # If it contains follow-up cue words
    for cue in _FOLLOW_UP_CUES:
        if cue in q:
            return True

    # Questions that are mostly pronouns / generic
    generic_starts = (
        "და",
        "კიდევ",
        "მაშ",
        "ანუ",
        "ok",
        "კი",
        "არა",
    )
    if q.startswith(generic_starts):
        return True

    return False


def _offer_key(offer: Dict[str, Any]) -> str:
    meta = (offer or {}).get("metadata", {}) or {}
    return str(meta.get("details_url") or meta.get("id") or "")


def _merge_offers(preferred: List[Dict[str, Any]], fallback: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for src in (preferred, fallback):
        for o in src or []:
            k = _offer_key(o)
            if not k:
                # If no stable key, just append and hope it's unique enough
                out.append(o)
                continue
            if k in seen:
                continue
            seen.add(k)
            out.append(o)
            if len(out) >= limit:
                return out
    return out[:limit]

class BOGChatbot:
    """Main chatbot class for BOG offers assistance.
    
    Integrates retrieval, taxonomy normalization, and LLM generation
    to provide helpful responses about Bank of Georgia offers.
    
    Attributes:
        retriever: OfferRetriever for fetching relevant offers.
        config: Configuration dictionary for LLM and behavior settings.
        taxonomy: TaxonomyEngine for normalizing benefit information.
        conversation_history: List of past messages for context.
    """

    def __init__(self, retriever, config: Dict[str, Any]) -> None:
        """Initialize the chatbot.
        
        Args:
            retriever: OfferRetriever instance for fetching offers.
            config: Configuration dictionary with keys:
                - provider: LLM provider ('openai', 'gemini', 'none')
                - model: Model name (e.g., 'gpt-4o-mini')
                - temperature: Generation temperature (0-1)
                - history_to_keep: Number of conversation turns to retain
        """

        self.retriever = retriever
        self.config = config
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get('temperature', 0.7)
        self.conversation_history = []

        # Keep last retrieval results so follow-ups like “ეს რა პირობებია?” stay on-topic.
        self._last_results: List[Dict[str, Any]] = []
        self._last_user_query: str = ""

        repo_root = Path(__file__).resolve().parents[1]
        self.taxonomy = TaxonomyEngine.from_repo_root(repo_root)
        
    def chat(self, user_message: str) -> str:
        """Process a user message and generate a response.
        
        Args:
            user_message: The user's input text.
            
        Returns:
            A response string from the chatbot (LLM-generated or formatted offers).
        """
        load_dotenv()

        provider = (self.config.get("provider") or os.getenv("LLM_PROVIDER") or "").strip().lower()
        if not provider or provider == "auto":
            provider = "gemini" if os.getenv("GEMINI_API_KEY") else "openai"

        model = (self.config.get("model") or os.getenv("LLM_MODEL") or "").strip()
        if not model:
            model = "auto" if provider == "gemini" else "gpt-4o-mini"

        query = (user_message or "").strip()
        if not query:
            return "მკითხე რაიმე." 

        is_follow_up = _looks_like_follow_up(query) and bool(self._last_results)

        # 1) Retrieve
        # For follow-ups, expand retrieval query slightly and fall back to last results if needed.
        retrieval_query = query
        if is_follow_up and self._last_user_query:
            retrieval_query = f"{self._last_user_query}\nFollow-up: {query}".strip()

        results = self.retriever.retrieve(retrieval_query)

        if is_follow_up:
            # If retrieval returns nothing (common for pronoun-y follow-ups), reuse last offers.
            if not results:
                results = list(self._last_results)
            else:
                # Merge so we don't unexpectedly switch away from the previous offer set.
                max_keep = int(self.config.get("followup_offer_limit", 5))
                results = _merge_offers(preferred=self._last_results, fallback=results, limit=max_keep)

        # 2) Taxonomy normalize (attach to metadata for prompt formatting)
        for r in results:
            meta = r.get("metadata", {}) or {}
            normalized = self.taxonomy.normalize_offer(meta)
            if normalized is None:
                continue
            meta["benefit_type_id"] = normalized.benefit_type_id
            meta["benefit_label_ka"] = normalized.benefit_label_ka
            meta["benefit_value"] = normalized.value
            meta["benefit_unit"] = normalized.value_unit
            meta["benefit_rule_id"] = normalized.rule_id
            r["metadata"] = meta

        # 3) Build prompt
        system_prompt = (self.config.get("system_prompt") or PromptTemplates.SYSTEM_PROMPT).strip()

        prev_answer = ""
        if is_follow_up and self.conversation_history:
            # last assistant content if available
            for m in reversed(self.conversation_history):
                if m.get("role") == "assistant":
                    prev_answer = str(m.get("content") or "")
                    break

        user_prompt = PromptTemplates.create_user_message(
            query=query,
            offers=results,
            previous_query=self._last_user_query if is_follow_up else None,
            previous_answer=_truncate(prev_answer, int(self.config.get("followup_prev_answer_max_chars", 900))),
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Optional short conversation memory
        history_to_keep = int(self.config.get("history_to_keep", 4))
        if history_to_keep > 0 and self.conversation_history:
            messages.extend(self.conversation_history[-history_to_keep:])

        messages.append({"role": "user", "content": user_prompt})

        if provider in {"none", "off", "false"}:
            formatted = PromptTemplates.format_offers(results)
            return "\n\n".join(
                [
                    "LLM გამორთულია (--no-llm). ქვემოთ არის ნაპოვნი შეთავაზებები:",
                    formatted,
                ]
            )

        # 4) If no API key, fall back to deterministic formatted output.
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            formatted = PromptTemplates.format_offers(results)
            return "\n\n".join(
                [
                    "OpenAI API key ვერ მოიძებნა (OPENAI_API_KEY). ქვემოთ არის ნაპოვნი შეთავაზებები:",
                    formatted,
                ]
            )

        if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            formatted = PromptTemplates.format_offers(results)
            return "\n\n".join(
                [
                    "Gemini API key ვერ მოიძებნა (GEMINI_API_KEY). ქვემოთ არის ნაპოვნი შეთავაზებები:",
                    formatted,
                ]
            )

        answer = self._call_llm(messages, provider=provider, model=model)

        # Save minimal history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Update follow-up context after producing a response.
        self._last_user_query = query
        self._last_results = list(results)

        return answer
    
    def reset_conversation(self) -> None:
        """Clear the conversation history.
        
        Use this to start a fresh conversation without context from previous exchanges.
        """
        self.conversation_history = []
        self._last_results = []
        self._last_user_query = ""
    
    def _call_llm(self, messages: List[Dict[str, str]], provider: str, model: str) -> str:
        """Make an API call to the configured LLM provider.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            provider: LLM provider name ('openai' or 'gemini').
            model: Model name to use.
            
        Returns:
            Generated response text.
            
        Raises:
            ValueError: If provider is not supported.
            RuntimeError: If API key is missing or API call fails.
        """
        if provider == "openai":
            # Using OpenAI python SDK (v1.x)
            from openai import OpenAI  # type: ignore

            client = OpenAI()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(self.temperature),
            )
            return resp.choices[0].message.content or ""

        if provider == "gemini":
            try:
                import google.generativeai as genai  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Gemini provider არჩეულია, მაგრამ 'google-generativeai' არ არის დაყენებული. "
                    "გაუშვი: pip install -r requirements.txt"
                ) from e

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY ვერ მოიძებნა.")

            genai.configure(api_key=api_key)

            system_prompt = ""
            if messages and messages[0].get("role") == "system":
                system_prompt = messages[0].get("content") or ""

            # Gemini SDK doesn't use OpenAI-style role arrays by default; pack as a single transcript.
            transcript_parts: List[str] = []
            for m in messages[1:] if system_prompt else messages:
                role = (m.get("role") or "user").upper()
                content = (m.get("content") or "").strip()
                if content:
                    transcript_parts.append(f"{role}: {content}")
            transcript = "\n\n".join(transcript_parts) if transcript_parts else ""

            def _resolve_model_name(requested: str) -> str:
                req = (requested or "").strip()
                if req.lower() in {"", "auto"}:
                    req = ""

                models = list(genai.list_models())
                available: List[str] = []
                for m in models:
                    name = getattr(m, "name", "") or ""
                    if name.startswith("models/"):
                        name = name.split("/", 1)[1]
                    supported = getattr(m, "supported_generation_methods", None) or []
                    if "generateContent" in supported:
                        available.append(name)

                if not available:
                    return requested

                if req and req in available:
                    return req

                if req:
                    for n in available:
                        if n.startswith(req):
                            return n

                preferred = [
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                ]
                for p in preferred:
                    for n in available:
                        if n == p or n.startswith(p):
                            return n

                return available[0]

            resolved_model = _resolve_model_name(model)
            gemini_model = genai.GenerativeModel(model_name=resolved_model, system_instruction=system_prompt or None)
            resp = gemini_model.generate_content(
                transcript,
                generation_config={
                    "temperature": float(self.temperature),
                },
            )

            text = getattr(resp, "text", None)
            if text:
                return str(text)

            # Fallback for SDK response shapes
            return str(resp)

        raise ValueError(f"Unsupported provider: {provider}")


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_default_chatbot() -> BOGChatbot:
    """Build a fully wired chatbot from .env configuration."""
    load_dotenv()

    # Lazy imports to keep module import light.
    from rag.embeddings import EmbeddingGenerator
    from rag.retriever import OfferRetriever
    from rag.vector_store import VectorStore

    qdrant_url = _require_env("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = (os.getenv("COLLECTION_NAME") or "bog_offers").strip()
    qdrant_distance = (os.getenv("QDRANT_DISTANCE") or "COSINE").strip()

    embedding_model = (os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3").strip()
    top_k = int((os.getenv("TOP_K") or "5").strip())

    # Relevance tuning (optional)
    min_similarity_score = float((os.getenv("MIN_SIMILARITY_SCORE") or "0.0").strip())
    lexical_boost = float((os.getenv("LEXICAL_BOOST") or "0.15").strip())
    max_results_for_prompt = int((os.getenv("MAX_RESULTS_FOR_PROMPT") or str(top_k)).strip())
    max_factual_results = int((os.getenv("MAX_FACTUAL_RESULTS") or "1").strip())
    min_factual_overlap = float((os.getenv("MIN_FACTUAL_OVERLAP") or "0.10").strip())

    vector_store = VectorStore(
        {
            "vector_store": "qdrant",
            "qdrant_url": qdrant_url,
            "qdrant_api_key": qdrant_api_key,
            "collection_name": collection_name,
            "qdrant_distance": qdrant_distance,
            # Keep generous defaults for cloud connections.
            "qdrant_timeout_seconds": float((os.getenv("QDRANT_TIMEOUT_SECONDS") or "120").strip()),
            "upsert_batch_size": int((os.getenv("UPSERT_BATCH_SIZE") or "64").strip()),
        }
    )

    embedding_generator = EmbeddingGenerator(model_name=embedding_model, normalize_embeddings=True)

    retriever = OfferRetriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        config={
            "top_k_results": top_k,
            "min_similarity_score": min_similarity_score,
            "lexical_boost": lexical_boost,
            "max_results_for_prompt": max_results_for_prompt,
            "max_factual_results": max_factual_results,
            "min_factual_overlap": min_factual_overlap,
        },
    )

    chatbot_config: Dict[str, Any] = {
        "provider": (os.getenv("LLM_PROVIDER") or "auto").strip().lower(),
        "model": (os.getenv("LLM_MODEL") or "").strip(),
        "temperature": float((os.getenv("LLM_TEMPERATURE") or "0.2").strip()),
        "history_to_keep": int((os.getenv("HISTORY_TO_KEEP") or "4").strip()),
    }
    return BOGChatbot(retriever=retriever, config=chatbot_config)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="BOG offers chatbot (RAG + taxonomy + LLM)")
    parser.add_argument("query", nargs="*", help="One-shot query. If omitted, runs interactive mode.")
    parser.add_argument("--no-llm", action="store_true", help="Never call an LLM; print retrieved offers only.")
    args = parser.parse_args(argv)

    bot = build_default_chatbot()

    if args.no_llm:
        bot.config["provider"] = "none"

    # One-shot
    if args.query:
        q = " ".join(args.query).strip()
        print(bot.chat(q))
        return 0

    # Interactive
    print("BOG chatbot ready. Type 'exit' to quit.")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        print(bot.chat(q))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
