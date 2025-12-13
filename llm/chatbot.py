from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from llm.prompts import PromptTemplates
from rag.taxonomy import TaxonomyEngine


class BOGChatbot:

    def __init__(self, retriever, config: Dict[str, Any]):

        self.retriever = retriever
        self.config = config
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get('temperature', 0.7)
        self.conversation_history = []

        repo_root = Path(__file__).resolve().parents[1]
        self.taxonomy = TaxonomyEngine.from_repo_root(repo_root)
        
    def chat(self, user_message: str) -> str:

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

        # 1) Retrieve
        results = self.retriever.retrieve(query)

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
        user_prompt = PromptTemplates.create_user_message(query=query, offers=results)
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

        return answer
    
    def reset_conversation(self):
        """
        Reset conversation history.
        """
        self.conversation_history = []
    
    def _call_llm(self, messages: List[Dict[str, str]], provider: str, model: str) -> str:
        """
        Make API call to LLM.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            LLM response
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
