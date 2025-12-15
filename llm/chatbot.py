from __future__ import annotations

import logging
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llm.prompts import PromptTemplates
from rag.taxonomy import TaxonomyEngine

logger = logging.getLogger(__name__)


_COUNT_RE = re.compile(r"\b(\d{1,2})\b")

# Regex for tokenizing Georgian and Latin text (keep consistent with retriever)
_WORD_RE = re.compile(r"[0-9A-Za-z\u10A0-\u10FF]+", re.UNICODE)

_FOLLOW_UP_CUES = (
    # Georgian follow-up
    "შესახებ",
    "წინა",
    "ბოლო",
    "კიდევ",
    "დამატებით",
    "დეტალ",
    "პირველი",
    "მეორე",
    "მესანმე",
    # English follow-up cues (just in case)
    "that",
    "this",
    "those",
    "it",
    "what about",
    "tell me more",
    "more details",
)


_DATE_QUESTION_CUES = (
    # Georgian
    "როდემდე",
    "სადამდე",
    "ვადამდე",
    "როდიდან",
    "საიდან",
    "პერიოდი",
    "თარიღ",
    "დაწყება",
    "დასრულება",
    "მოქმედებს",
    # English
    "until",
    "till",
    "valid",
    "expires",
    "expiration",
    "end date",
    "start date",
    "from when",
    "when does",
)


def _truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 1)].rstrip() + "…"


def _looks_like_follow_up(query: str, previous_category: str = "", category_labels: Optional[List[str]] = None) -> bool:
    """Check if query is referencing previous results.
    
    Follow-ups are ONLY queries that reference previously shown offers by ordinal/position:
    - "პირველი შემოთავაზების დეტალები მინდა" (details of the first offer)
    - "მეორე რას სთავაზობს?" (what does the second one offer?)
    - "show me the first one"
    - "details of #2"
    
    All other queries (categories, filters, questions) are treated as NEW independent queries.
    """
    q = (query or "").strip().lower()
    if not q:
        return False

    # Ordinal references in Georgian
    georgian_ordinals = [
        "პირველი", "მეორე", "მესამე", "მეოთხე", "მეხუთე",
        "უკანასკნელი", "ბოლო"
    ]
    
    # Ordinal references in English
    english_ordinals = [
        "first", "second", "third", "fourth", "fifth",
        "last", "#1", "#2", "#3", "#4", "#5",
        "number 1", "number 2", "number 3"
    ]
    
    # Check for ordinal references
    all_ordinals = georgian_ordinals + english_ordinals
    if any(ordinal in q for ordinal in all_ordinals):
        return True
    
    # NOT a follow-up - treat as independent query
    return False


def _is_date_question(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(cue in q for cue in _DATE_QUESTION_CUES)


def _format_offer_period_answer(offer: Dict[str, Any]) -> str:
    meta = (offer or {}).get("metadata", {}) or {}
    start_date = str(meta.get("start_date") or "").strip()
    end_date = str(meta.get("end_date") or "").strip()
    details_url = str(meta.get("details_url") or "").strip()

    brand = str(meta.get("brand_name") or "").strip()
    title = str(meta.get("title") or "").strip()

    label = ""
    if brand and title:
        label = f"{brand} — {title}"
    elif title:
        label = title
    elif brand:
        label = brand

    if start_date or end_date:
        # Always return both start/end if we have them.
        period = f"{start_date} - {end_date}".strip(" -")
        head = f"პერიოდი: {period}" if period else ""
        if label:
            head = f"{label}\n{head}" if head else label
        if details_url:
            head = f"{head}\nბმული: {details_url}" if head else f"ბმული: {details_url}"
        return head.strip()

    # No dates available in data.
    if details_url:
        return f"ამ შეთავაზებაზე პერიოდის თარიღები არ ჩანს. ბმული: {details_url}".strip()
    return "ამ შეთავაზებაზე პერიოდის თარიღები არ ჩანს.".strip()

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


def _load_city_labels(repo_root: Path) -> List[str]:
    """Load city labels from data/raw/cities.json.

    Returns a list of city labels as stored in the dataset (Georgian labels).
    If file is missing or unreadable, returns empty list.
    """

    cities_path = repo_root / "data" / "raw" / "cities.json"
    try:
        with cities_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        labels: List[str] = []
        for item in data or []:
            if isinstance(item, dict):
                label = str(item.get("label") or "").strip()
                if label:
                    labels.append(label)
        # Prefer longer labels first to avoid partial matches.
        labels.sort(key=len, reverse=True)
        return labels
    except Exception:
        return []


def _load_category_labels(repo_root: Path) -> List[str]:
    """Load distinct category_desc values from data/processed/found_offers.json."""

    offers_path = repo_root / "data" / "processed" / "found_offers.json"
    try:
        with offers_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cats: set[str] = set()
        for offer in data or []:
            if not isinstance(offer, dict):
                continue
            c = str(offer.get("category_desc") or "").strip()
            if c:
                cats.add(c)
        out = sorted(cats, key=len, reverse=True)
        return out
    except Exception:
        return []


# Semantic mappings: Georgian query words → category_desc value
_CATEGORY_KEYWORDS = {
    # გართობა და კულტურა (Entertainment & Culture)
    "გართობა და კულტურა": [
        "გართობ", "გავერთბი", "გართობის", "გასართობი",
        "კულტურ", "დასვენება", "entertainment", "fun", "enjoy"
    ],
    # შოპინგი (Shopping)
    "შოპინგი": [
        "შოპინგ", "შოფინგ", "shopping", "მაღაზი", "მაღაზიებ",
        "საყიდლ", "შოპ", "shop",
        "ტანსაცმელ", "სამოს", "ჩაცმულობ", 
    ],
    # კვება (Food)
    "კვება": [
        "კვებ", "რესტორან", "კაფე", "საჭმელ", "ჭამ",
    ],
    # დასვენება (Relaxation/Vacation)
    "დასვენება": [
        "დასვენებ", "დამსვენება", "relax", "spa",
        "სასტუმრო", "ჰოტელ", "hotel", "accommodation"
    ],
    # თავის მოვლა (Self-care/Beauty)
    "თავის მოვლა": [
        "თავის მოვლა", "სილამაზე", "სალონ", "beauty", "spa"
    ],
    # მოგზაურობა (Travel)
    "მოგზაურობა": [
        "მოგზაურობ", "ფრენ", "ავია", "travel", "flight"
    ],
    # განათლება (Education)
    "განათლება": [
        "განათლებ", "სწავლ", "ტრენინგ", "education", "training", "course"
    ],
    # სახლი და ოჯახი (Home & Family)
    "სახლი და ოჯახი": [
        "სახლ", "ოჯახ", "home", "family", "household"
    ],
    # ავტომობილები (Automobiles)
    "ავტომობილები": [
        "ავტომობილ", "მანქან", "car", "auto"
    ],
    # ტექნიკა (Technology)
    "ტექნიკა": [
        "ტექნიკ", "გაჯეტ", "კომპიუტერ", "tech", "gadget"
    ]
}


def _extract_all_categories(query: str, category_labels: List[str]) -> List[str]:
    """Extract ALL categories mentioned in query (for multi-category queries)."""
    q = (query or "").strip().lower()
    if not q or not category_labels:
        return []
    
    detected = []
    # Find all matching categories using shared keyword mapping
    for category_label in category_labels:
        keywords = _CATEGORY_KEYWORDS.get(category_label, [])
        for kw in keywords:
            if kw in q:
                if category_label not in detected:
                    detected.append(category_label)
                break
    
    # Fallback: substring match on the category label itself
    if not detected:
        for label in category_labels:
            lab = label.lower()
            if lab and lab in q:
                if label not in detected:
                    detected.append(label)
    
    return detected


def _extract_city(query: str, city_labels: List[str]) -> str:
    q = (query or "").strip().lower()
    if not q or not city_labels:
        return ""

    for label in city_labels:
        lab = label.lower()
        if lab and lab in q:
            return label

        # Georgian city labels often end with "ი" but user text may be inflected (e.g., "თბილისში").
        if lab.endswith("ი"):
            stem = lab[:-1]
            if stem and stem in q:
                return label
    return ""


def _parse_requested_count(query: str) -> Optional[int]:

    q = (query or "").lower()
    if not q:
        return None

    m = _COUNT_RE.search(q)
    if not m:
        return None

    n = int(m.group(1))
    if n <= 0:
        return None

    # Only accept if query mentions a collection-like noun.
    triggers = (
        "შეთავაზ",
        "ვარიანტ",
        "ადგილ",
    )
    if any(t in q for t in triggers):
        return max(1, min(20, n))
    return None


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


def _extract_topic_keywords(query: str, city: str = "", limit: int = 5) -> List[str]:
    """Extract a few stable keywords to carry context across turns."""
    q = (query or "").strip().lower()
    if not q:
        return []

    tokens = _tokenize(q)
    out: List[str] = []
    city_l = (city or "").strip().lower()
    city_stem = city_l[:-1] if city_l.endswith("ი") else city_l

    for t in tokens:
        if not t or t.isdigit():
            continue
        if len(t) <= 2:
            continue
        if city_l and city_l in t:
            continue
        if city_stem and city_stem in t:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= limit:
            break

    return out


def _extract_benefit_hint(query: str) -> str:
    q = (query or "").strip().lower()
    if not q:
        return ""

    # If user explicitly mentions cashback.
    if "ქეშბექ" in q or "cashback" in q:
        return "CASHBACK_PERCENT"

    # Points / MR multipliers.
    if "ქულ" in q or "mr" in q or "points" in q or "point" in q:
        return "POINTS_MULTIPLIER"

    # Discount.
    if "ფასდაკ" in q or "discount" in q or "%" in q:
        return "DISCOUNT_PERCENT"

    return ""


def _benefit_payload_constraints(benefit_hint: str) -> Dict[str, Any]:
    """Map a benefit hint to raw payload constraints (as stored in Qdrant payload)."""

    hint = (benefit_hint or "").strip().upper()
    if hint == "CASHBACK_PERCENT":
        return {"benef_name": "CASHBACK", "benef_badge": "%"}
    if hint == "DISCOUNT_PERCENT":
        return {"benef_name": "DISCOUNT", "benef_badge": "%"}
    if hint == "POINTS_MULTIPLIER":
        return {"benef_name": "MR", "benef_badge": "X"}
    return {}


def _is_category_list_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if "კატეგორი" not in q:
        return False
    triggers = (
        "რა",
        "რომელი",
        "გაქვს",
        "არსებ",
        "ჩამომითვალ",
        "სია",
        "list",
        "categories",
    )
    return any(t in q for t in triggers)


def _format_category_list(categories: List[str]) -> str:
    cats = [c.strip() for c in (categories or []) if str(c).strip()]
    if not cats:
        return "კატეგორიების სია ვერ მოიძებნა. (data/processed/found_offers.json შეამოწმე)"
    lines = ["კატეგორიები:"]
    lines.extend([f"- {c}" for c in cats])
    return "\n".join(lines)


def _dominant_category_desc(results: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for r in results or []:
        meta = (r or {}).get("metadata", {}) or {}
        cat = str(meta.get("category_desc") or "").strip()
        if not cat:
            continue
        counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]

class BOGChatbot:

    def __init__(self, retriever, config: Dict[str, Any]) -> None:

        self.retriever = retriever
        self.config = config
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get('temperature', 0.7)
        self.conversation_history = []

        # Keep last retrieval results so follow-ups like “ეს რა პირობებია?” stay on-topic.
        self._last_results: List[Dict[str, Any]] = []
        self._last_user_query: str = ""

        # Lightweight context: remember the last explicit city.
        self._last_city: str = ""

        # Additional context carry-over
        self._last_topic_keywords: List[str] = []
        self._last_benefit_hint: str = ""
        self._last_category_desc: str = ""

        repo_root = Path(__file__).resolve().parents[1]
        self.taxonomy = TaxonomyEngine.from_repo_root(repo_root)

        self._city_labels = _load_city_labels(repo_root)
        self._category_labels = _load_category_labels(repo_root)
        
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

        # Deterministic intent: list available categories.
        if _is_category_list_query(query):
            return _format_category_list(self._category_labels)

        requested_n = _parse_requested_count(query)

        city_in_query = _extract_city(query, self._city_labels)
        if city_in_query:
            self._last_city = city_in_query

        # Extract ALL categories mentioned in query (multi-category support)
        all_categories = _extract_all_categories(query, self._category_labels)
        category_in_query = all_categories[0] if all_categories else ""
        
        if category_in_query:
            self._last_category_desc = category_in_query

        benefit_in_query = _extract_benefit_hint(query)
        if benefit_in_query:
            self._last_benefit_hint = benefit_in_query

        # Follow-up detection: STRICT - only very ambiguous queries are follow-ups
        is_follow_up = (
            _looks_like_follow_up(query, previous_category=self._last_category_desc, category_labels=self._category_labels)
            and bool(self._last_results)
        )

        # Extract topic keywords for context
        topic_kw_limit = int(self.config.get("context_topic_keywords_limit", 5))
        topic_keywords = _extract_topic_keywords(query, city=city_in_query, limit=topic_kw_limit)

        # NEW LOGIC: If NOT a follow-up, clear ALL carried context
        if not is_follow_up:
            # Fresh query - don't carry anything from previous turns
            carry_city = False
            carry_benefit = False
            carry_category = False
            carry_topic = False
        else:
            # IS a follow-up - only carry context if user didn't specify new values
            carry_city = bool(self._last_city) and not city_in_query
            carry_benefit = bool(self._last_benefit_hint) and not benefit_in_query
            carry_category = bool(self._last_category_desc) and not category_in_query
            carry_topic = bool(self._last_topic_keywords) and not topic_keywords

        # 1) Retrieve - MULTI-CATEGORY SUPPORT
        # If multiple categories detected, make separate retrieval for each
        results = []
        
        if len(all_categories) > 1:
            # Multi-category query: retrieve for each category separately
            for cat in all_categories:
                retrieval_query = query
                
                # Build payload filter for this category
                payload_filter: Dict[str, Any] = {"category_desc": cat}
                
                active_city = city_in_query or (self._last_city if (is_follow_up and carry_city) else "")
                if active_city:
                    payload_filter["cities"] = active_city
                
                active_benefit = benefit_in_query or (self._last_benefit_hint if (is_follow_up and carry_benefit) else "")
                payload_filter.update(_benefit_payload_constraints(active_benefit))
                
                # Retrieve for this category
                cat_results = self.retriever.retrieve(
                    retrieval_query,
                    top_k=3,  # Get top 3 per category
                    payload_filter=(payload_filter or None),
                )
                results.extend(cat_results)
        else:
            # Single category or no category - original logic
            retrieval_query = query
            if is_follow_up and self._last_user_query:
                retrieval_query = f"{self._last_user_query}\nFollow-up: {query}".strip()
            elif is_follow_up:
                is_follow_up = False

            if carry_city:
                retrieval_query = f"{retrieval_query}\nCity context: {self._last_city}".strip()

            if carry_city:
                retrieval_query = f"{retrieval_query}\nCity context: {self._last_city}".strip()

            if category_in_query:
                retrieval_query = f"{retrieval_query}\nCategory context: {category_in_query}".strip()
            elif carry_category:
                retrieval_query = f"{retrieval_query}\nCategory context: {self._last_category_desc}".strip()

            if carry_benefit:
                retrieval_query = f"{retrieval_query}\nBenefit context: {self._last_benefit_hint}".strip()

            if carry_topic:
                retrieval_query = f"{retrieval_query}\nTopic context: {' '.join(self._last_topic_keywords)}".strip()

            # Build a structured payload filter for Qdrant (category/city/benefit).
            payload_filter: Dict[str, Any] = {}
            active_category = category_in_query or (self._last_category_desc if (is_follow_up and carry_category) else "")
            if active_category:
                payload_filter["category_desc"] = active_category

            active_city = city_in_query or (self._last_city if (is_follow_up and carry_city) else "")
            if active_city:
                payload_filter["cities"] = active_city

            active_benefit = benefit_in_query or (self._last_benefit_hint if (is_follow_up and carry_benefit) else "")
            payload_filter.update(_benefit_payload_constraints(active_benefit))

            # If the query is just a category (browse), default to showing more.
            is_category_browse = bool(category_in_query) and len(_tokenize(query)) <= 4 and requested_n is None
            if is_category_browse:
                requested_n = int(self.config.get("category_browse_default_n", 10))

            # If user requests N offers, pull more candidates and then limit to N.
            # (Reranking benefits from a bigger candidate pool.)
            top_k_override: Optional[int] = None
            limit_override: Optional[int] = None
            if requested_n is not None:
                default_top_k = int(getattr(self.retriever, "top_k", 5) or 5)
                top_k_override = max(default_top_k, min(50, requested_n * 4))
                limit_override = requested_n

            # Try with structured constraints first; fall back if too strict.
            results = self.retriever.retrieve(
                retrieval_query,
                top_k=top_k_override,
                limit=limit_override,
                payload_filter=(payload_filter or None),
            )
            if not results and payload_filter:
                results = self.retriever.retrieve(retrieval_query, top_k=top_k_override, limit=limit_override)

        available_n = len(results)

        # If the user asks about validity dates, answer deterministically from metadata.
        # This prevents hallucinated start/end dates and stays consistent with the dataset.
        if _is_date_question(query):
            tokens = _tokenize(query)
            is_generic_date_q = len(tokens) <= 4
            target_offer: Optional[Dict[str, Any]] = None

            if is_generic_date_q and self._last_results:
                target_offer = self._last_results[0]
            elif results:
                target_offer = results[0]

            if target_offer is not None:
                answer = _format_offer_period_answer(target_offer)
                self.conversation_history.append({"role": "user", "content": query})
                self.conversation_history.append({"role": "assistant", "content": answer})
                self._last_user_query = query
                self._last_results = list(results) if results else list(self._last_results)
                return answer

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

        user_prompt = PromptTemplates.create_user_message(
            query=query,
            offers=results,
            previous_query=None,  # No follow-up context
            previous_answer=None,
            requested_n=requested_n,
            available_n=available_n,
            carried_city=None,
            carried_category=None,
            carried_benefit=None,
            carried_topic=None,
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Keep minimal conversation history for LLM context only (not for retrieval)
        history_to_keep = int(self.config.get("history_to_keep", 2))
        if history_to_keep > 0 and self.conversation_history:
            messages.extend(self.conversation_history[-history_to_keep:])

        messages.append({"role": "user", "content": user_prompt})

        answer: str

        if provider in {"none", "off", "false"}:
            formatted = PromptTemplates.format_offers(results)
            header = "LLM გამორთულია (--no-llm). ქვემოთ არის ნაპოვნი შეთავაზებები:"
            if requested_n is not None and available_n < requested_n:
                header = f"მხოლოდ {available_n} შეთავაზება იყო ხელმისაწვდომი (მოთხოვნილი: {requested_n})."
            answer = "\n\n".join([header, formatted])

            # Save minimal history + follow-up context even in no-LLM mode.
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            self._last_user_query = query
            self._last_results = list(results)
            return answer

        # 4) If no API key, fall back to deterministic formatted output.
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            formatted = PromptTemplates.format_offers(results)
            header = "OpenAI API key ვერ მოიძებნა (OPENAI_API_KEY). ქვემოთ არის ნაპოვნი შეთავაზებები:"
            if requested_n is not None and available_n < requested_n:
                header = f"მხოლოდ {available_n} შეთავაზება იყო ხელმისაწვდომი (მოთხოვნილი: {requested_n})."
            answer = "\n\n".join([header, formatted])

            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            self._last_user_query = query
            self._last_results = list(results)
            return answer

        if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            formatted = PromptTemplates.format_offers(results)
            header = "Gemini API key ვერ მოიძებნა (GEMINI_API_KEY). ქვემოთ არის ნაპოვნი შეთავაზებები:"
            if requested_n is not None and available_n < requested_n:
                header = f"მხოლოდ {available_n} შეთავაზება იყო ხელმისაწვდომი (მოთხოვნილი: {requested_n})."
            answer = "\n\n".join([header, formatted])

            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            self._last_user_query = query
            self._last_results = list(results)
            return answer

        answer = self._call_llm(messages, provider=provider, model=model)

        # Save minimal history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Update follow-up context after producing a response.
        self._last_user_query = query
        self._last_results = list(results)
        if city_in_query:
            self._last_city = city_in_query

        return answer
    
    def reset_conversation(self) -> None:
        """Clear the conversation history.
        
        Use this to start a fresh conversation without context from previous exchanges.
        """
        self.conversation_history = []
        self._last_results = []
        self._last_user_query = ""
        self._last_city = ""
        self._last_topic_keywords = []
        self._last_benefit_hint = ""
        self._last_category_desc = ""
    
    def _call_llm(self, messages: List[Dict[str, str]], provider: str, model: str) -> str:

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
                    "gemini-3-pro-preview",
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

            # google-generativeai's `resp.text` is a "quick accessor" and may raise
            # when the response contains no valid content parts.
            try:
                quick_text = resp.text  # type: ignore[attr-defined]
            except Exception:
                quick_text = None
            if quick_text:
                return str(quick_text).strip()

            # More defensive extraction across SDK response shapes.
            candidates = getattr(resp, "candidates", None) or []
            extracted_parts: List[str] = []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        extracted_parts.append(str(t))
            if extracted_parts:
                return "".join(extracted_parts).strip()

            # If we got here, Gemini returned no usable text.
            finish_reason = None
            try:
                if candidates:
                    finish_reason = getattr(candidates[0], "finish_reason", None)
            except Exception:
                finish_reason = None
            raise RuntimeError(
                "Gemini-მ ცარიელი პასუხი დააბრუნა (no text parts). "
                f"finish_reason={finish_reason}. სცადე სხვა LLM_MODEL ან შეამოწმე safety/settings."
            )

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
        # Follow-up / context tuning
        "followup_offer_limit": int((os.getenv("FOLLOWUP_OFFER_LIMIT") or "6").strip()),
        "followup_prev_answer_max_chars": int((os.getenv("FOLLOWUP_PREV_ANSWER_MAX_CHARS") or "700").strip()),
        "context_topic_keywords_limit": int((os.getenv("CONTEXT_TOPIC_KEYWORDS_LIMIT") or "5").strip()),
        "category_browse_default_n": int((os.getenv("CATEGORY_BROWSE_DEFAULT_N") or "10").strip()),
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
