from __future__ import annotations

import re
from typing import List, Optional, Dict, Any


# Regex patterns
_COUNT_RE = re.compile(r"\b(\d{1,2})\b")
_WORD_RE = re.compile(r"[0-9A-Za-z\u10A0-\u10FF]+", re.UNICODE)

_DATE_QUESTION_CUES = (
    # Georgian
    "როდემდე",
    "სადამდე",
    "როდიდან",
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

# Semantic mappings: Georgian query words → category_desc value
_CATEGORY_KEYWORDS = {
    "გართობა და კულტურა": ["გართობ", "კულტურ", "entertainment"],
    "შოპინგი": ["შოპინგ", "shopping", "მაღაზი", "ტანსაცმ", "ფეხსაცმ", "სამკაული"],
    "კვება": ["კვებ", "რესტორან", "კაფე"],
    "დასვენება": ["დასვენებ", "relax", "spa", "სასტუმრო", "ჰოტელ", "hotel"],
    "თავის მოვლა": ["თავის მოვლა", "სილამაზე", "beauty"],
    "მოგზაურობა": ["მოგზაურობ", "travel"],
    "განათლება": ["განათლებ", "education"],
    "სახლი და ოჯახი": ["სახლ", "ოჯახ", "home"],
    "ავტომობილები": ["ავტომობილ", "მანქან", "car"],
    "ტექნიკა": ["ტექნიკ", "tech"],
}


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


def _looks_like_follow_up(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False

    # Short date questions like "როდემდეა?" should be treated as follow-ups.
    if _is_date_question(q) and len(_tokenize(q)) <= 4:
        return True

    # Ordinal references are always follow-ups (referring to numbered items from previous response)
    if _extract_ordinal_position(q) is not None:
        return True

    cues = (
        "ეს",
        "ამ",
        "იმ",
        "შესახებ",
        "წინა",
        "კიდევ",
        "დამატებით",
        "დეტალ",
        "მეტი",
        "უფრო",
        "that",
        "this",
        "it",
        "what about",
        "tell me more",
        "more details",
    )
    return any(c in q for c in cues)


def _is_date_question(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(cue in q for cue in _DATE_QUESTION_CUES)


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
    hint = (benefit_hint or "").strip().upper()
    if hint == "CASHBACK_PERCENT":
        return {"benef_name": "CASHBACK"}
    if hint == "DISCOUNT_PERCENT":
        return {"benef_name": "DISCOUNT"}
    if hint == "POINTS_MULTIPLIER":
        return {"benef_name": "MR"}
    return {}


def _extract_ordinal_position(query: str) -> Optional[int]:
    """Extract ordinal position from query (first=0, second=1, third=2, etc.).
    
    Returns the 0-based index, or None if no ordinal found.
    """
    q = (query or "").strip().lower()
    if not q:
        return None
    
    # Georgian ordinals (checking for stem to handle all case forms)
    ordinals = {
        "პირველ": 0,  # პირველი, პირველზე, პირველს...
        "მეორ": 1,    # მეორე, მეორეზე, მეორეს...
        "მესამ": 2,   # მესამე, მესამეზე, მესამეს...
        "მეოთხ": 3,
        "მეხუთ": 4,
        # English
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
    }
    
    for ord_word, index in ordinals.items():
        if ord_word in q:
            return index
    
    return None
