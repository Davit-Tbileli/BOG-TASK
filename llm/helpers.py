"""Helper utilities for BOG chatbot.

This module contains data loading and response formatting utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def _load_city_labels(repo_root: Path) -> List[str]:
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


def _format_category_list(categories: List[str]) -> str:
    cats = [c.strip() for c in (categories or []) if str(c).strip()]
    if not cats:
        return "კატეგორიების სია ვერ მოიძებნა. (data/processed/found_offers.json შეამოწმე)"
    lines = ["კატეგორიები:"]
    lines.extend([f"- {c}" for c in cats])
    return "\n".join(lines)

