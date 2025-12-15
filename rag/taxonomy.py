from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class NormalizedBenefit:
    benefit_type_id: str
    benefit_label_ka: str
    value: Optional[float] = None
    value_unit: Optional[str] = None
    rule_id: Optional[str] = None
    source: Optional[str] = None


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _coerce_number(value: str, cast: str | None) -> float:
    value = value.strip()
    # Allow commas as decimal separators if they ever appear.
    value = value.replace(",", ".")
    if cast == "int":
        return float(int(float(value)))
    if cast == "float":
        return float(value)
    # Default: try float.
    return float(value)


class TaxonomyEngine:
    def __init__(self, taxonomy_path: Path):
        self.taxonomy_path = taxonomy_path
        self._taxonomy: Dict[str, Any] = {}
        self._benefit_type_by_id: Dict[str, Dict[str, Any]] = {}
        self._annotation_by_url: Dict[str, Dict[str, Any]] = {}
        self._compiled_rules: List[Dict[str, Any]] = []

        self._load()

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "TaxonomyEngine":
        return cls(repo_root / "data" / "processed" / "taxonomy.json")

    def _load(self) -> None:
        with self.taxonomy_path.open("r", encoding="utf-8") as f:
            self._taxonomy = json.load(f)

        self._benefit_type_by_id = {b["id"]: b for b in self._taxonomy.get("benefit_types", []) if b.get("id")}

        self._annotation_by_url = {}
        for ann in self._taxonomy.get("offer_annotations", []) or []:
            url = ann.get("details_url")
            if url:
                self._annotation_by_url[url] = ann

        rules = list(self._taxonomy.get("rules", []) or [])
        # Highest priority first
        rules.sort(key=lambda r: int(r.get("priority", 0)), reverse=True)

        compiled: List[Dict[str, Any]] = []
        for rule in rules:
            if not rule.get("enabled", True):
                continue

            when = rule.get("when", {}) or {}
            title_regex = when.get("title_regex")
            desc_regex = when.get("description_regex")

            compiled_rule = dict(rule)
            compiled_rule["_title_re"] = re.compile(title_regex) if title_regex else None
            compiled_rule["_desc_re"] = re.compile(desc_regex) if desc_regex else None
            compiled.append(compiled_rule)

        self._compiled_rules = compiled

    def _benefit_label(self, benefit_type_id: str) -> str:
        b = self._benefit_type_by_id.get(benefit_type_id) or {}
        return _safe_str(b.get("label_ka")) or benefit_type_id

    def normalize_offer(self, offer: Dict[str, Any]) -> Optional[NormalizedBenefit]:
        """Return normalized benefit info for an offer payload/metadata."""

        details_url = _safe_str(offer.get("details_url"))
        title = _safe_str(offer.get("title"))
        description = _safe_str(offer.get("description"))
        benef_name = _safe_str(offer.get("benef_name"))
        benef_badge = _safe_str(offer.get("benef_badge"))

        # 1) Manual per-offer override
        if details_url and details_url in self._annotation_by_url:
            ann = self._annotation_by_url[details_url]
            benefit_type_id = _safe_str(ann.get("benefit_type_id"))
            if benefit_type_id:
                unit = self._benefit_type_by_id.get(benefit_type_id, {}).get("unit")
                return NormalizedBenefit(
                    benefit_type_id=benefit_type_id,
                    benefit_label_ka=self._benefit_label(benefit_type_id),
                    value=ann.get("value"),
                    value_unit=_safe_str(ann.get("value_unit") or unit) or None,
                    rule_id=None,
                    source="offer_annotations",
                )

        # 2) Rule-based matching
        for rule in self._compiled_rules:
            when = rule.get("when", {}) or {}

            benef_badge_in = when.get("benef_badge_in")
            if benef_badge_in and benef_badge not in set(map(str, benef_badge_in)):
                continue

            benef_name_in = when.get("benef_name_in")
            if benef_name_in and benef_name not in set(map(str, benef_name_in)):
                continue

            title_re = rule.get("_title_re")
            if title_re is not None and not title_re.search(title):
                continue

            desc_re = rule.get("_desc_re")
            if desc_re is not None and not desc_re.search(description):
                continue

            then = rule.get("then", {}) or {}
            benefit_type_id = _safe_str(then.get("benefit_type_id"))
            if not benefit_type_id:
                continue

            value: Optional[float] = None
            value_unit: Optional[str] = None

            value_extract = then.get("value_extract")
            if isinstance(value_extract, dict):
                src = _safe_str(value_extract.get("source") or "title")
                text = title if src == "title" else description
                regex = _safe_str(value_extract.get("regex"))
                group = int(value_extract.get("group", 1))
                cast = value_extract.get("cast")

                if regex:
                    m = re.search(regex, text)
                    if m:
                        raw = m.group(group)
                        try:
                            value = _coerce_number(raw, cast)
                        except Exception:
                            value = None

            value_unit = _safe_str(self._benefit_type_by_id.get(benefit_type_id, {}).get("unit")) or None

            return NormalizedBenefit(
                benefit_type_id=benefit_type_id,
                benefit_label_ka=self._benefit_label(benefit_type_id),
                value=value,
                value_unit=value_unit,
                rule_id=_safe_str(rule.get("id")) or None,
                source="rules",
            )

        # 3) Heuristic fallback (keeps system useful even if taxonomy rules are minimal)
        # We keep this conservative: only obvious patterns.
        title_l = title.lower()
        desc_l = description.lower()

        def _has_percent(text: str) -> Optional[float]:
            m = re.search(r"(?i)(\d{1,3})\s*%", text)
            if not m:
                return None
            try:
                return float(int(m.group(1)))
            except Exception:
                return None

        pct = _has_percent(title) or _has_percent(description)

        # Cashback percent: detect Georgian 'ქეშბექ' or English 'cashback'
        if pct is not None and ("ქეშბექ" in title_l or "ქეშბექ" in desc_l or "cashback" in title_l or "cashback" in desc_l):
            benefit_type_id = "CASHBACK_PERCENT"
            return NormalizedBenefit(
                benefit_type_id=benefit_type_id,
                benefit_label_ka=self._benefit_label(benefit_type_id),
                value=pct,
                value_unit="%",
                rule_id=None,
                source="heuristic",
            )

        # Discount percent: if offer meta already says DISCOUNT/% but rules didn't catch for some reason.
        if pct is not None and benef_name == "DISCOUNT" and benef_badge == "%":
            benefit_type_id = "DISCOUNT_PERCENT"
            return NormalizedBenefit(
                benefit_type_id=benefit_type_id,
                benefit_label_ka=self._benefit_label(benefit_type_id),
                value=pct,
                value_unit="%",
                rule_id=None,
                source="heuristic",
            )

        return None
