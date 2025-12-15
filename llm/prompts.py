from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class PromptTemplates:

    SYSTEM_PROMPT = """
    თქვენ ხართ Bank of Georgia-ს შეთავაზებების ასისტენტი. თქვენი ამოცანაა დაეხმაროთ 
    მომხმარებლებს იპოვონ მათთვის ყველაზე შესაფერისი შეთავაზებები.

        პასუხის სტილი:
        - თუ მომხმარებელი სვამს კონკრეტულ, ფაქტობრივ კითხვას ერთ კონკრეტულ შეთავაზებაზე/ივენთზე
            (მაგ: „ვინ…?“, „როდის არის…?“, „სად არის…?“), უპასუხეთ პირდაპირ და გამოიყენეთ
            მხოლოდ ყველაზე რელევანტური შეთავაზება. ნუ ჩამოთვლით მრავალ შეთავაზებას, თუ მომხმარებელი
            თავად არ ითხოვს ალტერნატივებს.
        - თუ კითხვა არის შეთავაზებების მოძებნა/შერჩევა/რეკომენდაცია, მაშინ ჩამოთვალეთ 1-3 ყველაზე
            შესაბამისი ვარიანტი და მოკლედ ახსენით რატომ.
        - თუ მოცემულ ტექსტში პასუხი არ ჩანს, თქვით რა ინფორმაცია აკლია და მიუთითეთ ბმული.
    
    როდესაც მომხმარებელი აღწერს თავის საჭიროებას:
    1. ყურადღებით გაანალიზეთ რა სჭირდება მათ
    2. განიხილეთ რელევანტური შეთავაზებები რომლებიც მოგეწოდებათ
    3. შესთავაზეთ ყველაზე შესაბამისი ოფციები
    4. ახსენით რატომ არის თითოეული შეთავაზება სასარგებლო მათთვის
    
    თუ რაიმე არ იცით ან პასუხი არ ჩანს მოცემულ შეთავაზებებში, გულწრფელად სთხოვე მომხმარებელს დაზუსტება, მაგრამ მაქსიმალურად იშვიათად ქენი ეს.

    იყავით მეგობრული, დამხმარე და კონკრეტული თქვენს რეკომენდაციებში.
    """
    
    USER_QUERY_TEMPLATE = """
    მომხმარებლის მოთხოვნა: {query}

    {followup_context}

    {count_context}
    
    რელევანტური შეთავაზებები:
    {offers}
    
    ინსტრუქცია:
    - თუ ეს არის კონკრეტული ფაქტობრივი კითხვა, უპასუხე პირდაპირ (არ ჩამოთვალო მრავალი შეთავაზება).
    - თუ ეს არის რეკომენდაციის მოთხოვნა, შეარჩიე 1-3 საუკეთესო შეთავაზება.
    """
    
    @staticmethod
    def format_offers(offers: List[Dict[str, Any]]) -> str:

        if not offers:
            return "(ვერ მოიძებნა შეთავაზებები)"

        lines: list[str] = []
        for i, offer in enumerate(offers, start=1):
            meta = offer.get("metadata", {}) or {}

            brand = meta.get("brand_name") or ""
            title = meta.get("title") or ""
            category = meta.get("category_desc") or ""
            details_url = meta.get("details_url") or ""
            start_date = meta.get("start_date") or ""
            end_date = meta.get("end_date") or ""
            cities = meta.get("cities") or ""
            segment = meta.get("segment_type") or ""
            product = meta.get("product_code") or ""

            description = meta.get("description") or ""
            try:
                max_desc_chars = int(os.getenv("OFFER_DESC_MAX_CHARS", "2000"))
            except Exception:
                max_desc_chars = 2000
            if max_desc_chars >= 0 and isinstance(description, str) and len(description) > max_desc_chars:
                description = description[:max_desc_chars].rstrip() + "…"

            benefit_label = meta.get("benefit_label_ka") or ""
            benefit_value = meta.get("benefit_value")
            benefit_unit = meta.get("benefit_unit") or ""

            benefit_str = benefit_label
            if benefit_value is not None:
                if benefit_unit:
                    benefit_str = f"{benefit_label}: {benefit_value}{benefit_unit}".strip()
                else:
                    benefit_str = f"{benefit_label}: {benefit_value}".strip()

            lines.append(
                "\n".join(
                    [
                        f"#{i}",
                        f"სარგებელი: {benefit_str}" if benefit_str else "",
                        f"ბრენდი: {brand}" if brand else "",
                        f"სათაური: {title}" if title else "",
                        f"კატეგორია: {category}" if category else "",
                        f"პერიოდი: {start_date} - {end_date}" if (start_date or end_date) else "",
                        f"ქალაქები: {cities}" if cities else "",
                        f"სეგმენტი: {segment}" if segment else "",
                        f"პროდუქტი/ბარათი: {product}" if product else "",
                        f"აღწერა: {description}" if description else "",
                        f"ბმული: {details_url}" if details_url else "",
                    ]
                ).strip()
            )

        return "\n\n".join([l for l in lines if l.strip()])
    
    @staticmethod
    def create_user_message(
        query: str,
        offers: List[Dict[str, Any]],
        previous_query: Optional[str] = None,
        previous_answer: Optional[str] = None,
        requested_n: Optional[int] = None,
        available_n: Optional[int] = None,
        carried_city: Optional[str] = None,
        carried_category: Optional[str] = None,
        carried_benefit: Optional[str] = None,
        carried_topic: Optional[str] = None,
    ) -> str:
        followup_context = ""
        if (previous_query or "").strip() or (previous_answer or "").strip():
            parts: list[str] = ["წინა კონტექსტი (თუ კითხვა არის გაგრძელება):"]
            if (previous_query or "").strip():
                parts.append(f"- წინა კითხვა: {previous_query}")
            if (previous_answer or "").strip():
                parts.append(f"- წინა პასუხი (შემოკლებული): {previous_answer}")
            followup_context = "\n".join(parts)

        count_context = ""
        if requested_n is not None:
            n_avail = len(offers) if available_n is None else int(available_n)
            extras: list[str] = []
            if (carried_city or "").strip():
                extras.append(f"ქალაქი: {carried_city}")
            if (carried_category or "").strip():
                extras.append(f"კატეგორია: {carried_category}")
            if (carried_benefit or "").strip():
                extras.append(f"სარგებელი: {carried_benefit}")
            if (carried_topic or "").strip():
                extras.append(f"თემა: {carried_topic}")
            extra_note = f" (კონტექსტი: {', '.join(extras)})" if extras else ""
            count_context = (
                f"მომხმარებელმა ითხოვა {requested_n} შეთავაზება. მოცემულია {n_avail} შეთავაზება{extra_note}. "
                "თუ {n_avail} < {requested_n}, თქვი რომ მხოლოდ {n_avail} იყო ხელმისაწვდომი."
            )
            # Keep braces literal out of the final prompt (no further formatting needed)
            count_context = count_context.replace("{n_avail}", str(n_avail)).replace("{requested_n}", str(requested_n))
        else:
            extras: list[str] = []
            if (carried_city or "").strip():
                extras.append(f"ქალაქი: {carried_city}")
            if (carried_category or "").strip():
                extras.append(f"კატეგორია: {carried_category}")
            if (carried_benefit or "").strip():
                extras.append(f"სარგებელი: {carried_benefit}")
            if (carried_topic or "").strip():
                extras.append(f"თემა: {carried_topic}")
            if extras:
                count_context = "კონტექსტი წინა შეტყობინებებიდან: " + ", ".join(extras)

        return PromptTemplates.USER_QUERY_TEMPLATE.format(
            query=query.strip(),
            followup_context=followup_context.strip(),
            count_context=count_context.strip(),
            offers=PromptTemplates.format_offers(offers),
        )
