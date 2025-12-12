"""
System prompts and templates for LLM.

Contains:
- System prompts
- Few-shot examples
- Response templates
"""


class PromptTemplates:
    """
    Manage prompt templates for the chatbot.
    """
    
    SYSTEM_PROMPT = """
    თქვენ ხართ Bank of Georgia-ს შეთავაზებების ასისტენტი. თქვენი ამოცანაა დაეხმაროთ 
    მომხმარებლებს იპოვონ მათთვის ყველაზე შესაფერისი შეთავაზებები.
    
    როდესაც მომხმარებელი აღწერს თავის საჭიროებას:
    1. ყურადღებით გაანალიზეთ რა სჭირდება მათ
    2. განიხილეთ რელევანტური შეთავაზებები რომლებიც მოგეწოდებათ
    3. შესთავაზეთ ყველაზე შესაბამისი ოფციები
    4. ახსენით რატომ არის თითოეული შეთავაზება სასარგებლო მათთვის
    
    იყავით მეგობრული, დამხმარე და კონკრეტული თქვენს რეკომენდაციებში.
    """
    
    USER_QUERY_TEMPLATE = """
    მომხმარებლის მოთხოვნა: {query}
    
    რელევანტური შეთავაზებები:
    {offers}
    
    გთხოვთ შესთავაზოთ პერსონალიზებული რეკომენდაციები და ახსენით რატომ არის 
    თითოეული შეთავაზება შესაფერისი ამ მომხმარებლისთვის.
    """
    
    @staticmethod
    def format_offers(offers: list) -> str:
        """
        Format offers for inclusion in prompt.
        
        Args:
            offers: List of offer dictionaries
            
        Returns:
            Formatted string of offers
        """
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
                        f"ბმული: {details_url}" if details_url else "",
                    ]
                ).strip()
            )

        return "\n\n".join([l for l in lines if l.strip()])
    
    @staticmethod
    def create_user_message(query: str, offers: list) -> str:
        """
        Create user message with query and retrieved offers.
        
        Args:
            query: User query
            offers: Retrieved offers
            
        Returns:
            Formatted message
        """
        return PromptTemplates.USER_QUERY_TEMPLATE.format(
            query=query.strip(),
            offers=PromptTemplates.format_offers(offers),
        )
