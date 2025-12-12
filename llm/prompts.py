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
        # Implementation will be added here
        pass
    
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
        # Implementation will be added here
        pass
