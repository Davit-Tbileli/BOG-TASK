"""
Embedding generation for offers data.

Handles:
- Loading multilingual embedding models
- Generating embeddings for offers
- Preprocessing text for embeddings
"""

from typing import List, Dict, Any


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join([_safe_str(x) for x in v if _safe_str(x)])
    return str(v).strip()


class EmbeddingGenerator:
    """
    Generate embeddings for offers data.
    """
    
    def __init__(self, model_name: str, normalize_embeddings: bool = True, device: str | None = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        # Lazy import to avoid import cost for unrelated commands.
        from sentence_transformers import SentenceTransformer  # type: ignore

        # For BGE-M3, SentenceTransformer works well.
        # device: None lets the library auto-pick (cuda if available).
        self.model = SentenceTransformer(model_name, device=device)
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # SentenceTransformers can normalize embeddings; recommended for cosine search.
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) > 32,
        )
        return embeddings.tolist()
    
    def preprocess_offer(self, offer: Dict[str, Any]) -> str:
        """
        Preprocess offer data into text for embedding.
        
        Args:
            offer: Offer dictionary
            
        Returns:
            Preprocessed text string
        """
        # Dense, multilingual-friendly representation.
        # Keep Georgian text as-is; add small field labels to help the model.
        brand = _safe_str(offer.get("brand_name"))
        category = _safe_str(offer.get("category_desc"))
        title = _safe_str(offer.get("title"))
        description = _safe_str(offer.get("description"))
        benef_badge = _safe_str(offer.get("benef_badge"))
        benef_name = _safe_str(offer.get("benef_name"))
        product_code = _safe_str(offer.get("product_code"))
        segment_type = _safe_str(offer.get("segment_type"))
        start_date = _safe_str(offer.get("start_date"))
        end_date = _safe_str(offer.get("end_date"))
        cities = _safe_str(offer.get("cities"))
        details_url = _safe_str(offer.get("details_url"))

        parts = [
            f"ბრენდი: {brand}" if brand else "",
            f"კატეგორია: {category}" if category else "",
            f"სათაური: {title}" if title else "",
            f"სარგებლის ტიპი: {benef_name} {benef_badge}".strip() if (benef_name or benef_badge) else "",
            f"პროდუქტი/ბარათი: {product_code}" if product_code else "",
            f"სეგმენტი: {segment_type}" if segment_type else "",
            f"პერიოდი: {start_date} - {end_date}" if (start_date or end_date) else "",
            f"ქალაქები: {cities}" if cities else "",
            f"აღწერა: {description}" if description else "",
            f"ბმული: {details_url}" if details_url else "",
        ]
        return "\n".join([p for p in parts if p])

    def embedding_dimension(self) -> int:
        """Return embedding dimensionality for the configured model."""
        return int(self.model.get_sentence_embedding_dimension())
