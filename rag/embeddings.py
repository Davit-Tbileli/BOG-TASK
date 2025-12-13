"""Embedding generation for offers data.

Handles:
- Loading multilingual embedding models (BAAI/bge-m3 by default)
- Generating embeddings for offers
- Preprocessing text for embeddings with Georgian language support
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_str(v: Any) -> str:
    """Safely convert any value to a string.
    
    Args:
        v: Any value to convert (can be None, list, tuple, or other).
        
    Returns:
        A cleaned string representation.
    """
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join([_safe_str(x) for x in v if _safe_str(x)])
    return str(v).strip()


class EmbeddingGenerator:

    def __init__(self, model_name: str, normalize_embeddings: bool = True, device: str | None = None):

        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
            normalize_embeddings: Whether to L2-normalize embeddings (recommended for cosine similarity).
            device: Device to run the model on ('cpu', 'cuda', etc.). Auto-detected if None.
            
        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        # Fix for torch.classes path issue on some systems
        try:
            import torch  # type: ignore
            try:
                _ = torch.classes.__path__  # type: ignore[attr-defined]
            except Exception:
                torch.classes.__path__ = []  # type: ignore[attr-defined]
        except ImportError:
            logger.warning("PyTorch not found, some features may not work")

        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors (as lists of floats).
        """
        if not texts:
            return []

        # SentenceTransformers can normalize embeddings !!! recommended for cosine search.
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) > 32,
        )
        return embeddings.tolist()
    
    def preprocess_offer(self, offer: Dict[str, Any]) -> str:
        """Preprocess an offer dictionary into a text representation for embedding.
        
        Creates a dense, multilingual-friendly representation with Georgian field labels.
        Keeps Georgian text as-is since the model (bge-m3) supports it natively.
        
        Args:
            offer: Dictionary containing offer data fields.
            
        Returns:
            A formatted text string suitable for embedding.
        """
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
        """Get the dimension of the embedding vectors produced by the model.
        
        Returns:
            The size of embedding vectors (e.g., 1024 for bge-m3).
        """
        return int(self.model.get_sentence_embedding_dimension())
