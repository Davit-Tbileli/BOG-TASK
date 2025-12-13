"""RAG (Retrieval Augmented Generation) module for offer recommendations.

Provides:
- EmbeddingGenerator: Generate embeddings for offers using multilingual models
- VectorStore: Qdrant-based vector storage and similarity search
- OfferRetriever: Query processing and result ranking
- TaxonomyEngine: Benefit type normalization using taxonomy rules
"""

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .retriever import OfferRetriever
from .taxonomy import TaxonomyEngine

__all__ = [
    'EmbeddingGenerator', 
    'VectorStore', 
    'OfferRetriever',
    'TaxonomyEngine',
]
