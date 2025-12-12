"""
RAG (Retrieval Augmented Generation) module for offer recommendations.
"""

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .retriever import OfferRetriever

__all__ = ['EmbeddingGenerator', 'VectorStore', 'OfferRetriever']
