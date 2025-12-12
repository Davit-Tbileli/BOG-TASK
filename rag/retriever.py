"""
Retrieval logic for finding relevant offers.

Handles:
- Query processing
- Retrieving relevant offers from vector store
- Ranking and filtering results
"""

from typing import List, Dict, Any


class OfferRetriever:
    """
    Retrieve relevant offers based on user queries.
    """
    
    def __init__(self, embedding_generator, vector_store, config: Dict[str, Any]):
        """
        Initialize retriever.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
            config: Configuration dictionary
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.config = config
        self.top_k = config.get('top_k_results', 5)
        
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant offers for a query.
        
        Args:
            query: User query string
            
        Returns:
            List of relevant offer dictionaries
        """
        if not query or not query.strip():
            return []

        query_vec = self.embedding_generator.generate_embeddings([query.strip()])[0]
        results = self.vector_store.similarity_search(query_vec, top_k=self.top_k)

        offers: List[Dict[str, Any]] = []
        for payload, score in results:
            # Keep original payload; extract text separately for prompt formatting.
            offers.append(
                {
                    "score": score,
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                }
            )

        return self.rerank_results(query=query, results=offers)
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results based on additional criteria.
        
        Args:
            query: User query string
            results: Initial retrieval results
            
        Returns:
            Reranked list of offers
        """
        # Placeholder: later we can add:
        # - rule-based boosts (exact brand/category/city matches)
        # - date validity boost
        # - cross-encoder reranking (BGE reranker) if needed
        return results
