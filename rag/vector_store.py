"""
Vector store operations for RAG.

Handles:
- Storing offer embeddings
- Similarity search
- Vector database management (ChromaDB/FAISS)
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional


def _ensure_serializable_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Qdrant payload must be JSON-serializable; coerce a few common types."""
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x for x in v if x is not None]
        else:
            out[k] = str(v)
    return out


class VectorStore:

    def __init__(self, config: Dict[str, Any]):

        self.config = config
        self.vector_store_type = config.get('vector_store', 'qdrant')

        # Qdrant config
        self.qdrant_url = config.get("qdrant_url", "http://localhost:6333")
        self.qdrant_api_key = config.get("qdrant_api_key")
        self.collection_name = config.get("collection_name", "bog_offers")
        self.distance = config.get("qdrant_distance", "COSINE")

        # Networking tuning (useful for Qdrant Cloud / slower connections)
        self.timeout_seconds = float(config.get("qdrant_timeout_seconds", 120))
        self.upsert_batch_size = int(config.get("upsert_batch_size", 64))

        self._client = None
        self._vector_size: Optional[int] = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient  # type: ignore

            self._client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=self.timeout_seconds,
            )
        return self._client
        
    def create_collection(self, collection_name: str | None = None, vector_size: int | None = None):
        """
        Create a new collection in the vector store.
        """
        from qdrant_client.http import models as qm  # type: ignore

        name = collection_name or self.collection_name
        if vector_size is not None:
            self._vector_size = int(vector_size)
        if self._vector_size is None:
            raise ValueError("vector_size must be provided at least once (e.g., from embedding model dimension).")

        dist = getattr(qm.Distance, self.distance.upper(), qm.Distance.COSINE)

        # Recreate=false behavior: if exists, we'll just keep it.
        collections = {c.name for c in self.client.get_collections().collections}
        if name in collections:
            return

        self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=self._vector_size, distance=dist),
        )
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict[str, Any]]):
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
        """
        if not documents:
            return
        if not (len(documents) == len(embeddings) == len(metadata)):
            raise ValueError("documents, embeddings, metadata must have the same length")

        if self._vector_size is None:
            self._vector_size = len(embeddings[0])
        self.create_collection(vector_size=self._vector_size)

        from qdrant_client.http import models as qm  # type: ignore

        points: List[qm.PointStruct] = []
        for idx, (doc, vec, meta) in enumerate(zip(documents, embeddings, metadata)):
            payload = _ensure_serializable_payload({**meta, "text": doc})
            # Use deterministic integer IDs if caller provides one; else fall back to incremental.
            point_id = meta.get("id", idx)
            points.append(qm.PointStruct(id=point_id, vector=vec, payload=payload))

        # Upsert in batches to avoid large request timeouts.
        batch_size = max(1, int(self.upsert_batch_size))
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
            )
    
    def similarity_search(self, query_embedding: List[float], 
                         top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform similarity search.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of (metadata, similarity_score) tuples
        """
        if self._vector_size is None:
            self._vector_size = len(query_embedding)
        self.create_collection(vector_size=self._vector_size)

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        out: List[Tuple[Dict[str, Any], float]] = []
        for h in hits:
            payload = dict(h.payload or {})
            out.append((payload, float(h.score)))
        return out
