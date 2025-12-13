from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _ensure_serializable_payload(payload: Dict[str, Any]) -> Dict[str, Any]:

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

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the vector store.
        
        Args:
            config: Configuration dictionary with keys:
                - qdrant_url: URL to Qdrant instance (required)
                - qdrant_api_key: API key for Qdrant Cloud (optional)
                - collection_name: Name for the collection (default: 'bog_offers')
                - qdrant_distance: Distance metric (default: 'COSINE')
                - qdrant_timeout_seconds: Request timeout (default: 120)
                - upsert_batch_size: Batch size for upserting (default: 64)
        """

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

            logger.info(f"Connecting to Qdrant at {self.qdrant_url}")
            self._client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=self.timeout_seconds,
            )
        return self._client
        
    def create_collection(
        self, 
        collection_name: Optional[str] = None, 
        vector_size: Optional[int] = None
    ) -> None:

        from qdrant_client.http import models as qm  # type: ignore

        name = collection_name or self.collection_name
        if vector_size is not None:
            self._vector_size = int(vector_size)
        if self._vector_size is None:
            raise ValueError(
                "vector_size must be provided at least once "
                "(e.g., from embedding model dimension)."
            )

        dist = getattr(qm.Distance, self.distance.upper(), qm.Distance.COSINE)

        # Check if collection already exists
        collections = {c.name for c in self.client.get_collections().collections}
        if name in collections:
            logger.debug(f"Collection '{name}' already exists, skipping creation")
            return

        logger.info(f"Creating collection '{name}' with vector size {self._vector_size}")
        self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=self._vector_size, distance=dist),
        )
    
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Add documents with their embeddings and metadata to the collection.
        
        Args:
            documents: List of document texts.
            embeddings: List of embedding vectors.
            metadata: List of metadata dictionaries for each document.
            
        Raises:
            ValueError: If lengths of documents, embeddings, and metadata don't match.
        """

        if not documents:
            return
        if not (len(documents) == len(embeddings) == len(metadata)):
            raise ValueError("documents, embeddings, metadata must have the same length")

        if self._vector_size is None:
            self._vector_size = len(embeddings[0])
        self.create_collection(vector_size=self._vector_size)

        from qdrant_client.http import models as qm  

        points: List[qm.PointStruct] = []
        for idx, (doc, vec, meta) in enumerate(zip(documents, embeddings, metadata)):
            payload = _ensure_serializable_payload({**meta, "text": doc})
            # Use deterministic integer IDs if caller provides one; else fall back to incremental.
            point_id = meta.get("id", idx)
            points.append(qm.PointStruct(id=point_id, vector=vec, payload=payload))

        # Upsert in batches to avoid large request timeouts.
        batch_size = max(1, int(self.upsert_batch_size))
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        for i, start in enumerate(range(0, len(points), batch_size)):
            batch = points[start : start + batch_size]
            logger.debug(f"Upserting batch {i + 1}/{total_batches} ({len(batch)} points)")
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        
        logger.info(f"Successfully added {len(documents)} documents to '{self.collection_name}'")
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents using vector similarity.
        
        Args:
            query_embedding: The query embedding vector.
            top_k: Number of top results to return.
            
        Returns:
            List of (payload_dict, score) tuples sorted by similarity.
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
