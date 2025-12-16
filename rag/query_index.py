from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

from rag.embeddings import EmbeddingGenerator
from rag.retriever import OfferRetriever
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def main() -> int:
    """Run a test query against the vector index.
    
    Returns:
        Exit code (0 for success, 2 for missing query).
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    load_dotenv()

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        logger.error("No query provided")
        print("Provide a query string.")
        return 2

    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    config: Dict[str, Any] = {
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "collection_name": os.getenv("COLLECTION_NAME", "bog_offers"),
        "top_k_results": int(os.getenv("TOP_K", "5")),
        "qdrant_distance": os.getenv("QDRANT_DISTANCE", "COSINE"),
    }

    embedder = EmbeddingGenerator(model_name=embedding_model, normalize_embeddings=True)
    store = VectorStore(config=config)
    retriever = OfferRetriever(embedding_generator=embedder, vector_store=store, config=config)

    # Fast fail if Qdrant isn't reachable.
    try:
        store.client.get_collections()
    except Exception as e:
        raise RuntimeError(
            "Qdrant is not reachable. Check QDRANT_URL/QDRANT_API_KEY and that your cluster is running."
        ) from e

    results = retriever.retrieve(query)

    for i, r in enumerate(results, start=1):
        meta = r.get("metadata", {})
        print(f"\n#{i} score={r.get('score'):.4f}")
        print(f"brand: {meta.get('brand_name')}")
        print(f"title: {meta.get('title')}")
        print(f"category: {meta.get('category_desc')}")
        print(f"url: {meta.get('details_url')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
